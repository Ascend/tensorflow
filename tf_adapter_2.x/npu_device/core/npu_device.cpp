/* Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <utility>
#include <future>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"

#include "absl/algorithm/container.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/platform/refcount.h"

#include "npu_custom_kernel.h"
#include "npu_device.h"
#include "npu_dp.h"
#include "npu_env.h"
#include "npu_logger.h"
#include "npu_micros.h"
#include "npu_parser.h"
#include "npu_unwrap.h"
#include "npu_utils.h"

#include "framework/common/ge_inner_error_codes.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_factory.h"

using Format = ge::Format;
const static uint64_t kInvalidGeGraphId = -1;

namespace {
template <typename T, typename DT>
class NpuHostFixedAllocator : public tensorflow::Allocator, public tensorflow::core::RefCounted  {
 public:
  static tensorflow::Allocator *Create(std::unique_ptr<T, DT> ptr) {
    return new (std::nothrow) NpuHostFixedAllocator(std::move(ptr));
  }

 private:
  explicit NpuHostFixedAllocator(std::unique_ptr<T, DT> ptr) : ptr_(std::move(ptr)) {
    DLOG() << "Zero copied ge tensor " << reinterpret_cast<uintptr_t>(ptr_.get());
  }
  ~NpuHostFixedAllocator() override {
    DLOG() << "Release zero copied ge tensor " << reinterpret_cast<uintptr_t>(ptr_.get());
  };
  std::string Name() override { return "NpuHostFixedAllocator"; }
  void *AllocateRaw(size_t alignment, size_t num_bytes) override { return ptr_.get(); }
  void DeallocateRaw(void *ptr) override { Unref(); }
  std::unique_ptr<T, DT> ptr_;
};

size_t RemoveRedundantHcomControlEdges(tensorflow::Graph *graph) {
  const static std::string kHcomType = "HcomAllReduce";
  std::vector<tensorflow::Edge *> edges_to_remove;
  for (auto edge : graph->edges()) {
    if (edge->IsControlEdge() && (edge->src()->type_string() == kHcomType || edge->dst()->type_string() == kHcomType)) {
      edges_to_remove.push_back(edge);
    }
  }
  for (auto edge : edges_to_remove) {
    graph->RemoveEdge(edge);
  }
  return edges_to_remove.size();
}

bool IsGraphNeedLoop(const tensorflow::Graph *graph, tensorflow::Node **key) {
  *key = nullptr;
  for (auto node : graph->op_nodes()) {
    if (node->IsWhileNode()) {
      if (*key != nullptr) {
        return false;
      }
      *key = node;
    }
  }
  if (*key == nullptr) {
    DLOG() << "Skip check as no while node in graph";
    return false;
  }
  size_t reserved_nums = 0;
  const std::function<void(tensorflow::Node *)> &enter = [&reserved_nums](tensorflow::Node *node) {
    if (node->IsOp()) {
      reserved_nums++;
    }
  };
  tensorflow::ReverseDFSFrom(*graph, {*key}, enter, {}, {}, {});
  DLOG() << "Reserved nodes " << reserved_nums << " vs. totally " << graph->num_op_nodes();
  return static_cast<int>(reserved_nums) == graph->num_op_nodes();
}

tensorflow::FunctionDefLibrary CollectGraphSubGraphs(const tensorflow::GraphDef &gdef,
                                                     tensorflow::FunctionLibraryDefinition *lib_def) {
  tensorflow::FunctionDefLibrary fdef_lib;

  std::unordered_set<std::string> related_function_names;
  std::queue<const tensorflow::FunctionDef *> related_functions;
  for (const auto &n : gdef.node()) {
    for (const auto &attr : n.attr()) {
      if (attr.second.has_func() && related_function_names.insert(attr.second.func().name()).second) {
        const auto *f = lib_def->Find(attr.second.func().name());
        if (f != nullptr) {
          *fdef_lib.add_function() = *f;
          related_functions.push(f);
        } else {
          LOG(ERROR) << "Function " << attr.second.func().name() << " not found";
        }
      }
    }
  }

  while (!related_functions.empty()) {
    const auto *f = related_functions.front();
    related_functions.pop();
    for (const auto &n : f->node_def()) {
      for (const auto &attr : n.attr()) {
        if (attr.second.has_func() && related_function_names.insert(attr.second.func().name()).second) {
          const auto *f_inner = lib_def->Find(attr.second.func().name());
          if (f_inner != nullptr) {
            *fdef_lib.add_function() = *f_inner;
            related_functions.push(f_inner);
          } else {
            LOG(ERROR) << "Function " << attr.second.func().name() << " not found";
          }
        }
      }
    }
  }
  return fdef_lib;
}

class OptimizeStageGraphDumper {
 public:
  explicit OptimizeStageGraphDumper(const std::string &graph) : graph_(graph), counter_(0) {}
  void Dump(const std::string &stage, const tensorflow::GraphDef &graph_def) {
    WriteTextProto(tensorflow::Env::Default(), graph_ + "." + std::to_string(counter_++) + "." + stage + ".pbtxt",
                   graph_def);
  }

  void DumpWithSubGraphs(const std::string &stage, const tensorflow::GraphDef &graph_def,
                         tensorflow::FunctionLibraryDefinition *lib_def) {
    tensorflow::GraphDef copied_graph_def = graph_def;
    *copied_graph_def.mutable_library() = CollectGraphSubGraphs(graph_def, lib_def);
    Dump(stage, copied_graph_def);
  }

 private:
  std::string graph_;
  int counter_;
};

}  // namespace

void NpuDevice::CreateIteratorProvider(TFE_Context *context, const tensorflow::Tensor *tensor,
                                       std::vector<int> device_ids, TF_Status *status) {
  auto resource = tensor->scalar<tensorflow::ResourceHandle>()();
  TensorPartialShapes shapes;
  TensorDataTypes types;
  NPU_CTX_REQUIRES_OK(status, GetMirroredIteratorShapesAndTypes(resource, shapes, types));
  auto dp_provider =
    IteratorResourceProvider::GetFunctionDef(resource.name(), std::move(device_ids), shapes, types, status);
  if (TF_GetCode(status) != TF_OK) return;

  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  NPU_CTX_REQUIRES_OK(status, lib_def->AddFunctionDef(dp_provider));
  tensorflow::ProcessFunctionLibraryRuntime *pflr = npu::UnwrapCtx(context)->pflr();
  tensorflow::FunctionLibraryRuntime *flr = pflr->GetFLR(underlying_device);
  tensorflow::FunctionLibraryRuntime::Handle f_handle;
  NPU_CTX_REQUIRES_OK(status, flr->Instantiate(dp_provider.signature().name(), tensorflow::AttrSlice{}, &f_handle));

  tensorflow::CancellationManager *cancel_manager = CancellationManager();
  auto consume_func = [flr, f_handle, cancel_manager](tensorflow::Tensor tensor, int64_t nums) -> tensorflow::Status {
    std::vector<tensorflow::Tensor> get_next_outputs;
    tensorflow::FunctionLibraryRuntime::Options options;
    options.cancellation_manager = cancel_manager;
    return flr->RunSync(options, f_handle, {std::move(tensor), tensorflow::Tensor(tensorflow::int64(nums))},
                        &get_next_outputs);
  };
  auto destroy_func = [resource, flr, f_handle]() -> tensorflow::Status {
    LOG(INFO) << "Stopping iterator resource provider for " << resource.name();
    return flr->ReleaseHandle(f_handle);
  };

  auto provider = std::make_shared<IteratorResourceProvider>(resource.name(), consume_func, destroy_func);
  LOG(INFO) << "Iterator resource provider for " << resource.name() << " created";

  NPU_CTX_REQUIRES(status, provider != nullptr,
                   tensorflow::errors::Internal("Failed create iterator reosurce provider for ", resource.name()));

  iterator_providers_[resource] = provider;

  if (kDumpExecutionDetail || kDumpGraph) {
    std::unique_ptr<tensorflow::FunctionBody> fbody;
    tensorflow::AttrSlice attr_slice;
    tensorflow::FunctionDefToBodyHelper(dp_provider, attr_slice, lib_def, &fbody);
    std::string file_name = "dp_provider_" + resource.name() + ".pbtxt";
    WriteTextProto(tensorflow::Env::Default(), file_name, fbody->graph->ToGraphDefDebug());
  }
}

std::string NpuDevice::CreateDevice(const char *name, int device_index,
                                    const std::map<std::string, std::string> &session_options, NpuDevice **device) {
  auto *ge_session = new (std::nothrow) ge::Session(session_options);
  if (ge_session == nullptr) {
    return "Failed init graph engine: create new session failed";
  }

  *device = new (std::nothrow) NpuDevice();
  if (*device == nullptr) {
    return "Failed create new npu device instance";
  }
  (*device)->device_id = device_index;
  (*device)->device_name = name;
  (*device)->underlying_device = "/job:localhost/replica:0/task:0/device:CPU:0";
  (*device)->ge_session_ = ge_session;
  (*device)->cancellation_manager_ = std::make_unique<tensorflow::CancellationManager>();
  return "";
}

void NpuDevice::ReleaseResource() {
  std::vector<std::future<void>> thread_guarder;
  for (auto &iterator_provider : iterator_providers_) {
    auto provider = iterator_provider.second;
    thread_guarder.emplace_back(std::async([provider]() { provider->Destroy(); }));
  }

  DLOG() << "Start cancel all uncompleted async call";
  CancellationManager()->StartCancel();
}

void NpuDevice::DeleteDevice(void *device) {
  DLOG() << "Start destroy npu device instance";
  if (device == nullptr) {
    return;
  }
  auto npu_device = reinterpret_cast<NpuDevice *>(device);
  delete npu_device->ge_session_;
  delete npu_device;
}

tensorflow::Status NpuDevice::ValidateResourcePlacement(const char *op_name, int num_inputs, TFE_TensorHandle **inputs,
                                                        bool &cpu_resource) {
  bool has_cpu = false;
  int cpu_index = 0;
  bool has_npu = false;
  int npu_index = 0;
  for (int i = 0; i < num_inputs; i++) {
    auto data_type = npu::UnwrapHandle(inputs[i])->DataType();
    if (data_type == tensorflow::DT_RESOURCE) {
      const tensorflow::Tensor *tensor;
      (void)npu::UnwrapTensor(inputs[i], &tensor);
      if (IsNpuTensorHandle(npu::UnwrapHandle(inputs[i]))) {
        has_npu = true;
        npu_index = i;
        if (has_cpu) {
          const tensorflow::Tensor *cpu_tensor;
          (void)npu::UnwrapTensor(inputs[cpu_index], &cpu_tensor);
          return tensorflow::errors::InvalidArgument(
            op_name, " resource input ", i, " ", tensor->scalar<tensorflow::ResourceHandle>()().name(),
            " on NPU but resource input ", cpu_index, " ", cpu_tensor->scalar<tensorflow::ResourceHandle>()().name(),
            " on CPU");
        }
      } else if (!Mirrored(tensor->scalar<tensorflow::ResourceHandle>()())) {
        has_cpu = true;
        cpu_index = i;
        if (has_npu) {
          const tensorflow::Tensor *npu_tensor;
          (void)npu::UnwrapTensor(inputs[npu_index], &npu_tensor);
          return tensorflow::errors::InvalidArgument(
            op_name, " resource input ", i, " ", tensor->scalar<tensorflow::ResourceHandle>()().name(),
            " on CPU but resource input ", npu_index, " ", npu_tensor->scalar<tensorflow::ResourceHandle>()().name(),
            " on NPU");
        }
      }
    }
  }
  cpu_resource = has_cpu;
  return tensorflow::Status::OK();
}

tensorflow::Status NpuDevice::ValidateInput(const char *op_name, int num_inputs, TFE_TensorHandle **inputs) {
  for (int i = 0; i < num_inputs; i++) {
    auto data_type = npu::UnwrapHandle(inputs[i])->DataType();
    if (data_type == tensorflow::DT_RESOURCE) {
      const tensorflow::Tensor *tensor;
      NPU_REQUIRES_OK(npu::UnwrapTensor(inputs[i], &tensor));
      if (!IsNpuTensorHandle(npu::UnwrapHandle(inputs[i]))) {
        if (!Mirrored(tensor->scalar<tensorflow::ResourceHandle>()())) {
          tensorflow::Status status;
          std::string src_name = npu::UnwrapHandle(inputs[i])->DeviceName(&status);
          if (!status.ok()) {
            src_name = status.ToString();
          }
          return tensorflow::errors::Unimplemented("Op ", op_name, " input ", i, " resource from ", src_name);
        } else {
          DLOG() << "Op" << op_name << " input " << i << " resource mirrored from "
                 << tensor->scalar<tensorflow::ResourceHandle>()().DebugString();
        }
      }
    } else if (!tensorflow::DataTypeCanUseMemcpy(data_type)) {
      return tensorflow::errors::Unimplemented("Op ", op_name, " input ", i, " unsupported type ",
                                               tensorflow::DataTypeString(data_type));
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status NpuDevice::ValidateOutput(const char *op_name, const TensorDataTypes &data_types) {
  for (size_t i = 0; i < data_types.size(); i++) {
    auto data_type = data_types[i];
    if (data_type == tensorflow::DT_RESOURCE) {
      if (!SupportedResourceGenerator(op_name)) {
        return tensorflow::errors::Unimplemented("Op ", op_name, " unsupported resource generator by NPU");
      }
    } else if (!tensorflow::DataTypeCanUseMemcpy(data_type)) {
      return tensorflow::errors::Unimplemented("Op ", op_name, " output ", i, " unsupported type ",
                                               tensorflow::DataTypeString(data_type));
    }
  }
  return tensorflow::Status::OK();
}

void NpuDevice::PruneFunction(const tensorflow::FunctionDef &fdef, tensorflow::Graph *g, bool keep_signature) {
  std::unordered_set<tensorflow::StringPiece, tensorflow::StringPieceHasher> control_ret_nodes;
  for (const auto &control_ret : fdef.control_ret()) {
    control_ret_nodes.insert(control_ret.second);
  }

  std::unordered_set<const tensorflow::Node *> nodes;
  for (auto n : g->nodes()) {
    if (n->IsControlFlow() || n->op_def().is_stateful() ||
        (control_ret_nodes.find(n->name()) != control_ret_nodes.end())) {
      if (n->type_string() == "VarHandleOp" || n->type_string() == "IteratorV2") {
        continue;
      }
      if (!keep_signature) {
        if (n->IsArg()) {
          continue;
        }
        if (n->IsRetval() && n->attrs().Find("T")->type() == tensorflow::DT_RESOURCE) {
          continue;
        }
      }
      nodes.insert(n);
    }
  }
  bool changed = PruneForReverseReachability(g, std::move(nodes));
  if (changed) {
    FixupSourceAndSinkEdges(g);
  }
}

void NpuDevice::FixGraphArgRetvalIndex(tensorflow::Graph *graph) {
  std::map<int, tensorflow::Node *> indexed_args;
  std::map<int, tensorflow::Node *> indexed_retvals;
  for (auto node : graph->nodes()) {
    if (node->IsArg()) {
      indexed_args[node->attrs().Find("index")->i()] = node;
    }
    if (node->IsRetval()) {
      indexed_retvals[node->attrs().Find("index")->i()] = node;
    }
  }
  int current_arg_index = 0;
  for (auto indexed_arg : indexed_args) {
    indexed_arg.second->AddAttr("index", current_arg_index++);
  }

  int current_retval_index = 0;
  for (auto indexed_retval : indexed_retvals) {
    indexed_retval.second->AddAttr("index", current_retval_index++);
  }
}

tensorflow::Status NpuDevice::TransResourceInput2GraphNode(
  TFE_Context *context, tensorflow::Graph *graph, int num_inputs, TFE_TensorHandle **inputs,
  std::map<int, std::shared_ptr<IteratorResourceProvider>> &dependent_host_resources) {
  (void)RemoveRedundantHcomControlEdges(graph);

  std::set<int> arg_is_variable;
  std::set<int> arg_is_iterator;

  std::map<int, tensorflow::ResourceHandle> arg_resource_handles;

  VecTensorDataTypes arg_handle_dtyes(num_inputs);
  VecTensorPartialShapes arg_handle_shapes(num_inputs);

  for (int i = 0; i < num_inputs; i++) {
    if (inputs[i] == nullptr) {
      continue;
    };
    const tensorflow::Tensor *tensor;
    NPU_REQUIRES_OK(npu::UnwrapTensor(inputs[i], &tensor));
    if (tensor->dtype() == tensorflow::DT_RESOURCE) {
      auto handle = tensor->flat<tensorflow::ResourceHandle>()(0);
      arg_resource_handles[i] = handle;
      if (MirroredIterator(handle)) {
        GetMirroredIteratorShapesAndTypes(handle, arg_handle_shapes[i], arg_handle_dtyes[i]);
        arg_is_iterator.insert(i);
      } else {
        const auto &dtypes_and_shapes = handle.dtypes_and_shapes();
        for (auto &dtype_and_shape : dtypes_and_shapes) {
          arg_handle_dtyes[i].push_back(dtype_and_shape.dtype);
          arg_handle_shapes[i].push_back(dtype_and_shape.shape);
        }
        arg_is_variable.insert(i);
      }
    }
  }

  std::map<tensorflow::Node *, tensorflow::Node *> arg_substitutes;
  for (auto node : graph->op_nodes()) {
    if (node->IsArg()) {
      auto index = node->attrs().Find("index")->i();
      if (arg_is_iterator.count(index)) {
        NPU_REQUIRES_OK(tensorflow::NodeBuilder(WrapResourceName(arg_resource_handles[index].name()), "IteratorV2")
                          .Attr("container", arg_resource_handles[index].container())
                          .Attr("shared_name", arg_resource_handles[index].name())
                          .Attr("output_types", arg_handle_dtyes[index])
                          .Attr("output_shapes", arg_handle_shapes[index])
                          .Attr("_arg_name", node->name())
                          .Attr("_arg_index", int(index))
                          .Finalize(graph, &arg_substitutes[node]));

      } else if (arg_is_variable.count(index)) {
        NPU_REQUIRES_OK(tensorflow::NodeBuilder(WrapResourceName(arg_resource_handles[index].name()), "VarHandleOp")
                          .Attr("container", arg_resource_handles[index].container())
                          .Attr("shared_name", arg_resource_handles[index].name())
                          .Attr("dtype", arg_handle_dtyes[index][0])
                          .Attr("shape", arg_handle_shapes[index][0])
                          .Attr("_arg_name", node->name())
                          .Attr("_arg_index", int(index))
                          .Finalize(graph, &arg_substitutes[node]));
      }
    }
  }

  // 这里需要把涉及的function的resource输入也一并替换了
  std::vector<tensorflow::Node *> nodes_to_remove;
  std::vector<tensorflow::Node *> control_flow_nodes;
  for (auto node : graph->op_nodes()) {
    if (node->IsRetval() && node->input_type(0) == tensorflow::DT_RESOURCE) {
      if (kDumpExecutionDetail) {
        const tensorflow::Edge *edge;
        NPU_REQUIRES_OK(node->input_edge(0, &edge));
        LOG(INFO) << "Retval " << node->def().DebugString() << " from " << edge->src()->name() << ":"
                  << edge->src_output() << " will be removed";
      }

      nodes_to_remove.push_back(node);
      continue;
    }
    if (node->IsIfNode() || node->IsCaseNode() || node->IsWhileNode() || node->IsPartitionedCall()) {
      DLOG() << "Start pruning control flow op " << node->def().DebugString();
      std::string func_input_name = node->IsPartitionedCall() ? "args" : "input";
      bool need_trans_resource = false;
      for (auto edge : node->in_edges()) {
        if (edge->src()->IsArg() && arg_substitutes.find(edge->src()) != arg_substitutes.end()) {
          DLOG() << node->name() << " input " << edge->src()->attrs().Find("index")->i() << " is resource arg";
          need_trans_resource = true;
        }
      }
      if (!need_trans_resource) continue;

      control_flow_nodes.push_back(node);

      tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
      const tensorflow::OpRegistrationData *op_reg_data;
      NPU_REQUIRES_OK(lib_def->LookUp(node->type_string(), &op_reg_data));
      int func_input_start = 0;
      int func_input_end = 0;
      for (const auto &in_arg : op_reg_data->op_def.input_arg()) {
        func_input_start = func_input_end;
        if (in_arg.type_list_attr().empty()) {
          func_input_end++;
        } else {
          func_input_end += node->attrs().Find(in_arg.type_list_attr())->list().type_size();
        }
        DLOG() << node->name() << " input arg " << in_arg.name() << " range [" << func_input_start << ", "
               << func_input_end << ")";
        if (in_arg.name() == func_input_name) {
          break;
        }
      }

      std::vector<TFE_TensorHandle *> func_inputs;
      for (int i = func_input_start; i < func_input_end; i++) {
        const tensorflow::Edge *edge;
        NPU_REQUIRES_OK(node->input_edge(i, &edge));
        if (edge->src()->IsArg() && arg_substitutes.find(edge->src()) != arg_substitutes.end()) {
          func_inputs.push_back(inputs[edge->src()->attrs().Find("index")->i()]);
        } else {
          func_inputs.push_back(nullptr);
        }
      }

      for (auto &attr : node->attrs()) {
        if (attr.second.has_func()) {
          std::string func_name =
            node->type_string() + "_" + attr.first + "_" + attr.second.func().name() + "_" + std::to_string(node->id());
          const tensorflow::FunctionDef *fdef = lib_def->Find(attr.second.func().name());
          std::unique_ptr<tensorflow::FunctionBody> fbody;
          FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice{}, lib_def, &fbody);
          std::map<int, std::shared_ptr<IteratorResourceProvider>> unused_host_resources;
          NPU_REQUIRES_OK(TransResourceInput2GraphNode(context, fbody->graph, func_inputs.size(), func_inputs.data(),
                                                       unused_host_resources));

          // Arg节点可能会被优化掉，因而需要重新排列index
          std::vector<int> remain_indexes;
          for (auto n : fbody->graph->nodes()) {
            if (n->IsArg()) {
              remain_indexes.push_back(n->attrs().Find("index")->i());
            }
          }
          FixGraphArgRetvalIndex(fbody->graph);
          DLOG() << func_name << " remained input index [0-" << func_inputs.size() << ") -> "
                 << VecToString(remain_indexes);

          tensorflow::FunctionDef optimized_fdef;
          auto lookup = [&fdef](const tensorflow::Node *node) -> absl::optional<std::string> {
            for (const auto &control_ret : fdef->control_ret()) {
              if (control_ret.second == node->name()) {
                return absl::make_optional(node->name());
              }
            }
            return absl::nullopt;
          };
          NPU_REQUIRES_OK(tensorflow::GraphToFunctionDef(*fbody->graph, func_name, lookup, &optimized_fdef));
          NPU_REQUIRES_OK(lib_def->AddFunctionDef(optimized_fdef));
          DLOG() << "Change " << node->name() << " attr " << attr.first << " func name " << attr.second.func().name()
                 << " to " << func_name;
          const_cast<tensorflow::AttrValue *>(node->attrs().Find(attr.first))->mutable_func()->set_name(func_name);
        }
      }
    }

    std::vector<const tensorflow::Edge *> edges;
    for (auto edge : node->in_edges()) {
      edges.emplace_back(edge);
    }  // You can never modify and iterator an EdgeSet
    for (auto edge : edges) {
      if (edge->src()->IsArg()) {
        auto iter = arg_substitutes.find(edge->src());
        if (iter != arg_substitutes.end()) {
          int index = edge->src()->attrs().Find("index")->i();
          if (arg_is_iterator.count(index)) {
            auto provider = iterator_providers_.find(arg_resource_handles[index]);
            NPU_REQUIRES(
              provider != iterator_providers_.end(),
              tensorflow::errors::Internal("Resource provider for ", arg_resource_handles[index].name(), " not found"));
            dependent_host_resources[index] = provider->second;
          }
          graph->AddEdge(iter->second, 0, node, edge->dst_input());
          graph->RemoveEdge(edge);
        }
      }
    }
  }

  for (auto node : control_flow_nodes) {
    if (node->IsWhileNode() || node->IsIfNode() || node->IsCaseNode() || node->IsPartitionedCall()) {
      tensorflow::NodeDef ndef = node->def();
      if (node->IsWhileNode()) {
        int removed_nums = 0;
        for (int i = 0; i < node->num_inputs(); i++) {
          if (node->input_type(i) == tensorflow::DT_RESOURCE) {
            int index = i - removed_nums;
            removed_nums++;

            ndef.mutable_input()->erase(ndef.mutable_input()->begin() + index);

            auto type = ndef.mutable_attr()->at("T").mutable_list()->mutable_type();
            type->erase(type->begin() + index);

            auto shape = ndef.mutable_attr()->at("output_shapes").mutable_list()->mutable_shape();
            shape->erase(shape->begin() + index);
          }
        }
      } else if (node->IsIfNode() || node->IsCaseNode() || node->IsPartitionedCall()) {
        int removed_nums = 0;
        int arg_start_index = node->IsPartitionedCall() ? 0 : 1;
        for (int i = arg_start_index; i < node->num_inputs(); i++) {
          if (node->input_type(i) == tensorflow::DT_RESOURCE) {
            int index = i - removed_nums;
            removed_nums++;

            ndef.mutable_input()->erase(ndef.mutable_input()->begin() + index);

            auto type = ndef.mutable_attr()->at("Tin").mutable_list()->mutable_type();
            type->erase(type->begin() + index - arg_start_index);
          }
        }
      }
      DLOG() << "Pruned control flow op " << ndef.DebugString();
      tensorflow::Status status;
      auto pruned_node = graph->AddNode(ndef, &status);
      NPU_REQUIRES_OK(status);
      int pruned_input_index = 0;
      for (auto edge : node->in_edges()) {
        if (edge->IsControlEdge()) {
          graph->AddControlEdge(edge->src(), pruned_node);
          DLOG() << "Add ctrl edge from " << edge->src()->name() << " to " << pruned_node->name();
        }
      }
      for (int i = 0; i < node->num_inputs(); i++) {
        const tensorflow::Edge *edge;
        NPU_REQUIRES_OK(node->input_edge(i, &edge));
        if (node->input_type(i) != tensorflow::DT_RESOURCE) {
          graph->AddEdge(edge->src(), edge->src_output(), pruned_node, pruned_input_index++);
          DLOG() << "Add edge from " << edge->src()->name() << ":" << edge->src_output() << " to "
                 << pruned_node->name() << ":" << pruned_input_index - 1;
        }
      }
      for (auto edge : node->out_edges()) {
        graph->AddEdge(pruned_node, edge->src_output(), edge->dst(), edge->dst_input());
        DLOG() << "Add edge from " << pruned_node->name() << ":" << edge->src_output() << " to " << edge->dst()->name()
               << ":" << edge->dst_input();
      }
      graph->RemoveNode(node);
    }
  }
  for (auto node : nodes_to_remove) {
    graph->RemoveNode(node);
  }
  for (auto arg_substitute : arg_substitutes) {
    graph->RemoveNode(arg_substitute.first);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status NpuDevice::MarkGraphNodeInOutDesc(TFE_Context *context, tensorflow::Graph *graph, int num_inputs,
                                                     TFE_TensorHandle **inputs) {
  tensorflow::ShapeRefiner shape_refiner(graph->versions(), npu::UnwrapCtx(context)->FuncLibDef());
  VecTensorShapes arg_shapes;
  VecTensorDataTypes arg_handle_dtyes;
  VecTensorPartialShapes arg_handle_shapes;
  for (int i = 0; i < num_inputs; i++) {
    const tensorflow::Tensor *tensor;
    NPU_REQUIRES_OK(npu::UnwrapTensor(inputs[i], &tensor));
    arg_shapes.push_back({tensor->shape()});
    TensorDataTypes handle_dtyes;
    TensorPartialShapes handle_shapes;
    if (tensor->dtype() == tensorflow::DT_RESOURCE) {
      auto handle = tensor->flat<tensorflow::ResourceHandle>()(0);
      const auto &dtypes_and_shapes = handle.dtypes_and_shapes();
      for (auto &dtype_and_shape : dtypes_and_shapes) {
        handle_dtyes.push_back(dtype_and_shape.dtype);
        handle_shapes.push_back(dtype_and_shape.shape);
      }
    }
    arg_handle_dtyes.push_back(handle_dtyes);
    arg_handle_shapes.push_back(handle_shapes);
  }

  auto node_shape_inference_lambda = [&shape_refiner, num_inputs, inputs, &arg_shapes, &arg_handle_dtyes,
                                      &arg_handle_shapes](tensorflow::Node *node) {
    AssembleOpDef(node);
    if (node->IsArg() && node->attrs().Find("index")) {
      auto index = node->attrs().Find("index")->i();
      if (index < num_inputs && !node->attrs().Find("_output_shapes")) {
        node->AddAttr("_output_shapes", arg_shapes[index]);
      }
      if (index < num_inputs && npu::UnwrapHandle(inputs[index])->DataType() == tensorflow::DT_RESOURCE) {
        if (!node->attrs().Find("_handle_shapes")) {
          node->AddAttr("_handle_shapes", arg_handle_shapes[index]);
        }
        if (!node->attrs().Find("_handle_dtypes")) {
          node->AddAttr("_handle_dtypes", arg_handle_dtyes[index]);
        }
      }
    }
    auto status = shape_refiner.AddNode(node);
    if (!status.ok()) {
      LOG(INFO) << "  " << node->name() << "[" << node->type_string() << "] Skip infer " << status.error_message();
      return;
    }
    auto node_ctx = shape_refiner.GetContext(node);

    DLOG() << "Shape of node " << node->DebugString();
    if (kDumpExecutionDetail) {
      TensorDataTypes input_types;
      tensorflow::InputTypesForNode(node->def(), node->op_def(), &input_types);
      TensorPartialShapes input_shapes;
      for (int i = 0; i < node_ctx->num_inputs(); ++i) {
        tensorflow::TensorShapeProto proto;
        node_ctx->ShapeHandleToProto(node_ctx->input(i), &proto);
        input_shapes.emplace_back(proto);
        LOG(INFO) << "    input " << i << ": " << tensorflow::DataTypeString(input_types[i])
                  << node_ctx->DebugString(node_ctx->input(i));
      }
    }

    TensorDataTypes input_types;
    TensorDataTypes output_types;
    tensorflow::InOutTypesForNode(node->def(), node->op_def(), &input_types, &output_types);

    if (!input_types.empty()) {
      tensorflow::AttrValue input_desc_attrs;
      bool input_desc_incomplete = false;
      for (int i = 0; i < node->num_inputs(); i++) {
        const tensorflow::Edge *edge = nullptr;
        status = node->input_edge(i, &edge);
        if (!status.ok()) {
          LOG(ERROR) << status.ToString();
          return;
        }

        auto input_attr = edge->src()->attrs().Find(kOutputDesc);
        if (input_attr == nullptr) {
          input_desc_incomplete = true;
          LOG(WARNING) << node->DebugString() << " input node " << edge->src()->DebugString()
                       << " has no desc for output " << edge->src_output();
          break;
        }
        *input_desc_attrs.mutable_list()->add_func() =
          edge->src()->attrs().Find(kOutputDesc)->list().func(edge->src_output());
      }
      if (!input_desc_incomplete) {
        node->AddAttr(kInputDesc, input_desc_attrs);
      } else {
        TensorPartialShapes input_shapes;
        for (int i = 0; i < node_ctx->num_inputs(); ++i) {
          tensorflow::TensorShapeProto proto;
          node_ctx->ShapeHandleToProto(node_ctx->input(i), &proto);
          input_shapes.emplace_back(proto);
        }
        AssembleInputDesc(input_shapes, input_types, node);
      }
    }

    if (!output_types.empty()) {
      TensorPartialShapes output_shapes;
      for (int i = 0; i < node_ctx->num_outputs(); ++i) {
        tensorflow::TensorShapeProto proto;
        node_ctx->ShapeHandleToProto(node_ctx->output(i), &proto);
        output_shapes.emplace_back(proto);
        DLOG() << "    output " << i << ": " << tensorflow::DataTypeString(output_types[i])
               << node_ctx->DebugString(node_ctx->output(i));
      }
      AssembleOutputDesc(output_shapes, output_types, node);
    }
  };
  tensorflow::ReverseDFS(*graph, {}, node_shape_inference_lambda);
  return tensorflow::Status::OK();
}

TFE_TensorHandle *NpuDevice::NewDeviceTensorHandle(TFE_Context *context, Format fmt,
                                                   const tensorflow::TensorShape &shape, tensorflow::DataType type,
                                                   TF_Status *status) {
  NpuManagedBuffer *npu_managed_buffer;
  NPU_CTX_REQUIRES_OK_RETURN(status, NpuManagedBuffer::Create(fmt, shape, type, &npu_managed_buffer), nullptr);
  std::vector<int64_t> dims;
  for (auto dim_size : shape.dim_sizes()) {
    dims.emplace_back(dim_size);
  }
  return TFE_NewTensorHandleFromDeviceMemory(context, device_name.c_str(), static_cast<TF_DataType>(type), dims.data(),
                                             dims.size(), npu_managed_buffer, sizeof(npu_managed_buffer),
                                             &NpuManagedBufferDeallocator, nullptr, status);
}

TFE_TensorHandle *NpuDevice::NewDeviceResourceHandle(TFE_Context *context, const tensorflow::TensorShape &shape,
                                                     TF_Status *status) {
  tensorflow::Tensor tensor(tensorflow::DT_RESOURCE, shape);
  tensorflow::CustomDevice *custom_device = nullptr;
  NPU_CTX_REQUIRES_RETURN(status, npu::UnwrapCtx(context)->FindCustomDeviceFromName(device_name, &custom_device),
                          tensorflow::errors::Internal("No custom device registered with name ", device_name), nullptr);
  return tensorflow::wrap(
    tensorflow::TensorHandle::CreateLocalHandle(std::move(tensor), custom_device, npu::UnwrapCtx(context)));
}

TFE_TensorHandle *NpuDevice::CopyTensorD2H(TFE_Context *context, TFE_TensorHandle *tensor, TF_Status *status) {
  const tensorflow::Tensor *npu_tensor;
  NPU_CTX_REQUIRES_OK_RETURN(status, npu::UnwrapTensor(tensor, &npu_tensor), nullptr);

  if (npu_tensor->dtype() == tensorflow::DT_RESOURCE) {
    tensorflow::ResourceHandle handle = npu_tensor->scalar<tensorflow::ResourceHandle>()();
    status->status =
      tensorflow::errors::Internal("Resources ", handle.DebugString(), " cannot be copied across devices[NPU->CPU]");
    return nullptr;
  }

  const tensorflow::Tensor *local_tensor;
  TFE_TensorHandle *local_handle = tensorflow::wrap(
    tensorflow::TensorHandle::CreateLocalHandle(tensorflow::Tensor(npu_tensor->dtype(), npu_tensor->shape())));
  NPU_CTX_REQUIRES_RETURN(status, local_handle != nullptr, tensorflow::errors::Internal("Failed create local handle"),
                          nullptr);
  NPU_CTX_REQUIRES_OK_RETURN(status, npu::UnwrapTensor(local_handle, &local_tensor), nullptr);
  NPU_CTX_REQUIRES_OK_RETURN(status, npu::Unwrap<NpuManagedBuffer>(npu_tensor)->AssembleTo(local_tensor), local_handle);
  return local_handle;
}

TFE_TensorHandle *NpuDevice::CopyTensorH2D(TFE_Context *context, TFE_TensorHandle *tensor, TF_Status *status) {
  return CopyTensorH2D(context, tensor, Format::FORMAT_ND, status);
}

TFE_TensorHandle *NpuDevice::CopyTensorH2D(TFE_Context *context, TFE_TensorHandle *tensor, Format fmt,
                                           TF_Status *status) {
  TFE_TensorHandle *local_handle = tensor;
  ScopeTensorHandleDeleter scope_handle_deleter;
  if (!IsCpuTensorHandle(npu::UnwrapHandle(tensor))) {
    local_handle = TFE_TensorHandleCopyToDevice(tensor, context, underlying_device.c_str(), status);
    scope_handle_deleter.Guard(local_handle);
  }

  if (TF_GetCode(status) != TF_OK) return nullptr;
  const tensorflow::Tensor *local_tensor = nullptr;
  NPU_CTX_REQUIRES_OK_RETURN(status, npu::UnwrapTensor(local_handle, &local_tensor), nullptr);
  if (local_tensor->dtype() == tensorflow::DT_RESOURCE) {
    tensorflow::ResourceHandle handle = local_tensor->scalar<tensorflow::ResourceHandle>()();
    status->status =
      tensorflow::errors::Internal("Resources ", handle.DebugString(), " cannot be copied across devices[CPU->NPU]");
    return nullptr;
  }

  TFE_TensorHandle *npu_handle =
    NewDeviceTensorHandle(context, fmt, local_tensor->shape(), local_tensor->dtype(), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  const tensorflow::Tensor *npu_tensor = nullptr;

  NPU_CTX_REQUIRES_OK_RETURN(status, npu::UnwrapTensor(npu_handle, &npu_tensor), nullptr);
  NPU_CTX_REQUIRES_OK_RETURN(status, npu::Unwrap<NpuManagedBuffer>(npu_tensor)->AssembleFrom(local_tensor), npu_handle);
  return npu_handle;
}

tensorflow::Status NpuDevice::InferShape(TFE_Context *context, const tensorflow::OpRegistrationData *op_reg_data,
                                         const tensorflow::NodeDef &ndef, int num_inputs, TFE_TensorHandle **inputs,
                                         TensorPartialShapes &shapes, bool &requested_input_value) {
  requested_input_value = false;
  NPU_REQUIRES(op_reg_data->shape_inference_fn,
               tensorflow::errors::Unimplemented("No infer shape function registered for op ", ndef.op()));

  tensorflow::shape_inference::InferenceContext ic(TF_GRAPH_DEF_VERSION, ndef, op_reg_data->op_def,
                                                   std::vector<tensorflow::shape_inference::ShapeHandle>(num_inputs),
                                                   {}, {}, {});
  NPU_REQUIRES_OK(ic.construction_status());
  for (int i = 0; i < num_inputs; i++) {
    auto input = npu::UnwrapHandle(inputs[i]);
    tensorflow::shape_inference::ShapeHandle shape;
    NPU_REQUIRES_OK(input->InferenceShape(&ic, &shape));
    ic.SetInput(i, shape);
  }

  for (int i = 0; i < num_inputs; i++) {
    auto input = inputs[i];
    if (npu::UnwrapHandle(input)->DataType() == tensorflow::DT_RESOURCE) {
      const tensorflow::Tensor *tensor;
      NPU_REQUIRES_OK(npu::UnwrapTensor(input, &tensor));
      auto handle = tensor->flat<tensorflow::ResourceHandle>()(0);
      const auto &dtypes_and_shapes = handle.dtypes_and_shapes();
      std::vector<tensorflow::shape_inference::ShapeAndType> inference_shapes_and_types;
      for (auto &dtype_and_shape : dtypes_and_shapes) {
        std::vector<tensorflow::shape_inference::DimensionHandle> dims_handle(dtype_and_shape.shape.dims());
        for (size_t j = 0; j < dims_handle.size(); j++) {
          dims_handle[j] = ic.MakeDim(dtype_and_shape.shape.dim_size(j));
        }
        inference_shapes_and_types.emplace_back(ic.MakeShape(dims_handle), dtype_and_shape.dtype);
      }
      ic.set_input_handle_shapes_and_types(i, inference_shapes_and_types);
      requested_input_value = true;
    }
  }
  // We need to feed the input tensors. TensorFlow performs inference based on the input shape for the first time.
  // If the shape function of an operator depends on the value of the input tensor, the shape function is marked for the
  // first time and the actual tensor value is used for inference for the second time.
  NPU_REQUIRES_OK(ic.Run(op_reg_data->shape_inference_fn));

  std::vector<const tensorflow::Tensor *> input_tensors;
  input_tensors.resize(num_inputs);
  ScopeTensorHandleDeleter scope_handle_deleter;
  bool input_requested = false;
  for (int i = 0; i < num_inputs; i++) {
    auto input = inputs[i];
    if (ic.requested_input_tensor(i)) {  // If requested, this must be a normal tensor
      if (IsNpuTensorHandle(npu::UnwrapHandle(input))) {
        auto s = TF_NewStatus();
        if (s == nullptr) {
          continue;
        }
        input = CopyTensorD2H(context, input, s);
        if (TF_GetCode(s) != TF_OK) {
          TF_DeleteStatus(s);
          continue;
        }
        DLOG() << "Copying " << ndef.op() << " input:" << i << " from NPU to CPU for infer shape";
        scope_handle_deleter.Guard(input);
      }
      const tensorflow::Tensor *tensor;
      NPU_REQUIRES_OK(npu::UnwrapTensor(input, &tensor));
      input_tensors[i] = tensor;
      input_requested = true;
      requested_input_value = true;
    }
  }
  if (input_requested) {
    ic.set_input_tensors(input_tensors);
    NPU_REQUIRES_OK(ic.Run(op_reg_data->shape_inference_fn));
  }

  for (int i = 0; i < ic.num_outputs(); i++) {
    shapes.emplace_back(tensorflow::PartialTensorShape());
    tensorflow::shape_inference::ShapeHandle shape_handle = ic.output(i);
    auto num_dims = ic.Rank(shape_handle);
    std::vector<tensorflow::int64> dims;
    if (num_dims == tensorflow::shape_inference::InferenceContext::kUnknownRank) {
      continue;
    }
    for (auto j = 0; j < num_dims; ++j) {
      dims.emplace_back(ic.Value(ic.Dim(shape_handle, j)));
    }
    NPU_REQUIRES_OK(tensorflow::PartialTensorShape::MakePartialShape(dims.data(), num_dims, &shapes[i]));
  }
  return tensorflow::Status::OK();
}

void NpuDevice::GetOrCreateSpec(TFE_Context *context, const char *op_name, const TFE_OpAttrs *attributes,
                                int num_inputs, TFE_TensorHandle **inputs, std::shared_ptr<const npu::TaskSpec> *spec,
                                TF_Status *s) {
  tensorflow::NodeDef ndef;
  ndef.set_op(op_name);
  tensorflow::unwrap(attributes)->FillAttrValueMap(ndef.mutable_attr());
  bool request_shape = false;
  GetCachedTaskSpec(ndef, spec, request_shape);
  if (request_shape) {
    TensorShapes input_shapes;
    input_shapes.resize(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
      NPU_CTX_REQUIRES_OK(s, npu::UnwrapHandle(inputs[i])->Shape(&input_shapes[i]));
    }
    GetCachedTaskSpec(ndef, input_shapes, spec);
  }
  if (*spec != nullptr) {
    DLOG() << "Found cached task spec for " << op_name;
    return;
  }
  DLOG() << "No cached task spec for " << op_name << ", start create and cache";
  // 上面校验resource源头的，都是不可以cache的，因为resource可能在多次调用中来自不同的设备，下面的部分是可以cache的
  // NodeDef保存节点的属性，比较重要的，对于单算子，则会保存T属性，表达输入输出的type<T>
  // OpRegistrationData保存算子的IR注册信息，对于单算子，则和RegisterOp传递的信息一致，对于function，则是确定了输入的dataType的
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  const tensorflow::OpRegistrationData *op_reg_data;
  NPU_CTX_REQUIRES_OK(s, lib_def->LookUp(op_name, &op_reg_data));
  bool is_function_op = op_reg_data->is_function_op;
  // 判断当前算子是否是NPU Device声明支持的算子
  if (!is_function_op && !Supported(op_name)) {
    *spec = CacheOpSpec(op_name, op_reg_data, ndef, {}, tensorflow::strings::StrCat("Op unsupported by NPU"));
    return;
  }
  // 这里获取输出的dataType，对于常规算子，通过NodeDef的T属性确定，对于function op，则是在ret上自带
  TensorDataTypes data_types;
  NPU_CTX_REQUIRES_OK(s, tensorflow::OutputTypesForNode(ndef, op_reg_data->op_def, &data_types));
  // 如果输出的dataType不支持，或者不是支持的ResourceGenerator，则fallback
  tensorflow::Status compat_status = ValidateOutput(op_name, data_types);
  if (!compat_status.ok()) {
    if (is_function_op) {
      *spec = CacheFuncSpec(op_name, op_reg_data, ndef, kInvalidGeGraphId, {}, {}, {}, compat_status.error_message());
      return;
    } else {
      *spec = CacheOpSpec(op_name, op_reg_data, ndef, {}, compat_status.error_message());
      return;
    }
  }
  // 需要进行函数算子的图优化，然后再判断NPU是否兼容
  if (is_function_op) {  // 对function_op，进行图优化，并固定缓存，如果需要fallback，也在spec中记录fallback的原因
    const tensorflow::FunctionDef *fdef = lib_def->Find(op_name);
    std::unique_ptr<tensorflow::Graph> optimize_graph = std::make_unique<tensorflow::Graph>(lib_def);
    std::unique_ptr<tensorflow::FunctionBody> fbody;
    tensorflow::ProcessFunctionLibraryRuntime *pflr = npu::UnwrapCtx(context)->pflr();
    tensorflow::FunctionLibraryRuntime *flr = pflr->GetFLR("/job:localhost/replica:0/task:0/device:CPU:0");
    FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice(&ndef.attr()), lib_def, &fbody);
    CopyGraph(*fbody->graph, optimize_graph.get());

    OptimizeStageGraphDumper graph_dumper(op_name);

    if (kDumpExecutionDetail || kDumpGraph) {
      graph_dumper.DumpWithSubGraphs("before_optimize", optimize_graph->ToGraphDefDebug(), lib_def);
    }

    tensorflow::OptimizeGraph(flr, &optimize_graph);

    if (kDumpExecutionDetail || kDumpGraph) {
      graph_dumper.DumpWithSubGraphs("after_optimize", optimize_graph->ToGraphDefDebug(), lib_def);
    }

    std::map<int, std::shared_ptr<IteratorResourceProvider>> dependent_host_resources;
    NPU_CTX_REQUIRES_OK(
      s, TransResourceInput2GraphNode(context, optimize_graph.get(), num_inputs, inputs, dependent_host_resources));

    if (kDumpExecutionDetail || kDumpGraph) {
      graph_dumper.DumpWithSubGraphs("after_replace_resource_inputs", optimize_graph->ToGraphDefDebug(), lib_def);
    }

    PruneFunction(*fdef, optimize_graph.get());

    DLOG() << "NPU Start inferring shape for function node " << op_name;

    std::vector<int> remain_indexes;
    std::vector<TFE_TensorHandle *> pruned_inputs;
    for (auto node : optimize_graph->nodes()) {
      if (node->IsArg()) {
        auto index = node->attrs().Find("index")->i();
        remain_indexes.push_back(index);
        pruned_inputs.push_back(inputs[index]);
      }
    }

    MarkGraphNodeInOutDesc(context, optimize_graph.get(), num_inputs, inputs);
    FixGraphArgRetvalIndex(optimize_graph.get());  // Arg节点可能会被优化掉，因而需要重新排列index，并且prune输入

    if (kDumpExecutionDetail || kDumpGraph) {
      graph_dumper.DumpWithSubGraphs("after_mark_node_shape", optimize_graph->ToGraphDefDebug(), lib_def);
    }

    DLOG() << op_name << " remained input index (0-" << num_inputs - 1 << ") -> " << VecToString(remain_indexes);
    auto lambda = [remain_indexes](int num_inputs, TFE_TensorHandle **inputs, std::vector<TFE_TensorHandle *> &pruned) {
      for (auto index : remain_indexes) {
        pruned.push_back(inputs[index]);
      }
    };

    // 对于function节点，可以将resource的输入NPU兼容性作为缓存项目，校验输入是否被NPU支持，如果类型不支持，或者是CPU的Resouce类型，则不支持
    // 如果是单算子，则不能缓存，需要在每次dev->Run的时候，校验单算子资源输入的兼容性
    auto status = ValidateInput(op_name, pruned_inputs.size(), pruned_inputs.data());
    if (!status.ok()) {
      *spec = CacheFuncSpec(op_name, op_reg_data, ndef, kInvalidGeGraphId, {}, {}, {}, status.error_message());
    } else {
      uint64_t graph_id = kInvalidGeGraphId;
      bool loop = false;
      auto loop_graph = std::make_unique<tensorflow::GraphDef>();
      if (!dependent_host_resources.empty()) {
        NPU_CTX_REQUIRES_OK(s, GetAutoLoopGraph(context, optimize_graph.get(), pruned_inputs.size(),
                                                pruned_inputs.data(), loop, loop_graph.get()));
      } else {
        DLOG() << "Skip trans " << op_name << " to loop graph as no iterator resource dependencies";
        optimize_graph->ToGraphDef(loop_graph.get());
      }

      LOG(INFO) << "Graph " << op_name << " can loop: " << (loop ? "true" : "false");
      if (kDumpExecutionDetail || kDumpGraph) {
        graph_dumper.DumpWithSubGraphs((loop ? "LOOP" : "NON-LOOP"), *loop_graph, lib_def);
      }
      graph_id = AddGeGraphInner(context, NextUUID(), op_name, *loop_graph, loop, s);
      if (TF_GetCode(s) != TF_OK) return;
      *spec = CacheFuncSpec(op_name, op_reg_data, ndef, graph_id, std::move(loop_graph), lambda,
                            dependent_host_resources, "");
      reinterpret_cast<const npu::FuncSpec *>(spec->get())->SetNeedLoop(loop);
    }
    return;
  } else {
    // 进行inferShape，输出可能是unknown shape，所以使用partial shape
    TensorShapes input_shapes;
    input_shapes.resize(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
      NPU_CTX_REQUIRES_OK(s, npu::UnwrapHandle(inputs[i])->Shape(&input_shapes[i]));
    }
    TensorPartialShapes partial_shapes;
    bool requested_input_value = false;
    if (!data_types.empty()) {
      DLOG() << "Infer shape for op " << op_name;
      tensorflow::Status infer_status =
        InferShape(context, op_reg_data, ndef, num_inputs, inputs, partial_shapes, requested_input_value);
      // 如果inferShape失败，或者期望输出数量不对，则fallback回CPU，因为CPU的计算并不依赖inferShape
      if (!infer_status.ok()) {
        *spec = CacheOpSpec(op_name, op_reg_data, ndef, input_shapes, partial_shapes, infer_status.error_message());
        return;
      }
    } else {
      DLOG() << "Skip infer shape for non-output op " << op_name;
    }
    const std::string reason = ValidateInput(op_name, num_inputs, inputs).error_message();
    if (requested_input_value) {
      *spec = CacheOpSpec(op_name, op_reg_data, ndef, input_shapes, reason);
    } else {
      *spec = CacheOpSpec(op_name, op_reg_data, ndef, input_shapes, partial_shapes, reason);
    }
    return;
  }
}

void NpuDevice::FallbackCPU(TFE_Context *context, const char *op_name, const TFE_OpAttrs *attributes, int num_inputs,
                            TFE_TensorHandle **inputs, int *num_outputs, TFE_TensorHandle **outputs,
                            TF_Status *status) {
  DLOG() << "Start fallback executing " << op_name << " by " << underlying_device;
  TFE_Op *op(TFE_NewOp(context, op_name, status));
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpAddAttrs(op, attributes);
  TFE_OpSetDevice(op, underlying_device.c_str(), status);
  ScopeTensorHandleDeleter scope_handle_deleter;
  for (int j = 0; j < num_inputs; ++j) {
    TFE_TensorHandle *input = inputs[j];
    if (IsNpuTensorHandle(npu::UnwrapHandle(input))) {
      input = CopyTensorD2H(context, input, status);  // 创建完成计数为1
      scope_handle_deleter.Guard(input);
      if (TF_GetCode(status) != TF_OK) return;
    }
    if (kDumpExecutionDetail) {
      const tensorflow::Tensor *tensor = nullptr;
      npu::UnwrapTensor(input, &tensor);
      LOG(INFO) << "    input " << j << "  " << tensor->DebugString();
    }
    TFE_OpAddInput(op, input, status);  // add完成计数为2
    if (TF_GetCode(status) != TF_OK) return;
  }

  std::vector<TFE_TensorHandle *> op_outputs(*num_outputs);
  TFE_Execute(op, op_outputs.data(), num_outputs, status);
  TFE_DeleteOp(op);
  if (TF_GetCode(status) != TF_OK) return;
  for (int i = 0; i < *num_outputs; ++i) {
    outputs[i] = op_outputs[i];
  }

  NpuFallbackHookFunc *hook = nullptr;
  if (CustomKernelRegistry::Instance().GetFallbackHookFunc(op_name, &hook)) {
    (*hook)(context, this, op_name, attributes, num_inputs, inputs, *num_outputs, outputs, status);
    if (TF_GetCode(status) != TF_OK) return;
  }
}

void NpuDevice::FallbackCPU(TFE_Context *context, const npu::OpSpec *spec, int num_inputs, TFE_TensorHandle **inputs,
                            int *num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  tensorflow::AttrBuilder attr_builder;
  attr_builder.Reset(spec->Op().c_str());
  attr_builder.BuildNodeDef();
  auto attrs = spec->NodeDef().attr();
  for (auto &attr : attrs) {
    attr_builder.Set(attr.first, attr.second);
  }
  FallbackCPU(context, spec->Op().c_str(), tensorflow::wrap(&attr_builder), num_inputs, inputs, num_outputs, outputs,
              status);
}

void NpuDevice::Execute(const TFE_Op *op, int *num_outputs, TFE_TensorHandle **outputs, TF_Status *s) {
  auto context = TFE_OpGetContext(op, s);
  if (TF_GetCode(s) != TF_OK) {
    return;
  }
  auto num_inputs = TFE_OpGetFlatInputCount(op, s);
  if (TF_GetCode(s) != TF_OK) {
    return;
  }
  std::vector<TFE_TensorHandle *> inputs;
  for (int i = 0; i < num_inputs; i++) {
    inputs.push_back(TFE_OpGetFlatInput(op, i, s));
    if (TF_GetCode(s) != TF_OK) {
      return;
    }
  }
  auto op_name = TFE_OpGetName(op, s);
  if (TF_GetCode(s) != TF_OK) {
    return;
  }
  auto attributes = TFE_OpGetAttrs(op);
  DLOG() << "NPU Start executing " << op_name;
  // 如果存在一个算子的输入来自多个设备的情况，需要直接报错
  bool cpu_resource = false;
  NPU_CTX_REQUIRES_OK(s, ValidateResourcePlacement(op_name, num_inputs, inputs.data(), cpu_resource));
  // 如果算子有resource输入来自CPU，则必须fallback CPU
  if (cpu_resource) {
    DLOG() << "NPU Executing " << op_name << " fallback[input resource from cpu]";
    FallbackCPU(context, op_name, attributes, inputs.size(), inputs.data(), num_outputs, outputs, s);
    return;
  }
  std::shared_ptr<const npu::TaskSpec> spec;
  GetOrCreateSpec(context, op_name, attributes, inputs.size(), inputs.data(), &spec, s);
  if (TF_GetCode(s) != TF_OK) {
    return;
  }
  DLOG() << "NPU Executing " << op_name << " found cached spec";
  if (spec->ShouldFallback()) {
    DLOG() << "NPU Executing " << op_name << " fallback[" << spec->FallbackReason() << "]";
    FallbackCPU(context, op_name, attributes, inputs.size(), inputs.data(), num_outputs, outputs, s);
    if (TF_GetCode(s) != TF_OK && kDumpExecutionDetail) {
      LOG(INFO) << "NPU Executing " << op_name << " fallback status: " << s->status.ToString();
      std::stringstream ss;
      ss << spec->DebugString() << std::endl;
      for (int i = 0; i < num_inputs; i++) {
        tensorflow::Status status;
        const tensorflow::Tensor *tensor = nullptr;
        npu::UnwrapHandle(inputs[i])->DeviceName(&status);
        npu::UnwrapTensor(inputs[i], &tensor);
        ss << "input " << i << " " << tensorflow::DataTypeString(tensor->dtype()) << " device "
           << npu::UnwrapHandle(inputs[i])->DeviceName(&status) << std::endl;
      }
      LOG(INFO) << ss.str();
    }
  } else {
    DLOG() << "NPU Executing " << op_name << " dispatched to npu executor";
    Run(context, spec, inputs.size(), inputs.data(), num_outputs, outputs, s);
  }
}

void NpuDevice::Run(TFE_Context *context, std::shared_ptr<const npu::TaskSpec> spec, int num_inputs,
                    TFE_TensorHandle **inputs, int *num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  if (spec->IsFunctionOp()) {
    DLOG() << "NPU Executor start executing function op " << spec->Op();
    RunGraph(context, reinterpret_cast<const npu::FuncSpec *>(spec.get()), num_inputs, inputs, num_outputs, outputs,
             status);
  } else {
    DLOG() << "NPU Executor start executing normal op " << spec->Op();
    RunOp(context, reinterpret_cast<const npu::OpSpec *>(spec.get()), num_inputs, inputs, num_outputs, outputs, status);
  }
}

void NpuDevice::RunOp(TFE_Context *context, const npu::OpSpec *spec, int num_inputs, TFE_TensorHandle **inputs,
                      int *num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  TensorShapes output_shapes;
  tensorflow::NodeDef parser_ndef = spec->ParserNodeDef();
  if (spec->ShouldInferShape()) {
    DLOG() << "NPU Executing op " << spec->Op() << " need re-infer shape";
    TensorPartialShapes partial_shapes;
    bool unused = false;
    bool should_fallback =
      !InferShape(context, spec->OpRegistrationData(), spec->NodeDef(), num_inputs, inputs, partial_shapes, unused)
         .ok();
    if (!should_fallback) {
      output_shapes.resize(partial_shapes.size());
      for (size_t i = 0; i < partial_shapes.size(); i++) {
        DLOG() << "NPU Executing op " << spec->Op() << " re-infer shape output " << i
               << partial_shapes[i].DebugString();
        if (!partial_shapes[i].AsTensorShape(&output_shapes[i])) {
          should_fallback = true;
          break;
        }
      }
    }
    if (should_fallback) {
      DLOG() << "NPU Executing op " << spec->Op() << " fallback cpu after re-infer shape";
      FallbackCPU(context, spec, num_inputs, inputs, num_outputs, outputs, status);
      return;
    }
    AssembleOutputDesc(output_shapes, spec->OutputTypes(), &parser_ndef);
  } else {
    output_shapes = spec->OutputShapes();
  }

  NpuCustomKernelFunc *custom_kernel = nullptr;
  if (CustomKernelRegistry::Instance().GetCustomKernelFunc(spec->Op(), &custom_kernel)) {
    (*custom_kernel)(context, this, spec, output_shapes, parser_ndef, num_inputs, inputs, *num_outputs, outputs,
                     status);
    return;
  }

  if (!kExecuteOpByAcl) {
    bool op_can_fallback = true;
    if (SupportedResourceGenerator(spec->Op())) {  // Should never fallback npu resource generator
      DLOG() << "Op " << spec->Op() << " not fallback cpu as it is resource generator";
      op_can_fallback = false;
    } else {
      for (int i = 0; i < num_inputs; ++i) {  // Should never fallback if op has npu resource input
        if (IsNpuTensorHandle(npu::UnwrapHandle(inputs[i])) &&
            npu::UnwrapHandle(inputs[i])->DataType() == tensorflow::DT_RESOURCE) {
          DLOG() << "Op " << spec->Op() << " not fallback cpu as it has resource input from NPU";
          op_can_fallback = false;
          break;
        }
      }
    }
    if (op_can_fallback) {
      DLOG() << "NPU Executing op " << spec->Op() << " fallback cpu as acl engine not enabled";
      FallbackCPU(context, spec, num_inputs, inputs, num_outputs, outputs, status);
      return;
    }
  }
  // 输入如果是CPU,此时要转换成NPU
  std::vector<TFE_TensorHandle *> npu_inputs(num_inputs);
  ScopeTensorHandleDeleter scope_handle_deleter;
  for (int i = 0; i < num_inputs; ++i) {
    TFE_TensorHandle *input = inputs[i];
    // 到达这里的Resource，要么是CPU的镜像 要么是NPU
    if (!IsNpuTensorHandle(npu::UnwrapHandle(input)) &&
        npu::UnwrapHandle(input)->DataType() != tensorflow::DT_RESOURCE) {
      tensorflow::Status s;
      auto src_name = npu::UnwrapHandle(input)->DeviceName(&s);
      NPU_CTX_REQUIRES_OK(status, s);
      DLOG() << "Copying " << spec->Op() << " input:" << i
             << " type:" << tensorflow::DataTypeString(npu::UnwrapHandle(input)->DataType()) << " to NPU from "
             << src_name << " for acl executing";
      // 这里需要根据算子选择输入格式了
      input = CopyTensorH2D(context, input, Format::FORMAT_ND, status);
      scope_handle_deleter.Guard(input);
      if (TF_GetCode(status) != TF_OK) return;
    }
    npu_inputs[i] = input;
  }
  const auto &output_types = spec->OutputTypes();
  for (size_t i = 0; i < output_types.size(); ++i) {
    if (output_types[i] == tensorflow::DT_RESOURCE) {
      outputs[i] = NewDeviceResourceHandle(context, output_shapes[i], status);
      if (TF_GetCode(status) != TF_OK) {
        return;
      }
    } else {
      outputs[i] = NewDeviceTensorHandle(context, Format::FORMAT_ND, output_shapes[i], output_types[i], status);
      if (TF_GetCode(status) != TF_OK) {
        return;
      }
    }
  }
  /******************************************模拟NPU执行Start************************************/
  // TODO:下面换成真实的ACL调用即可，当前直接FallbackCPU
  // npu_inputs 指向NPU内存的TFE_TensorHandle**
  // outputs 指向NPU内存的TFE_TensorHandle**
  // parser_ndef 打了输入输出描述的ndef，需要优化，后续直接存储ACL的结构体
  // output_shapes 临时变量，算子的输出shape
  // spec
  // 待运算算子的说明信息，必定包含InputShapes(),InputTypes(),OutputTypes()，不一定包含OutputShapes()(因为有的算子inferShape依赖输入的值（如reshape），输出shape需要使用上面的output_shapes临时变量)

  /*
   从TFE_TensorHandle*获取NpuManagedBuffer:
      const tensorflow::Tensor *npu_tensor = nullptr;
      NPU_CTX_REQUIRES_OK(status, npu::UnwrapTensor(npu_inputs[i], &npu_tensor));
      npu::Unwrap<NpuManagedBuffer>(npu_tensor); // 返回值就是NpuManagedBuffer*
  */
  std::vector<TFE_TensorHandle *> acl_inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const tensorflow::Tensor *npu_tensor = nullptr;
    NPU_CTX_REQUIRES_OK(status, npu::UnwrapTensor(npu_inputs[i], &npu_tensor));
    tensorflow::Tensor cpu_tensor(npu_tensor->dtype(), npu_tensor->shape());
    if (npu_tensor->dtype() == tensorflow::DT_RESOURCE) {
      for (int j = 0; j < npu_tensor->NumElements(); j++) {
        cpu_tensor.flat<tensorflow::ResourceHandle>()(j) =
          const_cast<tensorflow::Tensor *>(npu_tensor)->flat<tensorflow::ResourceHandle>()(j);
      }
    } else {
      NPU_CTX_REQUIRES_OK(status, npu::Unwrap<NpuManagedBuffer>(npu_tensor)->AssembleTo(&cpu_tensor));
    }
    acl_inputs[i] = tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(cpu_tensor));
    scope_handle_deleter.Guard(acl_inputs[i]);
    if (TF_GetCode(status) != TF_OK) return;
  }
  /**********调用CPU模拟NPU Start*************/
  std::vector<TFE_TensorHandle *> acl_outputs(*num_outputs);
  FallbackCPU(context, spec, num_inputs, acl_inputs.data(), num_outputs, acl_outputs.data(), status);
  if (TF_GetCode(status) != TF_OK) return;
  /**********调用CPU模拟NPU End*************/
  for (int i = 0; i < *num_outputs; ++i) {
    const tensorflow::Tensor *acl_tensor = nullptr;
    NPU_CTX_REQUIRES_OK(status, npu::UnwrapTensor(acl_outputs[i], &acl_tensor));
    const tensorflow::Tensor *npu_tensor = nullptr;
    NPU_CTX_REQUIRES_OK(status, npu::UnwrapTensor(outputs[i], &npu_tensor));
    if (spec->OutputTypes()[i] == tensorflow::DT_RESOURCE) {
      for (int j = 0; j < npu_tensor->NumElements(); j++) {
        const_cast<tensorflow::Tensor *>(npu_tensor)->flat<tensorflow::ResourceHandle>()(j) =
          acl_tensor->flat<tensorflow::ResourceHandle>()(j);
      }
    } else {
      NPU_CTX_REQUIRES_OK(status, npu::Unwrap<NpuManagedBuffer>(npu_tensor)->AssembleFrom(acl_tensor));
    }
    TFE_DeleteTensorHandle(acl_outputs[i]);
    if (TF_GetCode(status) != TF_OK) return;
  }
  /******************************************模拟NPU执行End************************************/
  DLOG() << "NPU Executing op " << spec->Op() << " succeed by npu excutor";
}

namespace {
tensorflow::Node *AddVarInitToGraph(TFE_Context *context, std::string name, tensorflow::Tensor tensor,
                                    tensorflow::Graph *graph, TF_Status *status) {
  tensorflow::Node *variable;
  tensorflow::Node *value;
  tensorflow::Node *assign_variable;

  NPU_CTX_REQUIRES_OK_RETURN(status,
                             tensorflow::NodeBuilder(name, "VarHandleOp")
                               .Attr("container", "")
                               .Attr("shared_name", name)
                               .Attr("dtype", tensor.dtype())
                               .Attr("shape", tensor.shape())
                               .Finalize(graph, &variable),
                             assign_variable);
  NPU_CTX_REQUIRES_OK_RETURN(status,
                             tensorflow::NodeBuilder(name + "_v", "Const")
                               .Attr("value", tensor)
                               .Attr("dtype", tensor.dtype())
                               .Finalize(graph, &value),
                             assign_variable);
  NPU_CTX_REQUIRES_OK_RETURN(status,
                             tensorflow::NodeBuilder(name + "_op", "AssignVariableOp")
                               .Input(variable, 0)
                               .Input(value, 0)
                               .Attr("dtype", tensor.dtype())
                               .Finalize(graph, &assign_variable),
                             assign_variable);

  AssembleOpDef(variable);
  AssembleOpDef(value);
  AssembleOpDef(assign_variable);

  AssembleOutputDesc(TensorShapes({kScalarShape}), {tensorflow::DT_RESOURCE}, variable);
  AssembleOutputDesc(TensorShapes({tensor.shape()}), {tensor.dtype()}, value);
  AssembleInputDesc(TensorShapes({kScalarShape, tensor.shape()}), {tensorflow::DT_RESOURCE, tensor.dtype()},
                    assign_variable);
  return assign_variable;
}
}  // namespace

void NpuDevice::SetNpuLoopSize(TFE_Context *context, int64_t loop, TF_Status *status) {
  static std::atomic_bool initialized{false};
  static std::atomic_int64_t current_loop_size{1};
  static tensorflow::Status init_status = tensorflow::Status::OK();
  static std::uint64_t loop_var_graph_id = 0;
  const static std::string kLoopVarName = "npu_runconfig/iterations_per_loop";

  if (current_loop_size == loop) return;

  LOG(INFO) << "Set npu loop size to " << loop;

  if (!initialized.exchange(true)) {
    tensorflow::Graph graph(tensorflow::OpRegistry::Global());
    AddVarInitToGraph(context, "npu_runconfig/loop_cond", tensorflow::Tensor(tensorflow::int64(0)), &graph, status);
    if (TF_GetCode(status) != TF_OK) return;
    AddVarInitToGraph(context, "npu_runconfig/one", tensorflow::Tensor(tensorflow::int64(1)), &graph, status);
    if (TF_GetCode(status) != TF_OK) return;
    AddVarInitToGraph(context, "npu_runconfig/zero", tensorflow::Tensor(tensorflow::int64(0)), &graph, status);
    if (TF_GetCode(status) != TF_OK) return;

    RunGeGraphPin2CpuAnonymous(context, "set_npu_loop_conditions", graph.ToGraphDefDebug(), 0, nullptr, 0, nullptr,
                               status);
    if (TF_GetCode(status) != TF_OK) return;

    tensorflow::Node *variable;
    tensorflow::Node *arg;
    tensorflow::Node *assign_variable;

    tensorflow::Graph graph2(tensorflow::OpRegistry::Global());

    NPU_CTX_REQUIRES_OK(status, tensorflow::NodeBuilder(kLoopVarName, "VarHandleOp")
                                  .Attr("container", "")
                                  .Attr("shared_name", kLoopVarName)
                                  .Attr("dtype", tensorflow::DT_INT64)
                                  .Attr("shape", kScalarShape)
                                  .Finalize(&graph2, &variable));
    NPU_CTX_REQUIRES_OK(status, tensorflow::NodeBuilder(kLoopVarName + "_v", "_Arg")
                                  .Attr("T", tensorflow::DT_INT64)
                                  .Attr("index", 0)
                                  .Finalize(&graph2, &arg));
    NPU_CTX_REQUIRES_OK(status, tensorflow::NodeBuilder(kLoopVarName + "_op", "AssignVariableOp")
                                  .Input(variable, 0)
                                  .Input(arg, 0)
                                  .Attr("dtype", tensorflow::DT_INT64)
                                  .Finalize(&graph2, &assign_variable));

    AssembleOpDef(variable);
    AssembleOpDef(arg);
    AssembleOpDef(assign_variable);

    AssembleOutputDesc(TensorShapes({kScalarShape}), {tensorflow::DT_RESOURCE}, variable);
    AssembleOutputDesc(TensorShapes({kScalarShape}), {tensorflow::DT_INT64}, arg);
    AssembleInputDesc(TensorShapes({kScalarShape, kScalarShape}), {tensorflow::DT_RESOURCE, tensorflow::DT_INT64},
                      assign_variable);

    loop_var_graph_id = AddGeGraph(context, "set_loop_var", graph2.ToGraphDefDebug(), status);
    init_status = status->status;
    if (TF_GetCode(status) != TF_OK) return;
  }

  status->status = init_status;
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<TFE_TensorHandle *> inputs(1);
  inputs[0] =
    tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensorflow::Tensor(tensorflow::int64(loop - 1))));

  RunGeGraphPin2Cpu(context, loop_var_graph_id, inputs.size(), inputs.data(), {}, 0, nullptr, status);

  if (TF_GetCode(status) == TF_OK) {
    current_loop_size = loop;
  }
  for (auto handle : inputs) {
    TFE_DeleteTensorHandle(handle);
  }
}

void NpuDevice::RunGraph(TFE_Context *context, const npu::FuncSpec *spec, int tf_num_inputs,
                         TFE_TensorHandle **tf_inputs, int *num_outputs, TFE_TensorHandle **outputs,
                         TF_Status *status) {
  std::vector<TFE_TensorHandle *> pruned_inputs;
  spec->PruneInputs(tf_num_inputs, tf_inputs, pruned_inputs);
  int num_inputs = pruned_inputs.size();
  TFE_TensorHandle **inputs = pruned_inputs.data();
  // 注意，因为GE当前执行图的时候，输入输出内存都是Host的，所以这里和ACL执行相反，如果输入是NPU，则需要转回CPU，特别的，对于资源类，当前采取的策略是资源入图
  // 输入如果是NPU,此时要转换成CPU
  std::vector<TFE_TensorHandle *> npu_inputs(num_inputs);
  ScopeTensorHandleDeleter scope_handle_deleter;
  for (int i = 0; i < num_inputs; ++i) {
    TFE_TensorHandle *input = inputs[i];
    // 到达这里的Resource，要么是CPU的镜像 要么是NPU
    if (IsNpuTensorHandle(npu::UnwrapHandle(input)) &&
        npu::UnwrapHandle(input)->DataType() != tensorflow::DT_RESOURCE) {
      tensorflow::Status tf_status;
      auto src_name = npu::UnwrapHandle(input)->DeviceName(&tf_status);
      NPU_CTX_REQUIRES_OK(status, tf_status);
      DLOG() << "Copying " << spec->Op() << " input:" << i
             << " type:" << tensorflow::DataTypeString(npu::UnwrapHandle(input)->DataType()) << " from " << src_name
             << " to CPU for graph engine executing";
      // 这里需要根据算子选择输入格式了
      input = CopyTensorD2H(context, input, status);
      scope_handle_deleter.Guard(input);
      if (TF_GetCode(status) != TF_OK) return;
    }
    npu_inputs[i] = input;
  }

  // TODO:这里根据小循环策略修改值
  int64_t iterations_per_loop = spec->NeedLoop() ? kGlobalLoopSize : 1;
  if (spec->NeedLoop()) {
    SetNpuLoopSize(context, iterations_per_loop, status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  for (const auto &resource : spec->DependentHostResources()) {
    if (spec->NeedLoop() || kDumpExecutionDetail) {
      LOG(INFO) << "Start consume iterator resource " << resource.second->Name() << " " << iterations_per_loop
                << " times";
    }
    const tensorflow::Tensor *tensor;
    NPU_CTX_REQUIRES_OK(status, npu::UnwrapTensor(tf_inputs[resource.first], &tensor));
    // 注意，这个callback不能引用捕获，防止中途因为消费某个资源失败而导致coredump
    bool need_loop = spec->NeedLoop();
    auto done = [resource, iterations_per_loop, need_loop](const tensorflow::Status &s) {
      if (need_loop || !s.ok() || kDumpExecutionDetail) {
        LOG(INFO) << "Iterator resource " << resource.second->Name() << " consume " << iterations_per_loop
                  << " times done with status " << s.ToString();
      }
    };
    NPU_CTX_REQUIRES_OK(status, resource.second->ConsumeAsync(*tensor, iterations_per_loop, done));
  }

  MaybeRebuildFuncSpecGraph(context, spec, status);
  if (TF_GetCode(status) != TF_OK) return;

  if (spec->NeedLoop() || kDumpExecutionDetail) {
    LOG(INFO) << "Start run ge graph " << spec->GeGraphId() << " pin to cpu, loop size " << iterations_per_loop;
  }
  npu::Timer timer("Graph engine run ", iterations_per_loop, " times for graph ", spec->GeGraphId());
  timer.Start();
  spec->SetBuilt();
  RunGeGraphPin2Cpu(context, spec->GeGraphId(), num_inputs, npu_inputs.data(), spec->OutputTypes(), *num_outputs,
                    outputs, status);
  timer.Stop();
}

void NpuDevice::RunGeGraphAsync(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                                bool pin_to_npu, const TensorDataTypes &output_types, int num_outputs,
                                TFE_TensorHandle **outputs, DoneCallback done, TF_Status *status) {
  std::vector<ge::Tensor> ge_inputs;

  DLOG() << "Ge graph " << graph_id << " input info";
  for (int i = 0; i < num_inputs; i++) {
    const tensorflow::Tensor *tensor = nullptr;
    npu::UnwrapTensor(inputs[i], &tensor);

    const static std::shared_ptr<domi::ModelParser> parser =
      domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
    if (parser == nullptr) {
      status->status = tensorflow::errors::Internal("NPU Create new tensorflow model parser failed");
      return;
    }
    ge::DataType ge_type = parser->ConvertToGeDataType(static_cast<uint32_t>(tensor->dtype()));
    NPU_CTX_REQUIRES(
      status, ge_type != ge::DT_UNDEFINED,
      tensorflow::errors::InvalidArgument("Failed map tensorflow data type ",
                                          tensorflow::DataTypeString(tensor->dtype()), " to ge data type"));
    ge::Tensor input;
    std::vector<int64_t> dims;
    for (auto dim_size : tensor->shape().dim_sizes()) {
      dims.emplace_back(dim_size);
    }
    input.SetTensorDesc(ge::TensorDesc(ge::Shape(dims), ge::FORMAT_ND, ge_type));
    input.SetData(reinterpret_cast<const uint8_t *>(tensor->tensor_data().data()), tensor->TotalBytes());
    ge_inputs.emplace_back(input);
    DLOG() << "    input " << i << " ge enum " << ge_type << " tf type " << tensorflow::DataTypeString(tensor->dtype())
           << VecToString(dims);
  }
  auto ge_callback = [&, graph_id](ge::Status s, std::vector<ge::Tensor> &ge_outputs) {
    DLOG() << "Graph engine callback with status:" << s;
    if (s == ge::END_OF_SEQUENCE) {
      done(tensorflow::errors::OutOfRange("Graph engine process graph ", graph_id, " reach end of sequence"));
      return;
    } else if (s != ge::SUCCESS) {
      std::string err_msg = ge::GEGetErrorMsg();
      if (err_msg.empty()) {
        err_msg = "<unknown error> code:" + std::to_string(s);
      }
      done(tensorflow::errors::Internal("Graph engine process graph failed: ", err_msg));
      return;
    } else if (ge_outputs.size() != static_cast<std::size_t>(num_outputs)) {
      done(tensorflow::errors::Internal("Graph engine process graph succeed but output num ", ge_outputs.size(),
                                        " mismatch with expected ", num_outputs));
      return;
    }

    DLOG() << "Ge graph " << graph_id << " output info";
    for (size_t i = 0; i < ge_outputs.size(); i++) {
      auto &ge_tensor = ge_outputs[i];
      std::vector<tensorflow::int64> dims;
      for (auto dim_size : ge_tensor.GetTensorDesc().GetShape().GetDims()) {
        dims.push_back(dim_size);
      }
      tensorflow::TensorShape shape;
      tensorflow::Status tf_status = tensorflow::TensorShapeUtils::MakeShape(dims.data(), dims.size(), &shape);
      if (!tf_status.ok()) {
        done(tensorflow::errors::Internal("Graph engine process graph succeed but output ", i, " dims invalid ",
                                          VecToString(ge_tensor.GetTensorDesc().GetShape().GetDims()), " ",
                                          tf_status.error_message()));
        return;
      }
      DLOG() << "    output " << i << " ge type enum " << ge_tensor.GetTensorDesc().GetDataType() << " tf type "
             << tensorflow::DataTypeString(output_types[i]) << shape.DebugString();

      const static int64_t kTensorAlignBytes = 64;
      if (reinterpret_cast<uintptr_t>(ge_tensor.GetData()) % kTensorAlignBytes == 0) {
        DLOG() << "Zero copy ge tensor " << reinterpret_cast<uintptr_t>(ge_tensor.GetData()) << " as aligned with "
               << kTensorAlignBytes << " bytes";
        size_t ge_tensor_total_bytes = ge_tensor.GetSize();
        tensorflow::Allocator *allocator =
          NpuHostFixedAllocator<uint8_t[], std::function<void(uint8_t *)>>::Create(std::move(ge_tensor.ResetData()));
        tensorflow::Tensor cpu_tensor(allocator, output_types[i], shape);
        if (ge_tensor_total_bytes != cpu_tensor.TotalBytes()) {
          done(tensorflow::errors::Internal("Graph engine process graph succeed but output ", i, " total bytes ",
                                            ge_tensor_total_bytes, " mismatch with expected ",
                                            cpu_tensor.TotalBytes()));
          return;
        }
        outputs[i] = tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(cpu_tensor));
      } else {
        DLOG() << "Skip zero copy as ge tensor " << reinterpret_cast<uintptr_t>(ge_tensor.GetData())
               << " not aligned with " << kTensorAlignBytes << " bytes";
        tensorflow::Tensor cpu_tensor(output_types[i], shape);
        if (ge_tensor.GetSize() != cpu_tensor.TotalBytes()) {
          done(tensorflow::errors::Internal("Graph engine process graph succeed but output ", i, " total bytes ",
                                            ge_tensor.GetSize(), " mismatch with expected ", cpu_tensor.TotalBytes()));
          return;
        }
        memcpy(const_cast<char *>(cpu_tensor.tensor_data().data()), ge_tensor.GetData(), ge_tensor.GetSize());
        outputs[i] = tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(cpu_tensor));
      }

      if (pin_to_npu) {
        TFE_TensorHandle *handle = outputs[i];
        outputs[i] = CopyTensorH2D(context, handle, status);
        TFE_DeleteTensorHandle(handle);
        if (TF_GetCode(status) != TF_OK) {
          done(tensorflow::Status(status->status.code(),
                                  std::string("Graph engine process graph succeed but copy output ") +
                                    std::to_string(i) + " to npu failed " + status->status.error_message()));
          return;
        }
      }
    }
    done(tensorflow::Status::OK());
  };
  NPU_CTX_REQUIRES_GE_OK(status, "NPU Schedule graph to graph engine",
                         ge_session_->RunGraphAsync(graph_id, ge_inputs, ge_callback));
}

uint64_t NpuDevice::AddGeGraphInner(TFE_Context *context, uint64_t graph_id, const std::string &name,
                                    const tensorflow::GraphDef &def, bool loop, TF_Status *status) {
  auto ge_compute_graph = std::make_shared<ge::ComputeGraph>(name);
  std::shared_ptr<domi::ModelParser> parser =
    domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  if (parser == nullptr) {
    status->status = tensorflow::errors::Internal("NPU Create new tensorflow model parser failed");
    return graph_id;
  }

  auto request_subgraph = [this, name, context](const std::string &fn) -> std::string {
    DLOG() << "Tensorflow model parser requesting subgraph " << fn << " for ge graph " << name;
    tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
    const tensorflow::FunctionDef *fdef = lib_def->Find(fn);
    if (fdef == nullptr) {
      return "";
    }
    std::unique_ptr<tensorflow::FunctionBody> fbody;
    auto status = FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice{}, lib_def, &fbody);
    if (!status.ok()) {
      LOG(ERROR) << "Failed trans function body to graph";
      return "";
    }

    tensorflow::ProcessFunctionLibraryRuntime *pflr = npu::UnwrapCtx(context)->pflr();
    tensorflow::FunctionLibraryRuntime *flr = pflr->GetFLR("/job:localhost/replica:0/task:0/device:CPU:0");

    std::unique_ptr<tensorflow::Graph> graph = std::make_unique<tensorflow::Graph>(lib_def);
    CopyGraph(*fbody->graph, graph.get());
    tensorflow::OptimizeGraph(flr, &graph);

    PruneFunction(*fdef, graph.get());

    MarkGraphNodeInOutDesc(context, graph.get(), 0, nullptr);

    if (kDumpExecutionDetail || kDumpGraph) {
      WriteTextProto(tensorflow::Env::Default(), name + "_subgraph_" + fn + ".pbtxt", graph->ToGraphDefDebug());
    }
    return graph->ToGraphDefDebug().SerializeAsString();
  };

  NPU_CTX_REQUIRES_GE_OK_RETURN(
    status, "NPU Parse tensorflow model",
    parser->ParseProtoWithSubgraph(def.SerializeAsString(), request_subgraph, ge_compute_graph), graph_id);

  ge::Graph ge_graph = ge::GraphUtils::CreateGraphFromComputeGraph(ge_compute_graph);

  ge_graph.SetNeedIteration(loop);

  NPU_CTX_REQUIRES_GE_OK_RETURN(status, "Graph engine Add graph", GeSession()->AddGraph(graph_id, ge_graph), graph_id);
  return graph_id;
}

uint64_t NpuDevice::AddGeGraph(TFE_Context *context, uint64_t graph_id, const std::string &name,
                               const tensorflow::GraphDef &def, TF_Status *status) {
  return AddGeGraphInner(context, graph_id, name, def, false, status);
}

uint64_t NpuDevice::AddGeGraph(TFE_Context *context, const std::string &name, const tensorflow::GraphDef &def,
                               TF_Status *status) {
  uint64_t graph_id = NextUUID();
  return AddGeGraph(context, graph_id, name, def, status);
}

tensorflow::Status NpuDevice::GetAutoLoopGraph(TFE_Context *context, tensorflow::Graph *origin_graph, int num_inputs,
                                               TFE_TensorHandle **inputs, bool &loop, tensorflow::GraphDef *def) {
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  std::unique_ptr<tensorflow::Graph> graph = std::make_unique<tensorflow::Graph>(lib_def);
  CopyGraph(*origin_graph, graph.get());

  tensorflow::Node *key;
  if (!IsGraphNeedLoop(graph.get(), &key)) {
    loop = false;
    graph->ToGraphDef(def);
    return tensorflow::Status::OK();
  }

  loop = true;

  const auto fn_name = key->attrs().Find("body")->func().name();
  DLOG() << "Inline while body func " << fn_name << " for node " << key->name();
  auto builder = tensorflow::NodeBuilder(fn_name, fn_name, lib_def);
  for (int i = 0; i < key->num_inputs(); i++) {
    const tensorflow::Edge *edge;
    NPU_REQUIRES_OK(key->input_edge(i, &edge));
    builder.Input(edge->src(), edge->src_output());
  }
  for (auto edge : key->in_edges()) {
    if (edge->IsControlEdge()) {
      builder.ControlInput(edge->src());
    }
  }

  tensorflow::Node *fn_node;
  NPU_REQUIRES_OK(builder.Finalize(graph.get(), &fn_node));

  graph->RemoveNode(key);
  tensorflow::FixupSourceAndSinkEdges(graph.get());

  tensorflow::ProcessFunctionLibraryRuntime *pflr = npu::UnwrapCtx(context)->pflr();
  tensorflow::FunctionLibraryRuntime *flr = pflr->GetFLR("/job:localhost/replica:0/task:0/device:CPU:0");

  tensorflow::OptimizeGraph(flr, &graph);

  for (auto node : graph->op_nodes()) {
    if (tensorflow::grappler::IsVariable(node->def())) {
      if (node->attrs().Find("shared_name") != nullptr) {
        DLOG() << "Change node " << node->name() << " name to " << node->attrs().Find("shared_name")->s();
        node->set_name(node->attrs().Find("shared_name")->s());
      }
    }
  }

  MarkGraphNodeInOutDesc(context, graph.get(), num_inputs, inputs);
  graph->ToGraphDef(def);
  return tensorflow::Status::OK();
}

void NpuDevice::RemoveGeGraph(TFE_Context *context, uint64_t graph_id, TF_Status *status) {
  NPU_CTX_REQUIRES_GE_OK(status, "Graph engine Remove graph", GeSession()->RemoveGraph(graph_id));
}

void NpuDevice::RunGeGraph(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                           bool pin_to_npu, const TensorDataTypes &output_types, int num_outputs,
                           TFE_TensorHandle **outputs, TF_Status *status) {
  tensorflow::Notification notification;
  auto done = [status, &notification](tensorflow::Status s) {
    status->status = std::move(s);
    notification.Notify();
  };
  RunGeGraphAsync(context, graph_id, num_inputs, inputs, pin_to_npu, output_types, num_outputs, outputs, done, status);
  notification.WaitForNotification();
}

void NpuDevice::RunGeGraphPin2CpuAsync(TFE_Context *context, uint64_t graph_id, int num_inputs,
                                       TFE_TensorHandle **inputs, const TensorDataTypes &output_types, int num_outputs,
                                       TFE_TensorHandle **outputs, DoneCallback done, TF_Status *status) {
  RunGeGraphAsync(context, graph_id, num_inputs, inputs, false, output_types, num_outputs, outputs, std::move(done),
                  status);
}

void NpuDevice::RunGeGraphPin2NpuAsync(TFE_Context *context, uint64_t graph_id, int num_inputs,
                                       TFE_TensorHandle **inputs, const TensorDataTypes &output_types, int num_outputs,
                                       TFE_TensorHandle **outputs, DoneCallback done, TF_Status *status) {
  RunGeGraphAsync(context, graph_id, num_inputs, inputs, true, output_types, num_outputs, outputs, std::move(done),
                  status);
}

void NpuDevice::RunGeGraphPin2Cpu(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                                  const TensorDataTypes &output_types, int num_outputs, TFE_TensorHandle **outputs,
                                  TF_Status *status) {
  RunGeGraph(context, graph_id, num_inputs, inputs, false, output_types, num_outputs, outputs, status);
}

void NpuDevice::RunGeGraphPin2Npu(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                                  const TensorDataTypes &output_types, int num_outputs, TFE_TensorHandle **outputs,
                                  TF_Status *status) {
  RunGeGraph(context, graph_id, num_inputs, inputs, true, output_types, num_outputs, outputs, status);
}

void NpuDevice::RunGeGraphAnonymous(TFE_Context *context, const std::string &name, const tensorflow::GraphDef &gdef,
                                    int num_inputs, TFE_TensorHandle **inputs, bool pin_to_npu, int num_outputs,
                                    TFE_TensorHandle **outputs, TF_Status *status) {
  uint64_t graph_id = AddGeGraph(context, name, gdef, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::map<int, tensorflow::DataType> indexed_types;

  for (const auto &node : gdef.node()) {
    if (node.op() == "_Retval") {
      tensorflow::DataType type;
      tensorflow::GetNodeAttr(node, "T", &type);
      int index;
      tensorflow::GetNodeAttr(node, "index", &index);
      indexed_types[index] = type;
    }
  }
  TensorDataTypes types;
  for (auto indexed_type : indexed_types) {
    types.emplace_back(indexed_type.second);
  }

  RunGeGraph(context, graph_id, num_inputs, inputs, pin_to_npu, types, num_outputs, outputs, status);
  if (TF_GetCode(status) != TF_OK) return;

  RemoveGeGraph(context, graph_id, status);
  if (TF_GetCode(status) != TF_OK) return;
}

void NpuDevice::RunGeGraphPin2CpuAnonymous(TFE_Context *context, const std::string &name,
                                           const tensorflow::GraphDef &gdef, int num_inputs, TFE_TensorHandle **inputs,
                                           int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  RunGeGraphAnonymous(context, name, gdef, num_inputs, inputs, false, num_outputs, outputs, status);
}

void NpuDevice::RunGeGraphPin2NpuAnonymous(TFE_Context *context, const std::string &name,
                                           const tensorflow::GraphDef &gdef, int num_inputs, TFE_TensorHandle **inputs,
                                           int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  RunGeGraphAnonymous(context, name, gdef, num_inputs, inputs, true, num_outputs, outputs, status);
}

void NpuDevice::MaybeRebuildFuncSpecGraph(TFE_Context *context, const npu::FuncSpec *spec, TF_Status *status) {
  if (spec->Built() && GeSession()->IsGraphNeedRebuild(spec->GeGraphId())) {
    LOG(INFO) << "Start rebuild ge graph " << spec->GeGraphId();
    RemoveGeGraph(context, spec->GeGraphId(), status);
    if (TF_GetCode(status) != TF_OK) return;
    AddGeGraphInner(context, spec->GeGraphId(), spec->Op(), *spec->GraphDef(), spec->NeedLoop(), status);
  }
}

void NpuDevice::GetCachedTaskSpec(const tensorflow::NodeDef &ndef, std::shared_ptr<const npu::TaskSpec> *spec,
                                  bool &request_shape) {
  *spec = nullptr;
  const auto &op = ndef.op();
  if (cached_func_specs_.find(op) == cached_func_specs_.end()) {
    HashKey attr_hash = Hash(ndef);
    request_shape = cached_op_specs_.count(op) && cached_op_specs_[op].count(attr_hash);
    return;
  }
  *spec = cached_func_specs_[op];
}

void NpuDevice::GetCachedTaskSpec(const tensorflow::NodeDef &ndef, const TensorShapes &shapes,
                                  std::shared_ptr<const npu::TaskSpec> *spec) {
  *spec = nullptr;
  bool request_shape = false;
  GetCachedTaskSpec(ndef, spec, request_shape);
  if (*spec != nullptr) {
    return;
  }
  if (!request_shape) {
    return;
  }
  HashKey attr_hash = Hash(ndef);
  HashKey shape_hash = Hash(shapes);
  const auto &op = ndef.op();
  if (cached_op_specs_.count(op) && cached_op_specs_[op].count(attr_hash) &&
      cached_op_specs_[op][attr_hash].count(shape_hash)) {
    *spec = cached_op_specs_[op][attr_hash][shape_hash];
  }
}

std::shared_ptr<const npu::TaskSpec> NpuDevice::CacheFuncSpec(
  const char *op, const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef, uint64_t ge_graph_id,
  std::unique_ptr<const tensorflow::GraphDef> graph, const npu::FuncSpec::PruneInputsFunc &prune_func,
  const std::map<int, std::shared_ptr<IteratorResourceProvider>> &dependent_host_resources, const std::string &reason) {
  auto spec = std::make_shared<npu::FuncSpec>(op_spec, ndef, ge_graph_id, std::move(graph), prune_func,
                                              dependent_host_resources, reason);
  cached_func_specs_[op] = spec;
  DLOG() << "Cache function op spec " << spec->DebugString();
  return spec;
}

std::shared_ptr<const npu::TaskSpec> NpuDevice::CacheOpSpec(
  const char *op, const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef,
  const TensorShapes &input_shapes, const TensorPartialShapes &output_shapes, const std::string &reason) {
  auto spec = std::make_shared<npu::OpSpec>(op_spec, ndef, input_shapes, output_shapes, reason);
  cached_op_specs_[op][Hash(ndef)][Hash(input_shapes)] = spec;
  DLOG() << "Cache op spec " << spec->DebugString();
  return spec;
}

std::shared_ptr<const npu::TaskSpec> NpuDevice::CacheOpSpec(const char *op,
                                                            const tensorflow::OpRegistrationData *op_spec,
                                                            const tensorflow::NodeDef &ndef,
                                                            const TensorShapes &input_shapes,
                                                            const std::string &reason) {
  auto spec = std::make_shared<npu::OpSpec>(op_spec, ndef, input_shapes, reason);
  cached_op_specs_[op][Hash(ndef)][Hash(input_shapes)] = spec;
  DLOG() << "Cache op spec " << spec->DebugString();
  return spec;
}

bool NpuDevice::Supported(const std::string &op) {
  const static std::unordered_set<std::string> kUnsupportedOps = {};
  return kUnsupportedOps.count(op) == 0;
}

bool NpuDevice::SupportedResourceGenerator(const std::string &op) {
  const static std::unordered_set<std::string> kUnsupportedOps = {"VarHandleOp"};
  return kUnsupportedOps.count(op) != 0;
}

void NpuDevice::RecordIteratorMirror(const tensorflow::ResourceHandle &src, const TensorPartialShapes &shapes,
                                     const TensorDataTypes &types) {
  iterator_mirrors_.emplace(src, std::make_pair(shapes, types));
}

bool NpuDevice::MirroredIterator(const tensorflow::ResourceHandle &src) {
  return iterator_mirrors_.find(src) != iterator_mirrors_.end();
}

bool NpuDevice::Mirrored(const tensorflow::ResourceHandle &src) {
  // TODO:可能后续还有其他需要mirror的资源，外层判断资源兼容时务必使用这个接口
  return iterator_mirrors_.find(src) != iterator_mirrors_.end();
}

tensorflow::Status NpuDevice::GetMirroredIteratorShapesAndTypes(const tensorflow::ResourceHandle &src,
                                                                TensorPartialShapes &shapes, TensorDataTypes &types) {
  auto iter = iterator_mirrors_.find(src);
  if (iter == iterator_mirrors_.end()) {
    return tensorflow::errors::Internal("Resource ", src.DebugString(), " has not been mirrored");
  }
  shapes.assign(iter->second.first.begin(), iter->second.first.end());
  types.assign(iter->second.second.begin(), iter->second.second.end());
  return tensorflow::Status::OK();
}
