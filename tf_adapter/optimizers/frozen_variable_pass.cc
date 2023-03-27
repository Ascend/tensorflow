/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/npu_attrs.h"

namespace tensorflow {
class DummyDevice : public ThreadPoolDevice {
 public:
  explicit DummyDevice(const std::string &name, ResourceMgr *rmgr) :
    ThreadPoolDevice({}, name, Bytes(INT64_MAX), DeviceLocality(), cpu_allocator()),
    wrapped_rmgr_(rmgr) {}
  ~DummyDevice() override = default;
  Status Sync() override { return Status::OK(); }
  ResourceMgr *resource_manager() override { return wrapped_rmgr_; }
 private:
  ResourceMgr *wrapped_rmgr_;
};

class FrozenVariablePass : public GraphOptimizationPass {
 public:
  FrozenVariablePass() = default;
  ~FrozenVariablePass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;
 private:
  bool IsAllOutputsIdentity(const Node * const node) const;
  bool IsAllOutputsReadOp(const Node * const node) const;
  bool IsNeedBuildPartitionedCall(const Node * const node) const;
  std::map<std::string, std::string> GetGraphConfigs(const Graph &graph) const;
  void RemoveDeadNodes(Graph* g) const;
  Status DoConstantFolding(const GraphOptimizationPassOptions &options, const uint64_t index) const;
};

struct StableNodeCompartor {
  bool operator() (const tensorflow::Node *a, const tensorflow::Node *b) const { return a->id() < b->id(); }
};

DataType EdgeDataType(const tensorflow::Edge &edge) { return edge.src()->output_type(edge.src_output()); }

bool FrozenVariablePass::IsAllOutputsIdentity(const Node * const node) const {
  for (auto out : node->out_nodes()) {
    if (!out->IsIdentity()) {
      return false;
    }
  }
  return true;
}

bool FrozenVariablePass::IsAllOutputsReadOp(const Node * const node) const {
  for (auto out : node->out_nodes()) {
    if (out->type_string() != "ReadVariableOp") {
      return false;
    }
  }
  return true;
}

bool FrozenVariablePass::IsNeedBuildPartitionedCall(const Node * const node) const {
  return ((node->type_string() == "Variable" || node->type_string() == "VariableV2") && IsAllOutputsIdentity(node)) ||
         (node->type_string() == "VarHandleOp" && IsAllOutputsReadOp(node));
}

std::map<std::string, std::string> FrozenVariablePass::GetGraphConfigs(const Graph &graph) const {
  for (Node *n : graph.nodes()) {
    if ((n != nullptr) && (n->attrs().Find("_NpuOptimizer") != nullptr)) {
      return NpuAttrs::GetAllAttrOptions(n->attrs());
    }
  }
  return {};
}

void FrozenVariablePass::RemoveDeadNodes(Graph* g) const {
  std::unordered_set<const Node*> nodes;
  for (auto n : g->nodes()) {
    ADP_LOG(DEBUG) << "Remove dead node, node type: " << n->type_string();
    if (n->IsControlFlow() || n->op_def().is_stateful()) {
      nodes.insert(n);
    }
  }
  (void)PruneForReverseReachability(g, std::move(nodes));
}

Status FrozenVariablePass::DoConstantFolding(const GraphOptimizationPassOptions &options,
        const uint64_t index) const {
  ADP_LOG(INFO) << "Before do const folding " << options.session_options->config.DebugString();
  if (options.device_set == nullptr) {
    return errors::Internal("Failed to get device set to run constant folding");
  }
  const std::string device_name = "/job:localhost/replica:0/task:0/device:CPU:0";
  Device* device = options.device_set->FindDeviceByName(device_name);
  if (device == nullptr) {
    return errors::Internal("Failed to get device to run constant folding");
  }

  std::unique_ptr<DummyDevice> dummy_device = absl::make_unique<DummyDevice>(device_name, device->resource_manager());
  OptimizerOptions opts;
  opts.set_do_constant_folding(true);
  opts.set_max_folded_constant_in_bytes(INT64_MAX);
#ifdef TF_VERSION_TF2
  std::unique_ptr<DeviceMgr> device_mgr =
          absl::make_unique<StaticDeviceMgr>(std::move(dummy_device));
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
          device_mgr.get(), options.session_options->env, nullptr, 0, options.flib_def, opts);
#else
  std::unique_ptr<DeviceMgr> device_mgr =
          absl::make_unique<DeviceMgr>(std::move(dummy_device));
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
          device_mgr.get(), options.session_options->env, 0, options.flib_def, opts);
#endif
  FunctionLibraryRuntime *flr = pflr->GetFLR("/job:localhost/replica:0/task:0/device:CPU:0");
  if (flr == nullptr) {
    return errors::Internal("Failed to create and retrieve function library runtime to run constant folding");
  }
  GraphOptimizer::Options graph_optimizer_options;
  GraphOptimizer optimizer(opts);
  optimizer.Optimize(flr, flr->env(), flr->device(), options.graph, graph_optimizer_options);
  (void)RemoveDeadNodes((options.graph)->get());
  ADP_LOG(INFO) << "After do const folding optimize.";
  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + "TF_AfterFrozenVariable_" + std::to_string(index) + ".pbtxt";
    tensorflow::GraphDef def;
    (*options.graph)->ToGraphDef(&def);
    WriteTextProto(Env::Default(), pbtxt_path, def);
  }

  return Status::OK();
}

Status FrozenVariablePass::Run(const GraphOptimizationPassOptions &options) {
  ADP_LOG(INFO) << "FrozenVariablePass Run";
  if (options.graph == nullptr || options.session_options == nullptr) {
    return Status::OK();
  }

  std::map<std::string, std::string> pass_options = NpuAttrs::GetPassOptions(options);
  std::string is_need_frozen = pass_options["frozen_variable"];
  if (is_need_frozen != "1") {
    ADP_LOG(INFO) << " Skip the optimizer : FrozenVariablePass.";
    return Status::OK();
  }

  Graph *graph_in = (options.graph)->get();
  static std::atomic_uint64_t graph_index{0U};
  uint64_t index = graph_index.fetch_add(1U);
  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + "TF_BeforeFrozenVariable_" + std::to_string(index) + ".pbtxt";
    tensorflow::GraphDef def;
    graph_in->ToGraphDef(&def);
    (void) WriteTextProto(Env::Default(), pbtxt_path, def);
  }

  std::vector<tensorflow::Node *> remove_nodes;
  bool generate_partitioned_call = false;
  for (Node *node : graph_in->op_nodes()) {
    if ((node != nullptr) && (IsNeedBuildPartitionedCall(node))) {
      std::vector<tensorflow::Node *> cluster_nodes = {node};
      std::vector<tensorflow::Node *> cluster_out_nodes;
      for (auto out_node : node->out_nodes()) {
        cluster_nodes.emplace_back(out_node);
        cluster_out_nodes.emplace_back(out_node);
      }
      auto cluster_graph = absl::make_unique<Graph>(tensorflow::OpRegistry::Global());
      std::map<tensorflow::Node *, tensorflow::Node *> node_map;
      for (auto &cluster_node : cluster_nodes) {
        tensorflow::Status status;
        node_map[cluster_node] = cluster_graph->AddNode(cluster_node->def(), &status);
        remove_nodes.emplace_back(cluster_node);
        TF_RETURN_IF_ERROR(status);
      }
      std::set<tensorflow::Node *, StableNodeCompartor> control_inputs;
      std::set<tensorflow::Node *, StableNodeCompartor> control_outputs;
      std::vector<NodeBuilder::NodeOut> inputs;
      std::vector<tensorflow::DataType> input_types;
      std::vector<tensorflow::DataType> output_types;
      for (auto &cluster_node : cluster_nodes) {
        for (auto edge : cluster_node->in_edges()) {
          if (node_map.find(edge->src()) == node_map.end()) {
            if (edge->IsControlEdge()) {
              control_inputs.insert(edge->src());
            } else {
              inputs.emplace_back(edge->src());
              input_types.emplace_back(EdgeDataType(*edge));
            }
          } else { // 给子图里的节点之间连边, _SOURCE节点不在这里连边
            ADP_LOG(INFO) << "cluster_node : " << cluster_node->DebugString();
            ADP_LOG(INFO) << "Edge src : " << edge->src()->DebugString();
            auto e = cluster_graph->AddEdge(node_map[edge->src()], edge->src_output(),
                                            node_map[cluster_node], edge->dst_input());
            REQUIRES_NOT_NULL(e);
            ADP_LOG(INFO) << "Add inner edge : " << e->DebugString();
          }
        }
      }

      for (size_t i = 0U; i < cluster_out_nodes.size(); ++i) {
        tensorflow::Node *ret;
        const string ret_val_name = cluster_out_nodes[i]->name() + "_out_" + std::to_string(i);
        TF_RETURN_IF_ERROR(NodeBuilder(ret_val_name, "_Retval")
                            .Device(node->def().device())
                            .Input(node_map[cluster_out_nodes[i]])
                            .Attr("index", int32_t(i))
                            .Attr("T", cluster_out_nodes[i]->output_type(0))
                            .Finalize(cluster_graph.get(), &ret));
        output_types.emplace_back(cluster_out_nodes[i]->output_type(0));
      }

      const std::string fn = "npu_f_" + node->name() + std::to_string(index);
      tensorflow::NameAttrList func;
      func.set_name(fn);
      Node *npu_node = nullptr;
      const string new_name = "npu_partitioned_call" + node->name();
      TF_RETURN_IF_ERROR(NodeBuilder(new_name, "PartitionedCall")
                          .AssignedDevice(node->assigned_device_name())
                          .ControlInputs(std::vector<Node *>(control_inputs.begin(), control_inputs.end()))
                          .Input(inputs)
                          .Attr("Tin", input_types)
                          .Attr("Tout", output_types)
                          .Attr("f", func)
                          .Finalize(graph_in, &npu_node));
      for (auto cluster_out_node: cluster_out_nodes) {
        for (const Edge *e : cluster_out_node->out_edges()) {
          (void) graph_in->AddEdge(npu_node, e->src_output(), e->dst(), e->dst_input());
        }
      }
      (void)tensorflow::FixupSourceAndSinkEdges(cluster_graph.get());
      FunctionDefLibrary flib;
      TF_RETURN_IF_ERROR(GraphToFunctionDef(*cluster_graph, fn, flib.add_function()));
      TF_RETURN_IF_ERROR(graph_in->AddFunctionLibrary(flib));
      TF_RETURN_IF_ERROR(options.flib_def->AddLibrary(flib));
      generate_partitioned_call = true;
    }
  }
  for (auto &remove_node : remove_nodes) {
    graph_in->RemoveNode(remove_node);
  }
  (void)tensorflow::FixupSourceAndSinkEdges(graph_in);

  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + "TF_AfterReplacePartitionedCall_" + std::to_string(index) + ".pbtxt";
    tensorflow::GraphDef def;
    graph_in->ToGraphDef(&def);
    (void) WriteTextProto(Env::Default(), pbtxt_path, def);
  }

  if (generate_partitioned_call) {
    TF_RETURN_IF_ERROR(DoConstantFolding(options, index));
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, -2, FrozenVariablePass);
}  // namespace tensorflow
