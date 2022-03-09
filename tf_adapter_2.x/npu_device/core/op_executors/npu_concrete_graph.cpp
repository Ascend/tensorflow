/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include "npu_concrete_graph.h"

#include "npu_device.h"
#include "npu_global.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/grappler/op_types.h"

namespace {
bool IsGraphNeedLoop(const tensorflow::Graph *graph, tensorflow::Node **key) {
  *key = nullptr;
  for (auto node : graph->op_nodes()) {
    if (node->IsWhileNode()) {
      if (*key != nullptr) {
        DLOG() << "Skip check as multi while nodes in graph first " << (*key)->name() << " another " << node->name();
        *key = nullptr;
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
  const std::function<void(const tensorflow::Node *)> &enter = [&reserved_nums](const tensorflow::Node *node) {
    if (node->IsOp()) {
      reserved_nums++;
    }
  };
  tensorflow::ReverseDFSFrom(*graph, {*key}, enter, {}, {}, {});
  DLOG() << "Reserved nodes " << reserved_nums << " vs. totally " << graph->num_op_nodes();
  return static_cast<int>(reserved_nums) == graph->num_op_nodes();
}
}  // namespace

namespace npu {
std::string NpuConcreteGraph::AttachedDebugString() const {
  std::stringstream ss;
  return ss.str();
}

void NpuConcreteGraph::RunImpl(TFE_Context *context, NpuDevice *device, int tf_num_inputs, TFE_TensorHandle **tf_inputs,
                               int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  if (execution_type_ != ExecutionType::NPU) {
    DLOG() << "Run function graph " << Op() << " on cpu";
    device->FallbackCPU(context, NodeDef(), tf_num_inputs, tf_inputs, num_outputs, outputs, status);
    return;
  }

  for (auto &item : bypass_outputs_) {
    DLOG() << "Ref " << Op() << " output " << item.first << " from input " << item.second;
    outputs[item.first] = tf_inputs[item.second];
    tensorflow::unwrap(outputs[item.first])->Ref();
  }

  if (empty_ge_graph_) {
    DLOG() << "Skipped run empty ge graph";
    return;
  }

  ScopeTensorHandleDeleter scope_handle_deleter;
  for (size_t i = 0; i < consumed_inputs_.size(); i++) {
    auto input_index = consumed_inputs_[i];
    DLOG() << "Mapping npu graph " << Op() << " input " << i << " from tensorflow input " << input_index;
    TFE_TensorHandle *input = tf_inputs[input_index];
    if (npu::IsNpuTensorHandle(input)) {
      DLOG() << "Copying " << Op() << " tensorflow input " << input_index << " to npu graph input " << i << " type "
             << tensorflow::DataTypeString(InputTypes()[input_index]) << " from NPU to CPU for graph engine executing";
      // 这里需要根据算子选择输入格式了
      input = device->CopyTensorD2H(context, input, status);
      scope_handle_deleter.Guard(input);
      NPU_REQUIRES_TFE_OK(status);
    }
    input_handles_[i] = input;
  }

  // 这里根据小循环策略修改值
  int64_t iterations_per_loop = 1;
  if (NeedLoop()) {
    iterations_per_loop = npu::global::g_npu_loop_size;
    device->SetNpuLoopSize(context, iterations_per_loop, status);
    NPU_REQUIRES_TFE_OK(status);
  }

  int64_t consume_resource_times = 1;
  if (NeedLoop() || BuiltinLoop()) {
    consume_resource_times = npu::global::g_npu_loop_size;
  }

  bool looped = NeedLoop() || BuiltinLoop();
  for (const auto &resource : ConsumedIteratos()) {
    if (looped || kDumpExecutionDetail) {
      LOG(INFO) << "Start consume iterator resource " << resource.second->Name() << " " << consume_resource_times
                << " times";
    }
    const tensorflow::Tensor *tensor;
    NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(tf_inputs[resource.first], &tensor));
    // 注意，这个callback不能引用捕获，防止中途因为消费某个资源失败而导致coredump
    auto done = [resource, consume_resource_times, looped](const tensorflow::Status &s) {
      if (looped || !s.ok() || kDumpExecutionDetail) {
        LOG(INFO) << "Iterator resource " << resource.second->Name() << " consume " << consume_resource_times
                  << " times done with status " << s.ToString();
      }
    };
    NPU_CTX_REQUIRES_OK(status, resource.second->ConsumeAsync(*tensor, consume_resource_times, done));
  }

  Load(context, device, status);
  if (empty_ge_graph_) {
    DLOG() << "Skipped run empty ge graph";
    return;
  }

  if (NeedLoop() || kDumpExecutionDetail) {
    LOG(INFO) << "Start run ge graph " << GeGraphId() << " pin to cpu, loop size " << iterations_per_loop;
  }
  npu::Timer timer("Graph engine run ", iterations_per_loop, " times for graph ", GeGraphId());
  timer.Start();
  device->RunGeGraphPin2Cpu(context, GeGraphId(), input_handles_.size(), input_handles_.data(), OutputTypes(),
                            output_handles_.size(), output_handles_.data(), status);
  for (size_t i = 0; i < output_handles_.size(); i++) {
    DLOG() << "Mapping npu graph " << Op() << " output " << i << " to tensorflow output " << produced_outputs_[i];
    outputs[produced_outputs_[i]] = output_handles_[i];
  }
  timer.Stop();
}

void NpuConcreteGraph::Load(TFE_Context *context, NpuDevice *device, TF_Status *status) const {
  if (Built() && device->GeSession()->IsGraphNeedRebuild(GeGraphId())) {
    LOG(INFO) << "Unload ge graph " << GeGraphId() << " for rebuild of op " << Op();
    device->RemoveGeGraph(context, GeGraphId(), status);
    NPU_REQUIRES_TFE_OK(status);
    built_ = false;
  }

  if (!built_) {
    DLOG() << "Load ge graph " << GeGraphId() << " of op " << Op();
    if (kEmptyGeGraphId == device->AddGeGraphInner(context, GeGraphId(), Op(), GraphDef(), NeedLoop(), status)) {
      empty_ge_graph_ = true;
    }
    NPU_REQUIRES_TFE_OK(status);
    built_ = true;
    graph_def_serialized_ = true;
  }
}

void NpuConcreteGraph::UnLoad(TFE_Context *context, NpuDevice *device, TF_Status *status) const {
  if (!Built()) {
    return;
  }
  DLOG() << "Unload ge graph " << GeGraphId() << " of op " << Op();
  device->RemoveGeGraph(context, GeGraphId(), status);
  NPU_REQUIRES_TFE_OK(status);
  built_ = false;
}

void NpuConcreteGraph::RunOneShot(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs,
                                  int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  DLOG() << "Run one shot ge graph " << GeGraphId() << " for resource consume op " << Op();
  RunImpl(context, device, num_inputs, inputs, num_outputs, outputs, status);
  NPU_REQUIRES_TFE_OK(status);
  UnLoad(context, device, status);
}

tensorflow::Status NpuMutableConcreteGraph::DevicePartition(TFE_Context *context, const NpuDevice *device) {
  tensorflow::Status input_supported = device->ValidateInputTypes(ConsumedTypes());
  tensorflow::Status output_supported = device->ValidateOutputTypes(ProducedTypes());
  if (!input_supported.ok() || !output_supported.ok()) {
    if (!NpuResources().empty()) {
      SetExecutionType(ExecutionType::MIX);
      std::stringstream ss;
      ss << Op() << " has npu resource input " << NpuResources().begin()->first << " "
         << NpuResources().begin()->second.maybe_type_name() << " but:" << std::endl;
      for (auto &item : CpuResources()) {
        ss << "Resource input " << item.first << " " << item.second.maybe_type_name() << " from cpu" << std::endl;
      }
      ss << "Input type check status " << input_supported.ToString() << std::endl;
      ss << "Output type check status " << output_supported.ToString() << std::endl;
      return tensorflow::errors::Unimplemented(ss.str());
    }
    DLOG() << Op() << " run on cpu as has " << CpuResources().size() << " cpu resources, "
           << "Output: " << output_supported.error_message() << ", Input: " << input_supported.error_message();
    SetExecutionType(ExecutionType::CPU);
  } else {
    SetExecutionType(ExecutionType::NPU);
    NPU_REQUIRES_OK(TryTransToNpuLoopGraph(context));
    AssembleParserAddons(context, MutableGraph());
  }
  return tensorflow::Status::OK();
}

tensorflow::Status NpuMutableConcreteGraph::TryTransToNpuLoopGraph(TFE_Context *context) {
  if (execution_type_ != ExecutionType::NPU) {
    DLOG() << "Skip trans " << Op() << " as npu loop graph as execution type not NPU";
    return tensorflow::Status::OK();
  }

  if (ConsumedIteratos().empty()) {
    DLOG() << "Skip trans " << Op() << " as npu loop graph as not consumed iterator resources";
    return tensorflow::Status::OK();
  }

  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  std::unique_ptr<tensorflow::Graph> graph = std::make_unique<tensorflow::Graph>(lib_def);
  CopyGraph(*Graph(), graph.get());

  tensorflow::Node *key;
  if (!IsGraphNeedLoop(graph.get(), &key)) {
    SetBuiltinLoop(key != nullptr);
    SetNeedLoop(false);
    return tensorflow::Status::OK();
  }

  SetBuiltinLoop(false);
  SetNeedLoop(true);

  const auto fn_name = key->attrs().Find("body")->func().name();
  DLOG() << "Inline while body func " << fn_name << " for node " << key->name();
  auto builder = tensorflow::NodeBuilder(key->name() + "/body", fn_name, lib_def);
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

  // Inline body function will change name of variable, which used as id for npu variable
  for (auto node : graph->op_nodes()) {
    if (!tensorflow::grappler::IsVariable(node->def())) {
      continue;
    }
    auto attr = node->attrs().Find("shared_name");
    if (attr != nullptr) {
      DLOG() << "Change variable " << node->name() << " " << node->type_string() << " name to " << attr->s();
      node->set_name(attr->s());
    }
  }

  OptimizeStageGraphDumper graph_dumper(Op());
  graph_dumper.DumpWithSubGraphs("LOOP", graph->ToGraphDefDebug(), lib_def);

  SetGraph(std::move(graph));
  return tensorflow::Status::OK();
}
}  // namespace npu
