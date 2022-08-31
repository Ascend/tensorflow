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

#include "npu_aoe.h"
#include "npu_device.h"
#include "npu_global.h"
#include "npu_run_context.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/grappler/op_types.h"

namespace npu {
std::string NpuConcreteGraph::AttachedDebugString() const {
  std::stringstream ss;
  return ss.str();
}

void NpuConcreteGraph::RunImpl(TFE_Context *context, NpuDevice *device, int tf_num_inputs, TFE_TensorHandle **tf_inputs,
                               int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  (void)tf_num_inputs;
  (void)num_outputs;
  for (auto &item : bypass_outputs_) {
    DLOG() << "Ref " << Op() << " output " << item.first << " from input " << item.second;
    outputs[item.first] = tf_inputs[item.second];
    tensorflow::unwrap(outputs[item.first])->Ref();
  }

  if (execution_type_ == ExecutionType::NPU && empty_ge_graph_) {
    DLOG() << "Skipped run empty ge graph";
    return;
  }

  ScopeTensorHandleDeleter scope_handle_deleter;
  for (size_t i = 0; i < consumed_inputs_.size(); i++) {
    auto input_index = consumed_inputs_[i];
    if (input_index < 0) {
      return;
    }
    DLOG() << "Mapping npu graph " << Op() << " input " << i << " from tensorflow input " << input_index;
    TFE_TensorHandle *input = tf_inputs[input_index];
    if (npu::IsNpuTensorHandle(input)) {
      DLOG() << "Copying " << Op() << " tensorflow input " << input_index << " to npu graph input " << i << " type "
             << tensorflow::DataTypeString(InputTypes()[static_cast<size_t>(input_index)])
             << " from NPU to CPU for graph engine executing";
      // 这里需要根据算子选择输入格式了
      input = device->CopyTensorD2H(context, input, status);
      scope_handle_deleter.Guard(input);
      NPU_REQUIRES_TFE_OK(status);
    }
    input_handles_[i] = input;
  }

  int64_t iterations_per_loop = 1;
  if (loop_type_ == LoopType::NPU_LOOP || loop_type_ == LoopType::HOST_LOOP) {
    iterations_per_loop = npu::global::g_npu_loop_size;
    if (loop_type_ == LoopType::NPU_LOOP) {
      device->SetNpuLoopSize(context, iterations_per_loop, status);
      NPU_REQUIRES_TFE_OK(status);
    }
  }

  bool looped = (loop_type_ != LoopType::NO_LOOP);
  int64_t consume_resource_times = 1;
  if (looped) {
    consume_resource_times = npu::global::g_npu_loop_size;
  }

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

  if (execution_type_ == ExecutionType::MIX) {
    npu::Timer timer("Mix mode run ", mixed_ndef_.name());
    timer.Start();
    device->FallbackCPU(context, mixed_ndef_, static_cast<int>(input_handles_.size()), input_handles_.data(),
                        static_cast<int>(output_handles_.size()), output_handles_.data(), status);
    timer.Stop();
  } else {
    Load(context, device, status);
    NPU_REQUIRES_TFE_OK(status);
    if (empty_ge_graph_) {
      DLOG() << "Skipped run empty ge graph";
      return;
    }

    RunAoeTuning(context, device, input_handles_, status);
    NPU_REQUIRES_TFE_OK(status);

    if (loop_type_ == LoopType::NPU_LOOP || loop_type_ == LoopType::HOST_LOOP || kDumpExecutionDetail) {
      LOG(INFO) << "Start run ge graph " << GeGraphId() << " pin to cpu, loop size " << iterations_per_loop;
    }
    npu::Timer timer("Graph engine run ", iterations_per_loop, " times for graph ", GeGraphId());
    timer.Start();
    int64_t times = 0;
    do {
      device->RunGeGraphPin2Cpu(context, GeGraphId(), static_cast<int>(input_handles_.size()), input_handles_.data(),
                                OutputTypes(), static_cast<int>(output_handles_.size()), output_handles_.data(),
                                status);
    } while (++times < iterations_per_loop && loop_type_ == LoopType::HOST_LOOP);
    timer.Stop();
  }
  for (size_t i = 0; i < output_handles_.size(); i++) {
    DLOG() << "Mapping npu graph " << Op() << " output " << i << " to tensorflow output " << produced_outputs_[i];
    outputs[produced_outputs_[i]] = output_handles_[i];
  }
}

void NpuConcreteGraph::RunAoeTuning(TFE_Context *context, NpuDevice *device, std::vector<TFE_TensorHandle *> inputs,
                                    TF_Status *status) const {
  if (function_op_) {
    // run aoe tuning if need
    if (!device->device_options["ge.jobType"].empty()) {
      auto &aoe = NpuAoe::GetInstance();
      NPU_CTX_REQUIRES_OK(status, aoe.RunAoeTuning(device, context, GeGraphId(), Op(), GraphDef(), inputs));
    }
  }
}

const std::string &NpuConcreteGraph::GraphLoopTypeString() const {
  const static std::string kInvalidLoopTypeString = "invalid";
  const static std::map<LoopType, std::string> kLoopTypeString{{LoopType::NPU_LOOP, "npu-loop"},
                                                               {LoopType::BUILTIN_LOOP, "builtin-loop"},
                                                               {LoopType::HOST_LOOP, "host-loop"},
                                                               {LoopType::NO_LOOP, "no-loop"}};
  auto iter = kLoopTypeString.find(loop_type_);
  if (iter != kLoopTypeString.end()) {
    return iter->second;
  }
  return kInvalidLoopTypeString;
}

bool NpuConcreteGraph::NeedFuzzCompile() const {
  if (fuzz_compile_.has_value()) {
    return fuzz_compile_.value();
  }
  for (auto node : graph_->op_nodes()) {
    if (!node->IsArg()) {
      continue;
    }
    const tensorflow::AttrValue *shape_attr = node->attrs().Find("_output_shapes");
    if (!shape_attr || !shape_attr->has_list() || shape_attr->list().shape().empty()) {
      LOG(ERROR) << Op() << " will be fuzz compiled as input " << node->attrs().Find("index")->i() << " "
                 << node->name() << " has no shape info";
      fuzz_compile_ = true;
      return fuzz_compile_.value();
    }
    tensorflow::PartialTensorShape shape(shape_attr->list().shape(0));
    if (!shape.IsFullyDefined()) {
      LOG(INFO) << Op() << " will be fuzz compiled as input " << node->attrs().Find("index")->i() << " " << node->name()
                << " shape unknown " << shape.DebugString();
      fuzz_compile_ = true;
      return fuzz_compile_.value();
    }
  }
  DLOG() << Op() << " will be compiled in static shape mode";
  fuzz_compile_ = false;
  return fuzz_compile_.value();
}

void NpuConcreteGraph::Load(TFE_Context *context, NpuDevice *device, TF_Status *status) const {
  if (Built() && device->GeSession()->IsGraphNeedRebuild(static_cast<uint32_t>(GeGraphId()))) {
    LOG(INFO) << "Unload ge graph " << GeGraphId() << " for rebuild of op " << Op();
    device->RemoveGeGraph(context, GeGraphId(), status);
    NPU_REQUIRES_TFE_OK(status);
    built_ = false;
  }

  if (!built_) {
    DLOG() << "Load ge graph " << GeGraphId() << " of op " << Op();
    const std::map<std::string, std::string> kOptions{
      {"ge.recompute", npu::GetRunContextOptions().memory_optimize_options.recompute},
      {"ge.graphParallelOptionPath", npu::GetRunContextOptions().graph_parallel_configs.config_path},
      {"ge.enableGraphParallel", npu::GetRunContextOptions().graph_parallel_configs.enable_graph_parallel}
    };
    const static std::map<std::string, std::string> kFuzzCompileOptions{
      {ge::OPTION_EXEC_DYNAMIC_INPUT, "1"},
      {ge::OPTION_EXEC_DYNAMIC_EXECUTE_MODE, "dynamic_execute"},
      {ge::SHAPE_GENERALIZED_BUILD_MODE, "shape_generalized"}};
    const auto need_fuzz_compile = NeedFuzzCompile();
    if (device->AddGeGraphInner(context, GeGraphId(), Op(), GraphDef(),
                                (loop_type_ == LoopType::NPU_LOOP), status,
                                (need_fuzz_compile ? kFuzzCompileOptions : kOptions)) == kEmptyGeGraphId) {
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
    if (key != nullptr) {
      SetLoopType(LoopType::BUILTIN_LOOP);
    }
    return tensorflow::Status::OK();
  }

  const auto fn_name = key->attrs().Find("body")->func().name();
  DLOG() << "Inline while body func " << fn_name << " for node " << key->name();
  auto builder = tensorflow::NodeBuilder(key->name() + "/body", fn_name, lib_def);
  for (int i = 0; i < key->num_inputs(); i++) {
    const tensorflow::Edge *edge;
    NPU_REQUIRES_OK(key->input_edge(i, &edge));
    (void)builder.Input(edge->src(), edge->src_output());
  }
  for (auto edge : key->in_edges()) {
    if (edge->IsControlEdge()) {
      (void)builder.ControlInput(edge->src());
    }
  }

  tensorflow::Node *fn_node;
  NPU_REQUIRES_OK(builder.Finalize(graph.get(), &fn_node));

  graph->RemoveNode(key);
  (void)tensorflow::FixupSourceAndSinkEdges(graph.get());

  tensorflow::ProcessFunctionLibraryRuntime *pflr = npu::UnwrapCtx(context)->pflr();
  tensorflow::FunctionLibraryRuntime *flr = pflr->GetFLR("/job:localhost/replica:0/task:0/device:CPU:0");

  NpuCustomizedOptimizeGraph(flr, &graph);

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

  if (IsGraphHasAnyUnknownShapeNode(graph.get(), lib_def)) {
    DLOG() << "Host loop " << Op() << " as body graph has unknown shape node";
    SetLoopType(LoopType::HOST_LOOP);
  } else {
    SetLoopType(LoopType::NPU_LOOP);
  }
  SetGraph(std::move(graph));

  return tensorflow::Status::OK();
}
}  // namespace npu
