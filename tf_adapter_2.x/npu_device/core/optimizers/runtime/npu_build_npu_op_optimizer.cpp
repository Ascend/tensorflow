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

#include "tensorflow/core/graph/algorithm.h"

#include "npu_device.h"
#include "npu_utils.h"
#include "optimizers/runtime/node_placer.h"
#include "optimizers/npu_optimizer_manager.h"

namespace npu {
tensorflow::Status BuildNpuOpOptimize(TFE_Context *context, NpuMutableConcreteGraph *graph,
                                      std::map<std::string, std::string> options, NpuDevice *device, int num_inputs,
                                      TFE_TensorHandle **inputs) {
  TF_UNUSED_VARIABLE(options);
  TF_UNUSED_VARIABLE(num_inputs);
  TF_UNUSED_VARIABLE(inputs);
  std::stringstream ss;
  ss << device->ValidateInputTypes(graph->ConsumedTypes()).error_message();
  ss << device->ValidateOutputTypes(graph->ProducedTypes()).error_message();
  std::set<std::string> unsupported_ops;
  NPU_REQUIRES_OK(
    GetGraphUnsupportedOps(device, graph->MutableGraph(), npu::UnwrapCtx(context)->FuncLibDef(), unsupported_ops));
  if (!unsupported_ops.empty()) {
    ss << "Unsupported ops " << SetToString(unsupported_ops);
  }
  if (!ss.str().empty()) {
    tensorflow::Node *key;
    graph->SetBuiltinLoop(IsGraphNeedLoop(graph->MutableGraph(), &key) || key != nullptr);
    graph->SetExecutionType(NpuConcreteGraph::ExecutionType::MIX);
    LOG(INFO) << graph->Op() << " compiled in mix mode on npu";
    DLOG() << graph->Op() << " not fully compiled on npu as " << std::endl << ss.str();
    NPU_REQUIRES_OK(NodePlacer(context, graph->MutableGraph(), device).Apply());
    tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
    tensorflow::FunctionDefLibrary flib;
    std::string mixed_fn = "partitioned_" + graph->Op();
    NPU_REQUIRES_OK(tensorflow::GraphToFunctionDef(*graph->MutableGraph(), mixed_fn, flib.add_function()));
    NPU_REQUIRES_OK(lib_def->AddLibrary(flib));
    DLOG() << graph->Op() << " run as mix function " << mixed_fn;
    graph->SetMixedFunctionName(mixed_fn);
  } else {
    LOG(INFO) << graph->Op() << " fully compiled on npu";
    graph->SetExecutionType(NpuConcreteGraph::ExecutionType::NPU);
    NPU_REQUIRES_OK(graph->TryTransToNpuLoopGraph(context));
    AssembleParserAddons(context, graph->MutableGraph());
  }
  return tensorflow::Status::OK();
}

NPU_REGISTER_RT_OPTIMIZER(999, "BuildNpuOpOptimizer", BuildNpuOpOptimize);
}  // namespace npu
