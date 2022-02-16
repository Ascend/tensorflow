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
#include "optimizers/npu_algorithm.h"
#include "optimizers/npu_optimizer_manager.h"

namespace npu {
tensorflow::Status MarkNodeInoutDescOptimize(TFE_Context *context, NpuMutableConcreteGraph *graph,
                                             std::map<std::string, std::string> options, NpuDevice *device,
                                             int num_inputs, TFE_TensorHandle **inputs) {
  return MarkGraphNodeInOutDesc(context, graph->MutableGraph(), num_inputs, inputs);
}

NPU_REGISTER_RT_OPTIMIZER(2, "MarkNodeInoutDescOptimizer", MarkNodeInoutDescOptimize);
}  // namespace npu
