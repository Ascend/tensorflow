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

#ifndef NPU_DEVICE_CORE_OPTIMIZERS_NPU_ALGORITHM_H
#define NPU_DEVICE_CORE_OPTIMIZERS_NPU_ALGORITHM_H

#include "npu_device.h"

namespace npu {
tensorflow::Status MarkGraphNodeInOutDesc(TFE_Context *context, tensorflow::Graph *graph, int num_inputs,
                                          TFE_TensorHandle **inputs);
}  // namespace npu
#endif  // NPU_DEVICE_CORE_OPTIMIZERS_NPU_ALGORITHM_H
