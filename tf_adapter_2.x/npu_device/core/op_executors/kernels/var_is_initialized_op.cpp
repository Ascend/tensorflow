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

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include "op_executors/npu_kernel_registry.h"

namespace npu {
static auto kernel = [](TFE_Context *context, NpuDevice *dev, const tensorflow::NodeDef &ndef, int num_inputs,
                        TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  if (!IsNpuTensorHandle(inputs[0])) {
    dev->FallbackCPU(context, ndef, num_inputs, inputs, num_outputs, outputs, status);
    return;
  }
  // 这里需要先判断下是否已经初始化
  tensorflow::Tensor tensor(tensorflow::DT_BOOL, {});
  tensor.scalar<bool>()() = true;
  outputs[0] = tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
};

NPU_REGISTER_CUSTOM_KERNEL("VarIsInitializedOp", kernel);
}  // namespace npu