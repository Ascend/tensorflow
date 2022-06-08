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
#include "npu_utils.h"

namespace npu {
static const auto kernel = [](TFE_Context *context, NpuDevice *dev, const tensorflow::NodeDef &ndef, int num_inputs,
                              TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  if (!IsNpuTensorHandle(inputs[0])) {
    dev->FallbackCPU(context, ndef, num_inputs, inputs, num_outputs, outputs, status);
    return;
  }

  const tensorflow::Tensor *npu_tensor = nullptr;
  NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(inputs[0], &npu_tensor));

  tensorflow::Tensor cpu_tensor(npu_tensor->dtype(), npu_tensor->shape());
  for (int j = 0; j < npu_tensor->NumElements(); j++) {
    cpu_tensor.flat<tensorflow::ResourceHandle>()(j) =
      const_cast<tensorflow::Tensor *>(npu_tensor)->flat<tensorflow::ResourceHandle>()(j);
  }

  npu::ScopeTensorHandleDeleter scope_handle_deleter;
  auto handle = tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(cpu_tensor));
  scope_handle_deleter.Guard(handle);

  dev->FallbackCPU(context, ndef, num_inputs, &handle, num_outputs, outputs, status);
};

NPU_REGISTER_CUSTOM_KERNEL("DestroyResourceOp", kernel);
}  // namespace npu
