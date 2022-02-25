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

#include "npu_mirrored_op.h"

#include "npu_device.h"

namespace npu {
NpuMirroredOp::NpuMirroredOp(const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef,
                             TensorShapes input_shapes, NpuFallbackHookFunc custom_kernel)
    : OpExecutor(op_spec, ndef, input_shapes) {
  custom_kernel_ = custom_kernel;
}

std::string NpuMirroredOp::AttachedDebugString() const {
  std::stringstream ss;
  return ss.str();
}

void NpuMirroredOp::RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs,
                            int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  NPU_CTX_REQUIRES(status, custom_kernel_ != nullptr, tensorflow::errors::Internal(Op(), " hook func is nullptr"));
  device->FallbackCPU(context, NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
  if (TF_GetCode(status) != TF_OK) { return; }
  custom_kernel_(context, device, NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
}
}  // namespace npu
