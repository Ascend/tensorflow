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

#include "npu_shape_depend_on_value_op.h"

#include "npu_device.h"
#include "npu_static_shape_op.h"

namespace npu {
std::string NpuShapeDependOnValueOp::AttachedDebugString() const {
  std::stringstream ss;
  return ss.str();
}

void NpuShapeDependOnValueOp::RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs,
                                      TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs,
                                      TF_Status *status) const {
  TensorShapes output_shapes;
  DLOG() << "NPU Executing op " << Op() << " need re-infer shape";
  TensorPartialShapes partial_shapes;
  bool unused = false;
  bool should_fallback =
    !device->InferShape(context, OpRegistrationData(), NodeDef(), num_inputs, inputs, partial_shapes, unused).ok();
  if (!should_fallback) {
    output_shapes.resize(partial_shapes.size());
    for (size_t i = 0; i < partial_shapes.size(); i++) {
      DLOG() << "NPU Executing op " << Op() << " re-infer shape output " << i << partial_shapes[i].DebugString();
      if (!partial_shapes[i].AsTensorShape(&output_shapes[i])) {
        should_fallback = true;
        break;
      }
    }
  }
  if (should_fallback) {
    DLOG() << "NPU Executing op " << Op() << " fallback cpu after re-infer shape";
    device->FallbackCPU(context, NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
    return;
  }

  NpuStaticShapeOp::RunWithShape(context, device, this, output_shapes, num_inputs, inputs, num_outputs, outputs,
                                 status);
}
}  // namespace npu
