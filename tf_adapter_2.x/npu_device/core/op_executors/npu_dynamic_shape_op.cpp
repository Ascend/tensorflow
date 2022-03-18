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

#include "npu_dynamic_shape_op.h"

#include "npu_device.h"

namespace npu {
NpuDynamicShapeOp::NpuDynamicShapeOp(const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef,
                                     TensorShapes input_shapes, TensorPartialShapes output_shapes)
    : OpExecutor(op_spec, ndef, input_shapes), output_shapes_(std::move(output_shapes)) {
  AssembleInputDesc(input_shapes_, input_dtypes_, &attached_attrs_);
  AssembleOutputDesc(output_shapes_, output_dtypes_, &attached_attrs_);
}

std::string NpuDynamicShapeOp::AttachedDebugString() const {
  std::stringstream ss;
  for (size_t i = 0; i < output_dtypes_.size(); i++) {
    ss << "output " << i << " " << tensorflow::DataTypeString(output_dtypes_[i]) << " "
       << output_shapes_[i].DebugString() << std::endl;
  }
  return ss.str();
}

void NpuDynamicShapeOp::RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs,
                                int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  device->FallbackCPU(context, NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
}
}  // namespace npu
