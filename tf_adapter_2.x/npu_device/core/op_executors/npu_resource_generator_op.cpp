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

#include "npu_device.h"
#include "npu_managed_buffer.h"
#include "npu_resource_generator_op.h"

namespace npu {
using Format = ge::Format;
NpuResourceGeneratorOp::NpuResourceGeneratorOp(const tensorflow::OpRegistrationData *op_spec,
                                               const tensorflow::NodeDef &ndef, TensorShapes input_shapes)
    : OpExecutor(op_spec, ndef, input_shapes) {
  AssembleInputDesc(input_shapes_, input_dtypes_, &attached_attrs_);
}

std::string NpuResourceGeneratorOp::AttachedDebugString() const {
  std::stringstream ss;
  return ss.str();
}

void NpuResourceGeneratorOp::RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs,
                                     int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  if ((!device->SupportedResourceGenerator(Op())) || (!InputTypes().empty()) || (num_outputs != 1)) {
    device->FallbackCPU(context, NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
    return;
  }

  outputs[0] = device->NewDeviceResourceHandle(context, kScalarShape, status);
  if (TF_GetCode(status) != TF_OK) { return; }

  npu::ScopeTensorHandleDeleter scope_handle_deleter;
  TFE_TensorHandle *cpu_output = nullptr;
  device->FallbackCPU(context, NodeDef(), num_inputs, inputs, num_outputs, &cpu_output, status);
  if (TF_GetCode(status) != TF_OK) { return; }
  scope_handle_deleter.Guard(cpu_output);

  const tensorflow::Tensor *cpu_tensor = nullptr;
  NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(cpu_output, &cpu_tensor));
  const tensorflow::Tensor *npu_tensor = nullptr;
  NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(outputs[0], &npu_tensor));
  auto resource = cpu_tensor->flat<tensorflow::ResourceHandle>()(0);
  const_cast<tensorflow::Tensor *>(npu_tensor)->flat<tensorflow::ResourceHandle>()(0) = resource;

  auto ndef = std::make_shared<tensorflow::NodeDef>(NodeDef());

  tensorflow::SetAttrValue(resource.container(), &ndef->mutable_attr()->at("container"));
  ndef->set_name(WrapResourceName(resource.name()));
  tensorflow::SetAttrValue(ndef->name(), &ndef->mutable_attr()->at("shared_name"));

  device->RecordResourceGeneratorDef(resource, std::make_shared<ResourceGenerator>(ndef, 0));
  DLOG() << "Create resource " << Op() << " " << resource.DebugString() << " by " << ndef->DebugString() << " on NPU";
}
}  // namespace npu
