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

#include "npu_static_shape_op.h"

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include "npu_device.h"
#include "npu_managed_buffer.h"

namespace npu {
using Format = ge::Format;
NpuStaticShapeOp::NpuStaticShapeOp(const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef,
                                   TensorShapes input_shapes, TensorShapes output_shapes)
    : OpExecutor(op_spec, ndef, input_shapes) {
  output_shapes_ = std::move(output_shapes);

  AssembleInputDesc(input_shapes_, input_dtypes_, &attached_attrs_);
  AssembleOutputDesc(output_shapes_, output_dtypes_, &attached_attrs_);
}

std::string NpuStaticShapeOp::AttachedDebugString() const {
  std::stringstream ss;
  for (size_t i = 0; i < output_dtypes_.size(); i++) {
    ss << "output " << i << " " << tensorflow::DataTypeString(output_dtypes_[i]) << " "
       << output_shapes_[i].DebugString() << std::endl;
  }
  return ss.str();
}

void NpuStaticShapeOp::RunWithShape(TFE_Context *context, NpuDevice *device, const OpExecutor *spec,
                                    TensorShapes output_shapes, int num_inputs, TFE_TensorHandle **inputs,
                                    int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  if (kGraphEngineGreedyMemory) {
    DLOG() << "NPU Executing op " << spec->Op() << " fallback cpu in graph engine greedy memory mode";
    device->FallbackCPU(context, spec->NodeDef(), num_inputs, inputs, num_outputs, outputs, status);
    return;
  }
  // 输入如果是CPU,此时要转换成NPU
  std::vector<TFE_TensorHandle *> npu_inputs(num_inputs);
  ScopeTensorHandleDeleter scope_handle_deleter;
  for (int i = 0; i < num_inputs; ++i) {
    TFE_TensorHandle *input = inputs[i];
    // 到达这里的Resource，要么是CPU的镜像 要么是NPU
    if (!npu::IsNpuTensorHandle(input)) {
      tensorflow::Status s;
      auto src_name = tensorflow::unwrap(input)->DeviceName(&s);
      NPU_CTX_REQUIRES_OK(status, s);
      DLOG() << "Copying " << spec->Op() << " input:" << i
             << " type:" << tensorflow::DataTypeString(tensorflow::unwrap(input)->DataType()) << " to NPU from "
             << src_name << " for acl executing";
      // 这里需要根据算子选择输入格式了
      input = device->CopyTensorH2D(context, input, Format::FORMAT_ND, status);
      scope_handle_deleter.Guard(input);
      if (TF_GetCode(status) != TF_OK) return;
    }
    npu_inputs[i] = input;
  }
  const auto &output_types = spec->OutputTypes();
  for (size_t i = 0; i < output_types.size(); ++i) {
    outputs[i] = device->NewDeviceTensorHandle(context, Format::FORMAT_ND, output_shapes[i], output_types[i], status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  /******************************************模拟NPU执行Start************************************/
  std::vector<TFE_TensorHandle *> acl_inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const tensorflow::Tensor *npu_tensor = nullptr;
    NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(npu_inputs[i], &npu_tensor));
    tensorflow::Tensor cpu_tensor(npu_tensor->dtype(), npu_tensor->shape());
    if (npu_tensor->dtype() == tensorflow::DT_RESOURCE) {
      for (int j = 0; j < npu_tensor->NumElements(); j++) {
        cpu_tensor.flat<tensorflow::ResourceHandle>()(j) = npu_tensor->flat<tensorflow::ResourceHandle>()(j);
      }
    } else {
      NPU_CTX_REQUIRES_OK(status, npu::Unwrap<npu::NpuManagedBuffer>(npu_tensor)->AssembleTo(&cpu_tensor));
    }
    acl_inputs[i] = tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(cpu_tensor));
    scope_handle_deleter.Guard(acl_inputs[i]);
    if (TF_GetCode(status) != TF_OK) return;
  }
  /**********调用CPU模拟NPU Start*************/
  std::vector<TFE_TensorHandle *> acl_outputs(num_outputs);
  device->FallbackCPU(context, spec->NodeDef(), num_inputs, acl_inputs.data(), num_outputs, acl_outputs.data(), status);
  if (TF_GetCode(status) != TF_OK) return;
  /**********调用CPU模拟NPU End***************/
  for (int i = 0; i < num_outputs; ++i) {
    const tensorflow::Tensor *acl_tensor = nullptr;
    NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(acl_outputs[i], &acl_tensor));
    const tensorflow::Tensor *npu_tensor = nullptr;
    NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(outputs[i], &npu_tensor));
    NPU_CTX_REQUIRES_OK(status, npu::Unwrap<npu::NpuManagedBuffer>(npu_tensor)->AssembleFrom(acl_tensor));
    TFE_DeleteTensorHandle(acl_outputs[i]);
    if (TF_GetCode(status) != TF_OK) return;
  }
  /******************************************模拟NPU执行End************************************/
  DLOG() << "NPU Executing op " << spec->Op() << " succeed by npu executor";
}

void NpuStaticShapeOp::RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs,
                               int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  NpuStaticShapeOp::RunWithShape(context, device, this, OutputShapes(), num_inputs, inputs, num_outputs, outputs,
                                 status);
}
}  // namespace npu
