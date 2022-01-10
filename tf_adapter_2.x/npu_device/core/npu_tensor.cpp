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

#include "npu_tensor.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

namespace npu {
NpuTensor::NpuTensor(const tensorflow::Tensor& tensor)
    : handle(tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor))) {}

NpuTensor::~NpuTensor() { TFE_DeleteTensorHandle(handle); }

int64_t NpuTensor::Dim(void* data, int dim_index, TF_Status* status) {
  return TFE_TensorHandleDim(reinterpret_cast<NpuTensor*>(data)->handle, dim_index, status);
}

int NpuTensor::NumDims(void* data, TF_Status* status) {
  return TFE_TensorHandleNumDims(reinterpret_cast<NpuTensor*>(data)->handle, status);
}

void NpuTensor::Deallocator(void* data) { delete reinterpret_cast<NpuTensor*>(data); }

TFE_CustomDeviceTensorHandle NpuTensor::handle_methods = []() {
  TFE_CustomDeviceTensorHandle handle_methods;
  handle_methods.num_dims = &NpuTensor::NumDims;
  handle_methods.dim = &NpuTensor::Dim;
  handle_methods.deallocator = &NpuTensor::Deallocator;
  return handle_methods;
}();
}  // namespace npu