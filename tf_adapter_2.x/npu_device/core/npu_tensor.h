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

#ifndef NPU_DEVICE_CORE_NPU_TENSOR_H
#define NPU_DEVICE_CORE_NPU_TENSOR_H

#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/framework/tensor.h"

namespace npu {
struct NpuTensor {
  TF_DISALLOW_COPY_AND_ASSIGN(NpuTensor);

  TFE_TensorHandle* handle;

  explicit NpuTensor(const tensorflow::Tensor& tensor);

  ~NpuTensor();

  static int64_t Dim(void* data, int dim_index, TF_Status* status);

  static int NumDims(void* data, TF_Status* status);

  static void Deallocator(void* data);

  static TFE_CustomDeviceTensorHandleMethods handle_methods;
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_TENSOR_H