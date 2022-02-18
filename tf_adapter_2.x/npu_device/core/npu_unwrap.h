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

#ifndef NPU_DEVICE_CORE_NPU_UNWRAP_H
#define NPU_DEVICE_CORE_NPU_UNWRAP_H

#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"

#include "tensorflow/core/common_runtime/eager/custom_device.h"

namespace npu {
template <typename T>
static T *Unwrap(const tensorflow::Tensor *tensor) {
  return reinterpret_cast<T *>(const_cast<char *>(tensor->tensor_data().data()));
}

static inline tensorflow::EagerContext *UnwrapCtx(TFE_Context *context) {
  return tensorflow::ContextFromInterface(tensorflow::unwrap(context));
}

static const tensorflow::AttrBuilder *UnwrapAttrs(const TFE_OpAttrs *attrs) {
  return static_cast<const tensorflow::AttrBuilder *>(tensorflow::unwrap(attrs));
}

static bool IsNpuTensorHandle(TFE_TensorHandle *handle) {
  return tensorflow::CustomDeviceTensorHandle::classof(tensorflow::unwrap(handle));
}

static bool IsCpuTensorHandle(TFE_TensorHandle *handle) {
  return !tensorflow::CustomDeviceTensorHandle::classof(tensorflow::unwrap(handle));
}

tensorflow::Status GetTensorHandleShape(TFE_TensorHandle *handle, tensorflow::TensorShape &shape);

tensorflow::Status GetTensorHandleTensor(TFE_TensorHandle *handle, const tensorflow::Tensor **tensor);
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_UNWRAP_H
