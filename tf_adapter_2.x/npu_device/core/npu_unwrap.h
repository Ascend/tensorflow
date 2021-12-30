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

#ifndef TENSORFLOW_NPU_UNWRAP_H
#define TENSORFLOW_NPU_UNWRAP_H

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/shape_inference.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"

#include "npu_managed_buffer.h"

namespace npu {
template <typename T>
static inline NpuManagedBuffer *Unwrap(const tensorflow::Tensor *tensor) {
  return reinterpret_cast<T *>(const_cast<char *>(tensor->tensor_data().data()));
}

TF_ATTRIBUTE_UNUSED static inline tensorflow::EagerContext *UnwrapCtx(TFE_Context *context) {
  return tensorflow::ContextFromInterface(tensorflow::unwrap(context));
}

static inline tensorflow::TensorHandle *UnwrapHandle(TFE_TensorHandle *tensor_handle) {
  return tensorflow::TensorHandleFromInterface(tensorflow::unwrap(tensor_handle));
}

TF_ATTRIBUTE_UNUSED static inline tensorflow::EagerOperation *UnwrapOp(TFE_Op *op) {
  return reinterpret_cast<tensorflow::EagerOperation *>(tensorflow::unwrap(op));
}

TF_ATTRIBUTE_UNUSED static inline tensorflow::Status UnwrapTensor(TFE_TensorHandle *tensor_handle,
                                                                  const tensorflow::Tensor **tensor) {
  return UnwrapHandle(tensor_handle)->Tensor(tensor);
}

}  // namespace npu

#endif  // TENSORFLOW_NPU_UNWRAP_H
