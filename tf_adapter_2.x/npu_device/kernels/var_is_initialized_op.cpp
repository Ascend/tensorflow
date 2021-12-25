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

#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/platform/logging.h"

#include "absl/algorithm/container.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"

#include "npu_custom_kernel.h"
#include "npu_utils.h"

static auto kernel = [](TFE_Context *context, NpuDevice *dev, const npu::OpSpec *spec,
                        const TensorShapes &output_shapes, const tensorflow::NodeDef &parser_ndef, int num_inputs,
                        TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  TF_UNUSED_VARIABLE(context);
  TF_UNUSED_VARIABLE(dev);
  TF_UNUSED_VARIABLE(spec);
  TF_UNUSED_VARIABLE(output_shapes);
  TF_UNUSED_VARIABLE(parser_ndef);
  TF_UNUSED_VARIABLE(num_inputs);
  TF_UNUSED_VARIABLE(inputs);
  TF_UNUSED_VARIABLE(num_outputs);
  TF_UNUSED_VARIABLE(status);
  // 这里需要先判断下是否已经初始化
  tensorflow::Tensor tensor(tensorflow::DT_BOOL, {});
  tensor.scalar<bool>()() = true;
  outputs[0] = tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
};

NPU_REGISTER_CUSTOM_KERNEL("VarIsInitializedOp", kernel);
