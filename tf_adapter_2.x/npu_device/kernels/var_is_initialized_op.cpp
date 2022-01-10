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

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include "npu_custom_kernel.h"

namespace npu {
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
}  // namespace npu
