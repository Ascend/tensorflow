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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
template <typename T> class LayerNormGradOp : public OpKernel {
public:
  explicit LayerNormGradOp(OpKernelConstruction *context) : OpKernel(context) {
    LOG(INFO) << "new LayerNormGradOp";
  }
  ~LayerNormGradOp() {
    LOG(INFO) << "del LayerNormGradOp";
  }
  void Compute(OpKernelContext *context) override {
    LOG(INFO) << "LayerNormGradOp Compute, num_inputs: " << context->num_inputs();
  }
  bool IsExpensive() override {
    LOG(INFO) << "in LayerNormGrad IsExpensive";
    return false; }
};

REGISTER_KERNEL_BUILDER(Name("LayerNormGrad").Device(DEVICE_CPU), LayerNormGradOp<float>);
}  // namespace tensorflow

