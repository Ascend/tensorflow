/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
#include "tf_adapter/common/adp_logger.h"

namespace tensorflow {
class LayerNormOp : public OpKernel {
 public:
  explicit LayerNormOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~LayerNormOp() override = default;
  void Compute(OpKernelContext *context) override {
    ADP_LOG(INFO) << "LayerNormOp Compute, num_inputs: " << context->num_inputs();
  }
  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("LayerNorm").Device(DEVICE_CPU), LayerNormOp);
}  // namespace tensorflow

