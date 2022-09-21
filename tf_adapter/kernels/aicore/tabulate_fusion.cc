/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
template <typename T>
class TabulateFusionOp : public OpKernel {
public:
  explicit TabulateFusionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    LOG(INFO) << "new TabulateFusionOp";
  }
  ~TabulateFusionOp() override = default;
  void Compute(OpKernelContext* ctx) override {
    (void)ctx;
    LOG(INFO) << "in TabulateFusionOp";
  }
  bool IsExpensive() override {
    return false;
  }
};

REGISTER_KERNEL_BUILDER(Name("TabulateFusion").Device(DEVICE_CPU), TabulateFusionOp<float>);
}  // namespace tensorflow