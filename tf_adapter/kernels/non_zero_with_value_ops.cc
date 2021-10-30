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
template <typename T> class NonZeroWithValueOP : public OpKernel {
public:
  explicit NonZeroWithValueOP(OpKernelConstruction *ctx) : OpKernel(ctx) {
    LOG(INFO) << "new NonZeroWithValueOP";
  }
  ~NonZeroWithValueOP() { LOG(INFO) << "del NonZeroWithValueOP"; }
  void Compute(OpKernelContext *ctx) override {
    LOG(INFO) << "compute in NonZeroWithValueOP";
  }
  bool IsExpensive() override {
    LOG(INFO) << "in NonZeroWithValue IsExpensive";
    return false;
  }
};

REGISTER_KERNEL_BUILDER(Name("NonZeroWithValue").Device(DEVICE_CPU),
                        NonZeroWithValueOP<float>);
} // namespace tensorflow