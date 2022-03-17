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
class FastGeluV2Op : public OpKernel {
 public:
  explicit FastGeluV2Op(OpKernelConstruction *context) : OpKernel(context) {
    LOG(INFO) << "new FastGeluV2Op";
  }
  ~FastGeluV2Op() {
    LOG(INFO) << "del FastGeluV2Op";
  }
  void Compute(OpKernelContext *context) override {
    (void) context;
    LOG(INFO) << "FastGeluV2Op Compute";
  }
  bool IsExpensive() override {
    LOG(INFO) << "in FastGeluV2 IsExpensive";
    return false;
  }
};

REGISTER_KERNEL_BUILDER(Name("FastGeluV2").Device(DEVICE_CPU), FastGeluV2Op);
}  // namespace tensorflow
