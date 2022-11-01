/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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
#include "tf_adapter/common/adapter_logger.h"

namespace tensorflow {
class DropOutDoMaskOp : public OpKernel {
 public:
  explicit DropOutDoMaskOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DropOutDoMaskOp() override {}
  void Compute(OpKernelContext *context) override {
    (void) context;
    ADP_LOG(INFO) << "DropOutDoMaskOp Compute ";
  }
  bool IsExpensive() override {
    return false;
  }
};

class DropOutGenMaskOp : public OpKernel {
 public:
  explicit DropOutGenMaskOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DropOutGenMaskOp() override {}
  void Compute(OpKernelContext *context) override {
    (void) context;
    ADP_LOG(INFO) << "DropOutGenMaskOp Compute";
  }
  bool IsExpensive() override {
    return false;
  }
};

REGISTER_KERNEL_BUILDER(Name("DropOutGenMask").Device(DEVICE_CPU), DropOutGenMaskOp);
REGISTER_KERNEL_BUILDER(Name("DropOutGenMaskV3").Device(DEVICE_CPU), DropOutGenMaskOp);
REGISTER_KERNEL_BUILDER(Name("DropOutDoMask").Device(DEVICE_CPU), DropOutDoMaskOp);
REGISTER_KERNEL_BUILDER(Name("DropOutDoMaskV3").Device(DEVICE_CPU), DropOutDoMaskOp);
}  // namespace tensorflow