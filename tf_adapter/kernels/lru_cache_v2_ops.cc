/*
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
#include "tf_adapter/common/adp_logger.h"

namespace tensorflow {
class LRUCacheV2Op : public OpKernel {
public:
  explicit LRUCacheV2Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~LRUCacheV2Op() override = default;
  void Compute(OpKernelContext *context) override {
    (void)context;
    ADP_LOG(INFO) << "LRUCacheV2Op Compute running";
  }
  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("LRUCacheV2").Device(DEVICE_CPU), LRUCacheV2Op);
} // namespace tensorflow
