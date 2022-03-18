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
#include "tf_adapter/common/adp_logger.h"

namespace tensorflow {
class KMeansCentroidsV2Op : public OpKernel {
 public:
  explicit KMeansCentroidsV2Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~KMeansCentroidsV2Op() override = default;
  void Compute(OpKernelContext *context) override {
    (void) context;
    ADP_LOG(INFO) << "KMeansCentroidsV2Op Compute";
  }
  bool IsExpensive() override {
    return false;
  }
};

REGISTER_KERNEL_BUILDER(Name("KMeansCentroidsV2").Device(DEVICE_CPU), KMeansCentroidsV2Op);
}  // namespace tensorflow
