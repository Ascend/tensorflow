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
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tf_adapter/common/adp_logger.h"

namespace tensorflow {
namespace {
class MaxPoolingGradGradWithArgmaxOp : public OpKernel {
 public:
  explicit MaxPoolingGradGradWithArgmaxOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    ADP_LOG(INFO) << "MaxPoolingGradGradWithArgmaxOp built";
  }
  ~MaxPoolingGradGradWithArgmaxOp() override { ADP_LOG(INFO) << "MaxPoolingGradGradWithArgmaxOp has been destructed"; }
  void Compute(OpKernelContext* ctx) override {
    (void) ctx;
    ADP_LOG(INFO) << "[ATTENTION] MaxPoolingGradGradWithArgmaxOp can not run on cpu, \
                  only running on npu, please open use_off_line ";
  }
  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("MaxPoolGradGradWithArgmax").Device(DEVICE_CPU), MaxPoolingGradGradWithArgmaxOp);
}  // namespace
}  // namespace tensorflow
