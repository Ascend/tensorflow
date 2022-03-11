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
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace {
class NpuOnnxGraphOp : public OpKernel {
 public:
  explicit NpuOnnxGraphOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~NpuOnnxGraphOp() override = default;
  void Compute(OpKernelContext *context) override {
    (void) context;
    return;
  }
  bool IsExpensive() override { return false; }
};

class FileConstant : public OpKernel {
 public:
  explicit FileConstant(OpKernelConstruction *context) : OpKernel(context) {}
  ~FileConstant() override = default;
  void Compute(OpKernelContext *context) override {
    return;
  }
  bool IsExpensive() override {
    return false;
  }
};

REGISTER_KERNEL_BUILDER(Name("NpuOnnxGraphOp").Device(DEVICE_CPU), NpuOnnxGraphOp);
REGISTER_KERNEL_BUILDER(Name("FileConstant").Device(DEVICE_CPU), FileConstant);
}  // namespace
}  // namespace tensorflow
