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

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
// Mixed-precisions training
class NpuAllocFloatStatusOp : public tensorflow::OpKernel {
public:
  explicit NpuAllocFloatStatusOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}
  ~NpuAllocFloatStatusOp() override = default;
  void Compute(tensorflow::OpKernelContext *context) override {
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1}), &output_tensor));
    // Set the float_status tensor to be 0
    auto flat = output_tensor->flat<float>();
    flat(0) = 0.0;
  }
};

REGISTER_KERNEL_BUILDER(Name("NpuAllocFloatStatus").Device(tensorflow::DEVICE_CPU), NpuAllocFloatStatusOp);

class NpuGetFloatStatusOp : public tensorflow::OpKernel {
public:
  explicit NpuGetFloatStatusOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}
  ~NpuGetFloatStatusOp() override = default;
  void Compute(tensorflow::OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor &input_tensor = context->input(0);
    // Create an output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    // Set the float_status tensor to be 0
    auto flat = output_tensor->flat<float>();
    // For testing
    flat(0) = 0.0;
  }
};

REGISTER_KERNEL_BUILDER(Name("NpuGetFloatStatus").Device(tensorflow::DEVICE_CPU), NpuGetFloatStatusOp);

class NpuGetFloatStatusV2Op : public OpKernel {
public:
  explicit NpuGetFloatStatusV2Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~NpuGetFloatStatusV2Op() override = default;
  void Compute(OpKernelContext *context) override {
    LOG(INFO) << "NpuGetFloatStatusV2 Compute";
  }
};

REGISTER_KERNEL_BUILDER(Name("NpuGetFloatStatusV2").Device(DEVICE_CPU), NpuGetFloatStatusV2Op);

class NpuClearFloatStatusOp : public OpKernel {
public:
  explicit NpuClearFloatStatusOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~NpuClearFloatStatusOp() override = default;
  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    const Tensor &input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
    // Create an output tensor
    Tensor *output_tensor = nullptr;
    // Clear the status
    auto flat = output_tensor->flat<float>();
    // For testing
    for (int i = 0; i < input.size(); i++) { flat(i) = 0.0; }
  }
};

REGISTER_KERNEL_BUILDER(Name("NpuClearFloatStatus").Device(DEVICE_CPU), NpuClearFloatStatusOp);

class NpuClearFloatStatusV2Op : public tensorflow::OpKernel {
public:
  explicit NpuClearFloatStatusV2Op(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}
  ~NpuClearFloatStatusV2Op() override = default;
  void Compute(tensorflow::OpKernelContext *context) override {
    LOG(INFO) << "NpuClearFloatStatusV2 Compute";
  }
};

REGISTER_KERNEL_BUILDER(Name("NpuClearFloatStatusV2").Device(tensorflow::DEVICE_CPU), NpuClearFloatStatusV2Op);
}  // namespace tensorflow
