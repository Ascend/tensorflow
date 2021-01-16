/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tf_adapter/kernels/amct_common.h"

using namespace tensorflow;

template <typename T>
int AscendAntiQuantInternelCpu(struct AntiQuantInputParam<T> input_param) {
  for (int i = 0; i < input_param.size; i++) {
    input_param.out[i] = input_param.in[i] * input_param.scale;
  }
  return 0;
}

template <typename T>
class AscendAntiQuantOp : public OpKernel {
 public:
  explicit AscendAntiQuantOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &(scale)));
    OP_REQUIRES_OK(context, context->GetAttr("offset", &(offset)));
    input_param.size = 0;
    input_param.in = NULL;
    input_param.out = NULL;
    input_param.scale = scale;
    input_param.offset = offset;
  }

  ~AscendAntiQuantOp(){}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    input_param.size = static_cast<int>(input_tensor.NumElements());
    input_param.in = input_tensor.flat<T>().data();
    input_param.out = output_tensor->flat<T>().data();

    if (input_param.size == 0) {
      OP_REQUIRES(context, false, errors::InvalidArgument("AscendAntiQuantOp: input_tensor is empty!"));
    }
    AscendAntiQuantInternelCpu(input_param);
  }

 private:
  struct AntiQuantInputParam<T> input_param;
  float scale;
  float offset;
};

REGISTER_KERNEL_BUILDER(
  Name("AscendAntiQuant").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"),
  AscendAntiQuantOp<float>);
