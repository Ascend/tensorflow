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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tf_adapter/kernels/amct_common.h"

using namespace tensorflow;

template <typename T>
int AscendQuantInternelCpu(struct QuantInputParam<T> input_param) {
  int bound = pow(BASE, input_param.quant_bits - 1);
  for (int i = 0; i < input_param.size; i++) {
    float quant_input = round(input_param.in[i] * input_param.scale) + input_param.offset;
    if (quant_input < -bound) {
      quant_input = -bound;
    } else if (quant_input > bound - 1) {
      quant_input = bound - 1;
    }
    input_param.out[i] = quant_input - input_param.offset;
  }
  return 0;
}

template <typename T>
class AscendQuantOp : public OpKernel {
 public:
  explicit AscendQuantOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dst_type", &(dst_type)));
    OP_REQUIRES_OK(context, context->GetAttr("scale", &(scale)));
    OP_REQUIRES_OK(context, context->GetAttr("offset", &(offset)));
    input_param.size = 0;
    input_param.in = NULL;
    input_param.out = NULL;
    input_param.scale = scale;
    input_param.offset = offset;
    if (dst_type == "INT4") {
      input_param.quant_bits = 4;
    } else if (dst_type == "INT8") {
      input_param.quant_bits = 8;
    }
  }

  ~AscendQuantOp(){}

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
      OP_REQUIRES(context, false, errors::InvalidArgument("AscendQuantOp: input_tensor is empty!"));
    }
    AscendQuantInternelCpu(input_param);
  }

 private:
  struct QuantInputParam<T> input_param;
  std::string dst_type;
  float scale;
  float offset;
};

REGISTER_KERNEL_BUILDER(
  Name("AscendQuant").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"),
  AscendQuantOp<float>);
