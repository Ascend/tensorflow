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
int AscendWeightQuantInternelCpu(struct WeightQuantInputParam<T> input_param)
{
  if (input_param.channel_wise) {
    if (input_param.transpose) {
      for (int i = 0; i < input_param.size; i++) {
        int index = i % (input_param.channel_in_num * input_param.channel_out_num) / input_param.channel_in_num;
        input_param.out[i] = static_cast<T>(input_param.weight[i]) - static_cast<T>(input_param.offset[index]);
      }
    } else {
      for (int i = 0; i < input_param.size; i++) {
        int index = i % input_param.channel_out_num;
        input_param.out[i] = static_cast<T>(input_param.weight[i]) - static_cast<T>(input_param.offset[index]);
      }
    }
  } else {
    for (int i = 0; i < input_param.size; i++) {
      input_param.out[i] = static_cast<T>(input_param.weight[i]) - static_cast<T>(input_param.offset[0]);
    }
  }
  return 0;
}

template <typename T>
class AscendWeightQuantOp : public OpKernel {
 public:
  explicit AscendWeightQuantOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dst_type", &(dst_type)));
    input_param.size = 0;
    input_param.weight = NULL;
    input_param.offset = NULL;
    input_param.channel_in_num = 1;
    input_param.channel_out_num = 1;
    input_param.channel_wise = false;
    input_param.transpose = false;
  }

  ~AscendWeightQuantOp(){}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    TensorShape input_tensor_shape = input_tensor.shape();
    const Tensor& offset_tensor = context->input(1);
    TensorShape offset_tensor_shape = offset_tensor.shape();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    input_param.size = static_cast<int>(input_tensor.NumElements());
    input_param.weight = input_tensor.flat<int8>().data();
    input_param.offset = offset_tensor.flat<int8>().data();
    input_param.out = output_tensor->flat<T>().data();

    std::vector<int> weight_shape, offset_shape;
    weight_shape.resize(input_tensor_shape.dim_sizes().size());
    offset_shape.resize(offset_tensor_shape.dim_sizes().size());

    for (unsigned int i = 0; i < weight_shape.size(); i++) {
      weight_shape[i] = input_tensor_shape.dim_sizes()[i];
    }
    for (unsigned int i = 0; i < offset_shape.size(); i++) {
      offset_shape[i] = offset_tensor_shape.dim_sizes()[i];
    }

    int offset_size = 1;
    for (unsigned int i = 0; i < offset_shape.size(); i++) {
      offset_shape[i] = offset_tensor_shape.dim_sizes()[i];
      offset_size *= offset_shape[i];
    }
    if (offset_size <= 1) {
      input_param.channel_wise = false;
    } else {
      input_param.channel_wise = true;
      if (offset_shape[CIN_DIM] > 1) {
        input_param.transpose = true;
        input_param.channel_in_num = weight_shape[COUT_DIM];
        input_param.channel_out_num = weight_shape[CIN_DIM];
      } else {
        input_param.transpose = false;
        input_param.channel_in_num = weight_shape[CIN_DIM];
        input_param.channel_out_num = weight_shape[COUT_DIM];
      }
    }

    if (input_param.size == 0) {
      OP_REQUIRES(context, false, errors::InvalidArgument("AscendWeightQuantOp: input_tensor is empty!"));
    }
    AscendWeightQuantInternelCpu(input_param);
  }

 private:
  struct WeightQuantInputParam<T> input_param;
  std::string dst_type;
};

REGISTER_KERNEL_BUILDER(
  Name("AscendWeightQuant").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"),
  AscendWeightQuantOp<float>);
