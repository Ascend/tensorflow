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
#include "tf_adapter/kernels/aicore/amct_common.h"

using namespace tensorflow;

template <typename T>
int AscendDequantInternelCpu(struct DequantInputParam<T> input_param) {
  int bound = pow(BASE, SHIFT_POW);
  int channel_index = 0;
  for (int i = 0; i < input_param.size; i++) {
    if (input_param.channel_wise) {
      if (input_param.transpose) {
        channel_index = i % (input_param.channel_num * input_param.hw_size) / input_param.hw_size;
      } else {
        channel_index = i % input_param.channel_num;
      }
    }
    unsigned int deqscale_int = (input_param.deqscale[channel_index] << DEQ_SCALE_BINS) >> DEQ_SCALE_BINS;
    unsigned int shift_n_int = (input_param.deqscale[channel_index] << N_LFET_BINS) >> N_RIGHT_BINS;
    float* deqscale = reinterpret_cast<float*>(&(deqscale_int));
    NULLPTR_CHECK(deqscale);
    input_param.out[i] = input_param.input[i] * input_param.area_factor;
    if (shift_n_int > 0) {
      input_param.out[i] = floor(input_param.out[i] / pow(BASE, shift_n_int));
      if (input_param.out[i] > bound - 1) {
        input_param.out[i] = bound - 1;
      } else if (input_param.out[i] < -bound) {
        input_param.out[i] = -bound;
      }
    }
    input_param.out[i] = input_param.out[i] * (*deqscale) * pow(BASE, shift_n_int) / input_param.area_factor;
  }
  return 0;
}

template <typename T>
class AscendDequantOp : public OpKernel {
 public:
  explicit AscendDequantOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &(input_param.data_format)));
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &(ksize)));
      input_param.area_factor = ksize[0] * ksize[1];
      input_param.size = 0;
      input_param.input = NULL;
      input_param.out = NULL;
      input_param.deqscale = NULL;
      input_param.channel_num = 1;
      input_param.hw_size = 1;
      input_param.channel_wise = false;
      input_param.transpose = false;
  }

  ~AscendDequantOp(){}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    TensorShape input_tensor_shape = input_tensor.shape();
    const Tensor& deqscale_tensor = context->input(1);
    TensorShape deqscale_tensor_shape = deqscale_tensor.shape();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    input_param.size = static_cast<int>(input_tensor.NumElements());
    input_param.input = input_tensor.flat<T>().data();
    input_param.out = output_tensor->flat<T>().data();
    input_param.deqscale = deqscale_tensor.flat<uint64>().data();

    std::vector<int> input_shape, deqscale_shape;
    input_shape.resize(input_tensor_shape.dim_sizes().size());
    deqscale_shape.resize(deqscale_tensor_shape.dim_sizes().size());

    for (unsigned int i = 0; i < input_shape.size(); i++) {
      input_shape[i] = input_tensor_shape.dim_sizes()[i];
    }
    int deqscale_size = 1;
    for (unsigned int i = 0; i < deqscale_shape.size(); i++) {
      deqscale_shape[i] = deqscale_tensor_shape.dim_sizes()[i];
      deqscale_size *= deqscale_shape[i];
    }

    if (deqscale_size <= 1) {
      input_param.channel_wise = false;
    } else {
      input_param.channel_wise = true;
      if (input_param.data_format == "NCHW") {
        input_param.transpose = true;
        input_param.channel_num = deqscale_size;
        input_param.hw_size = input_shape[NCHW_H_DIM] * input_shape[NCHW_W_DIM];
      } else {
        input_param.transpose = false;
        input_param.channel_num = deqscale_size;
        input_param.hw_size = input_shape[NHWC_H_DIM] * input_shape[NHWC_W_DIM];
      }
    }

    if (input_param.size == 0) {
      OP_REQUIRES(context, false, errors::InvalidArgument("AscendDequantOp: input_tensor is empty!"));
    }
    int errorCode = AscendDequantInternelCpu(input_param);
    ERROR_CHECK(errorCode);
  }

 private:
  struct DequantInputParam<T> input_param;
  std::vector<int> ksize;
};

REGISTER_KERNEL_BUILDER(
  Name("AscendDequant").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"),
  AscendDequantOp<float>);
