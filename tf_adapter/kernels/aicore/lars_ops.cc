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

#include "tf_adapter/common/adp_logger.h"

namespace tensorflow {
template<typename T>
class LarsOp : public OpKernel {
 public:
  explicit LarsOp(OpKernelConstruction *context) : OpKernel(context) { ADP_LOG(INFO) << "new LarsOp"; }
  ~LarsOp() override { ADP_LOG(INFO) << "del LarsOp"; }

  void Compute(OpKernelContext *context) override {
    int32_t input_num = num_inputs();
    ADP_LOG(INFO) << "LarsOp: input num " << input_num;
    input_num = ((input_num - 1) / 2);

    for (int32_t j = 0; j < input_num; j++) {
      // Grab the w_input tensor
      const Tensor &w_tensor = context->input(j);
      auto w_input = w_tensor.flat<T>();

      const Tensor &g_tensor = context->input(j + input_num);
      auto g_input = g_tensor.flat<T>();

      // Create an output tensor
      Tensor *output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(j, w_tensor.shape(), &output_tensor));
      // handle any data type for w_input and output
      auto output_flat = output_tensor->flat<T>();

      // Set the value of each element
      const int32_t N = static_cast<int32_t>(w_input.size());
      ADP_LOG(INFO) << "LarsOp idx " << j << ", data num " << N;

      auto sum_w = w_input(0);
      auto sum_g = g_input(0);
      for (int32_t i = 1; i < N; i++) {
        auto w = w_input(i);
        sum_w += w;
        ADP_LOG(INFO) << "LarsOp w " << w << ", sum_w " << sum_w;

        auto g = g_input(i);
        sum_g += g;
        ADP_LOG(INFO) << "LarsOp g " << g << ", sum_g " << sum_g;
      }

      auto w_norm = sqrt(sum_w);
      auto g_norm = sqrt(sum_g);
      auto b = g_norm + w_norm + T(0.00001);

      for (int32_t i = 1; i < N; i++) {
        auto w = w_input(i);
        auto g = g_input(i);
        output_flat(i) = b * (g + w);
      }
    }

    ADP_LOG(INFO) << "in LarsOp";
  }
  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("LARS").Device(DEVICE_CPU).TypeConstraint<float>("T"), LarsOp<float>);
}  // namespace tensorflow
