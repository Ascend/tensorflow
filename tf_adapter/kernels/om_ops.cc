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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "om_executor.h"

namespace tensorflow {
namespace {
class LoadAndExecuteOmOp : public OpKernel {
 public:
  explicit LoadAndExecuteOmOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("om_path", &om_path_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("executor_type", &executor_type_));
  }
  ~LoadAndExecuteOmOp() override = default;

  void Compute(OpKernelContext *ctx) override {
    std::unique_lock<std::mutex> lk(mu_);
    OP_REQUIRES_OK(ctx, Initialize());
    std::vector<Tensor> inputs;
    inputs.reserve(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); i++) {
      inputs.push_back(ctx->input(i));
    }

    std::vector<Tensor> outputs;
    OP_REQUIRES_OK(ctx, executor_->Execute(inputs, outputs));
    OP_REQUIRES(ctx, outputs.size() == static_cast<size_t>(ctx->num_outputs()),
                errors::Internal("Om outputs num mismatch expect ", ctx->num_outputs(), " vs. ", outputs.size()));

    for (int i = 0; i < static_cast<int>(outputs.size()); i++) {
      ctx->set_output(i, std::move(outputs[i]));
    }
  }

 private:
  Status Initialize() {
    if (initialized_) {
      return Status::OK();
    }
    // todo: 将om_path_转换为绝对路径
    TF_RETURN_IF_ERROR(OmExecutor::Create(om_path_, executor_));
    initialized_ = true;
    return Status::OK();
  }

  std::mutex mu_;
  bool initialized_{false};
  std::string om_path_;
  std::string executor_type_;  // Reserved

  std::unique_ptr<OmExecutor> executor_;
};
}  // namespace
REGISTER_KERNEL_BUILDER(Name("LoadAndExecuteOm").Device(DEVICE_CPU), LoadAndExecuteOmOp);
}  // namespace tensorflow
