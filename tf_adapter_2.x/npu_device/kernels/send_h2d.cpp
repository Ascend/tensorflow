/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: Common depends and micro defines for and only for data preprocess module
 */

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/env_var.h"

#include "npu_hdc.h"

using namespace tensorflow;

class SendH2D : public OpKernel {
 public:
  explicit SendH2D(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("channel_name", &channel_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ids", &device_ids_));
  }

  void Compute(OpKernelContext *ctx) override {
    if (!initialized_.exchange(true)) {
      std::stringstream ss;
      for (auto device_id : device_ids_) {
        ss << device_id << " ";
      }
      channels_.resize(device_ids_.size());
      for (size_t i = 0; i < device_ids_.size(); i++) {
        OP_REQUIRES_OK(ctx, HdcChannel::Create(device_ids_[i], channel_name_, &channels_[i]));
      }
      LOG(INFO) << "Hdc channel for iterator resource " << channel_name_ << " to device ["
                << ss.str().substr(0, ss.str().size() - 1) << "] created";
    }
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));
    std::vector<Tensor> tensors;
    for (int64 i = 0; i < inputs.size(); i++) {
      tensors.push_back(inputs[i]);
    }
    for (auto channel : channels_) {
      OP_REQUIRES_OK(ctx, channel->SendTensors(tensors));
    }
  }

 private:
  std::string channel_name_;
  std::vector<int> device_ids_;
  std::vector<std::shared_ptr<HdcChannel>> channels_;
  std::atomic_bool initialized_{false};
};

REGISTER_KERNEL_BUILDER(Name("SendH2D").Device(DEVICE_CPU).Priority(3), SendH2D);
