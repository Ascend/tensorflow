/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tf_adapter/common/adapter_logger.h"

namespace tensorflow {
namespace {
class GetNextOp : public OpKernel {
public:
  explicit GetNextOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("channel_name", &channel_name_));
    ADP_LOG(INFO) << "GetNextOp built " << channel_name_;
  }
  ~GetNextOp() override {
    ADP_LOG(INFO) << "GetNextOp has been destructed";
  }
  void Compute(OpKernelContext *ctx) override {
    (void) ctx;
    ADP_LOG(INFO) << "GetNextOp running";
  }

private:
  std::string channel_name_;
};

REGISTER_KERNEL_BUILDER(Name("GetNext").Device(DEVICE_CPU), GetNextOp);
}  // namespace
}  // namespace tensorflow
