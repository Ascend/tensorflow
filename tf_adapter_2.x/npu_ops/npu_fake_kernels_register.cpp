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

namespace tensorflow {
namespace {
class FakeOp : public AsyncOpKernel {
 public:
  explicit FakeOp(OpKernelConstruction *context) : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext *context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(
      context, errors::Internal(context->op_kernel().name(), " registered as fake op and should never run on cpu"),
      done);
  }
};
}  // namespace

REGISTER_KERNEL_BUILDER(Name("HcomAllReduce").Device(DEVICE_CPU), FakeOp);
REGISTER_KERNEL_BUILDER(Name("HcomAllGather").Device(DEVICE_CPU), FakeOp);
REGISTER_KERNEL_BUILDER(Name("HcomBroadcast").Device(DEVICE_CPU), FakeOp);
REGISTER_KERNEL_BUILDER(Name("HcomReduce").Device(DEVICE_CPU), FakeOp);
REGISTER_KERNEL_BUILDER(Name("HcomReduceScatter").Device(DEVICE_CPU), FakeOp);
REGISTER_KERNEL_BUILDER(Name("HcomSend").Device(DEVICE_CPU), FakeOp);
REGISTER_KERNEL_BUILDER(Name("HcomReceive").Device(DEVICE_CPU), FakeOp);
REGISTER_KERNEL_BUILDER(Name("HcomRemoteRead").Device(DEVICE_CPU), FakeOp);
REGISTER_KERNEL_BUILDER(Name("HcomRemoteRefRead").Device(DEVICE_CPU), FakeOp);
REGISTER_KERNEL_BUILDER(Name("HcomRemoteWrite").Device(DEVICE_CPU), FakeOp);
REGISTER_KERNEL_BUILDER(Name("HcomRemoteScatterWrite").Device(DEVICE_CPU), FakeOp);
}  // namespace tensorflow