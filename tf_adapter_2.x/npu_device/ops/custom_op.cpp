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

using namespace tensorflow;

class FakeOp : public AsyncOpKernel {
 public:
  explicit FakeOp(OpKernelConstruction *context) : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext *context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(
        context, errors::Internal(context->op_kernel().name(), " registered as fake op and should never run on cpu"),
        done);
  }
};

REGISTER_OP("DPOP")
    .Input("inputs: Tin")
    .Output("outputs: Tout")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("function: func")
    .Attr("data_format: { 'NHWC', 'NCHW'} = 'NHWC'")
    .SetIsStateful();

REGISTER_OP("DeviceQueueDataset")
    .Output("handle: variant")
    .Attr("channel_name: string")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("SendH2D")
    .Input("inputs: Tin")
    .Attr("channel_name: string")
    .Attr("device_ids: list(int)")
    .Attr("Tin: list(type) = [DT_FLOAT, DT_HALF, DT_INT8, DT_INT32, DT_UINT8, DT_INT16, DT_UINT16, DT_UINT32, "
          "DT_INT64, DT_UINT64, DT_DOUBLE, DT_BOOL, DT_STRING]")
    .SetIsStateful();

REGISTER_OP("IteratorH2D")
    .Input("input: resource")
    .Attr("channel_name: string")
    .Attr("device_ids: list(int)")
    .SetIsStateful();

REGISTER_KERNEL_BUILDER(Name("DPOP").Device(DEVICE_CPU).Priority(3), FakeOp);
REGISTER_KERNEL_BUILDER(Name("DeviceQueueDataset").Device(DEVICE_CPU).Priority(3), FakeOp);
