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

#include "npu_ops.h"

namespace tensorflow {
  using shape_inference::DimensionHandle;
  using shape_inference::InferenceContext;
  using shape_inference::ShapeHandle;

  REGISTER_OP("HcomAllReduce")
      .Input("input: T")
      .Output("output: T")
      .Attr("T: {int8, int16, int32, float16, float32}")
      .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
      .Attr("group: string")
      .Attr("fusion: int")
      .Attr("fusion_id: int")
      .SetIsStateful()
      .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
      })
      .Doc(R"doc(
Outputs a tensor containing the reduction across all input tensors passed to ops.

The graph should be constructed so if one op runs with shared_name value `c`,
then `num_devices` ops will run with shared_name value `c`.  Failure to do so
will cause the graph execution to fail to complete.

input: the input to the reduction
output: the value of the reduction across all `num_devices` devices.
reduction: the reduction operation to perform.
group: all devices of the group participating in this reduction.
)doc");

  REGISTER_OP("HcomAllGather")
      .Input("input: T")
      .Output("output: T")
      .Attr("T: {int8, int16, int32, float16, float32, int64, uint64}")
      .Attr("group: string")
      .Attr("rank_size: int")
      .SetIsStateful()
      .SetShapeFn([](shape_inference::InferenceContext *c) {
        int rankSize = 0;
        TF_CHECK_OK(c->GetAttr("rank_size", &rankSize));
        Status rankSizeStatus =
            ((rankSize > 0) ? (Status::OK()) : (errors::InvalidArgument("rank_size should be greater than 0.")));
        TF_CHECK_OK(rankSizeStatus);

        int32 inputRank = c->Rank(c->input(0));
        if (InferenceContext::kUnknownRank == inputRank) {
          ShapeHandle out = c->UnknownShapeOfRank(1);
          c->set_output(0, out);
          return Status::OK();
        }
        for (int32 i = 0; i < inputRank; i++) {
          DimensionHandle dimHandle = c->Dim(c->input(0), i);
          int64 value = c->Value(dimHandle);
          if (InferenceContext::kUnknownDim == value) {
            ShapeHandle out = c->UnknownShapeOfRank(1);
            c->set_output(0, out);
            return Status::OK();
          }
        }

        shape_inference::ShapeHandle unused;
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &unused));

        shape_inference::ShapeHandle inSubshape;
        TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 1, &inSubshape));

        auto inputFirstDimValue = c->Value(c->Dim(c->input(0), 0));
        shape_inference::ShapeHandle outputFirstDimAsShape;
        std::vector<shape_inference::DimensionHandle> outputFirstDim;
        outputFirstDim.push_back(c->MakeDim(rankSize * inputFirstDimValue));
        outputFirstDimAsShape = c->MakeShape(outputFirstDim);
        shape_inference::ShapeHandle output;
        TF_RETURN_IF_ERROR(c->Concatenate(outputFirstDimAsShape, inSubshape, &output));
        c->set_output(0, output);
        return Status::OK();
      })
      .Doc(R"doc(

)doc");

  REGISTER_OP("HcomBroadcast")
      .Input("input: T")
      .Output("output: T")
      .Attr("T: list(type) >= 0")
      .Attr("fusion: int")
      .Attr("fusion_id: int")
      .Attr("group: string")
      .Attr("root_rank: int")
      .SetIsStateful()
      .SetShapeFn([](shape_inference::InferenceContext *c) {
        for (int i = 0; i < c->num_inputs(); i++) { c->set_output(i, c->input(i)); }
        return Status::OK();
      })
      .Doc(R"doc(
Sends `input` to all devices that are connected to the output.

The graph should be constructed so that all ops connected to the output have a
valid device assignment, and the op itself is assigned one of these devices.

input: The input to the broadcast.
output: The same as input.
)doc");

  REGISTER_OP("HcomReduce")
      .Input("input: T")
      .Output("output: T")
      .Attr("T: {int8, int16, int32, float16, float32}")
      .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
      .Attr("group: string")
      .Attr("root_rank: int")
      .Attr("fusion: int")
      .Attr("fusion_id: int")
      .SetIsStateful()
      .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
      })
      .Doc(R"doc(
Outputs a tensor containing the reduction across all input tensors passed to ops.

The graph should be constructed so if one op runs with shared_name value `c`,
then `num_devices` ops will run with shared_name value `c`.  Failure to do so
will cause the graph execution to fail to complete.

input: the input to the reduction
output: the value of the reduction across all `num_devices` devices.
reduction: the reduction operation to perform.
group: all devices of the group participating in this reduction.
)doc");

  REGISTER_OP("HcomReduceScatter")
      .Input("input: T")
      .Output("output: T")
      .Attr("T: {int8, int16, int32, float16, float32}")
      .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
      .Attr("group: string")
      .Attr("rank_size: int")
      .SetIsStateful()
      .SetShapeFn([](shape_inference::InferenceContext *c) {
        int rankSize = 0;
        TF_CHECK_OK(c->GetAttr("rank_size", &rankSize));
        Status rankSizeStatus =
            ((rankSize > 0) ? (Status::OK()) : (errors::InvalidArgument("rank_size should be greater than 0.")));
        TF_CHECK_OK(rankSizeStatus);

        int32 inputRank = c->Rank(c->input(0));
        if (InferenceContext::kUnknownRank == inputRank) {
          ShapeHandle out = c->UnknownShapeOfRank(1);
          c->set_output(0, out);
          return Status::OK();
        }
        for (int32 i = 0; i < inputRank; i++) {
          DimensionHandle dimHandle = c->Dim(c->input(0), i);
          int64 value = c->Value(dimHandle);
          if (InferenceContext::kUnknownDim == value) {
            ShapeHandle out = c->UnknownShapeOfRank(1);
            c->set_output(0, out);
            return Status::OK();
          }
        }

        shape_inference::ShapeHandle unused;
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &unused));

        shape_inference::ShapeHandle inSubshape;
        TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 1, &inSubshape));

        auto inputFirstDimValue = c->Value(c->Dim(c->input(0), 0));
        shape_inference::ShapeHandle outputFirstDimAsShape;
        Status outputFirstDimStatus = ((inputFirstDimValue % rankSize) == 0)
                                          ? (Status::OK())
                                          : (errors::InvalidArgument("input first dim should be N * rank_size."));
        TF_CHECK_OK(outputFirstDimStatus);
        std::vector<shape_inference::DimensionHandle> outputFirstDim;
        outputFirstDim.push_back(c->MakeDim(inputFirstDimValue / rankSize));
        outputFirstDimAsShape = c->MakeShape(outputFirstDim);
        shape_inference::ShapeHandle output;
        TF_RETURN_IF_ERROR(c->Concatenate(outputFirstDimAsShape, inSubshape, &output));
        c->set_output(0, output);
        return Status::OK();
      })
      .Doc(R"doc(

)doc");

  REGISTER_OP("HcomSend")
      .Input("input: T")
      .Attr("T: {int8, int16, int32, float16, float32, int64, uint64}")
      .Attr("group: string")
      .Attr("sr_tag: int")
      .Attr("dest_rank: int")
      .SetIsStateful()
      .SetShapeFn(shape_inference::NoOutputs)
      .Doc(R"doc(

)doc");

  REGISTER_OP("HcomReceive")
      .Output("output: T")
      .Attr("T: {int8, int16, int32, float16, float32, int64, uint64}")
      .Attr("shape: shape")
      .Attr("group: string")
      .Attr("sr_tag: int")
      .Attr("src_rank: int")
      .SetIsStateful()
      .SetShapeFn(shape_inference::ExplicitShape)
      .Doc(R"doc(

)doc");

  REGISTER_OP("HcomRemoteRead")
      .Input("remote: T")
      .Output("local: dtype")
      .Attr("T: {int64, uint64}")
      .Attr("dtype: {int8, int16, int32, float16, float32, int64, uint64}")
      .SetIsStateful()
      .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->UnknownShape());// һάshapeȷڶάunknown
        return Status::OK();
      })
      .Doc(R"doc(

)doc");

  REGISTER_OP("HcomRemoteRefRead")
      .Input("remote: T")
      .Input("cache_var: Ref(dtype)")
      .Input("local_offset: T")
      .Output("cache_var1:Ref(dtype)")
      .Attr("T: {uint64}")
      .Attr("dtype: {int8, int16, int32, float16, float32, int64, uint64}")
      .SetIsStateful()
      .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(1));
        return Status::OK();
      })
      .Doc(R"doc(

)doc");

  REGISTER_OP("HcomRemoteWrite")
      .Input("remote: T")
      .Input("local: dtype")
      .Attr("T: {int64, uint64}")
      .Attr("dtype: {int8, int16, int32, float16, float32, int64, uint64}")
      .SetIsStateful()
      .SetShapeFn(shape_inference::NoOutputs)
      .Doc(R"doc(

)doc");

  REGISTER_OP("HcomRemoteScatterWrite")
      .Input("remote: T")
      .Input("local: Ref(dtype)")
      .Input("local_offset: T")
      .Attr("T: {int64, uint64}")
      .Attr("dtype: {int8, int16, int32, float16, float32, int64, uint64}")
      .SetIsStateful()
      .SetShapeFn(shape_inference::NoOutputs)
      .Doc(R"doc(

)doc");
}// namespace tensorflow

namespace tensorflow {
  REGISTER_KERNEL_BUILDER(Name("HcomAllReduce").Device(DEVICE_CPU).Priority(3), FakeOp);
  REGISTER_KERNEL_BUILDER(Name("HcomAllGather").Device(DEVICE_CPU).Priority(3), FakeOp);
  REGISTER_KERNEL_BUILDER(Name("HcomBroadcast").Device(DEVICE_CPU).Priority(3), FakeOp);
  REGISTER_KERNEL_BUILDER(Name("HcomReduce").Device(DEVICE_CPU).Priority(3), FakeOp);
  REGISTER_KERNEL_BUILDER(Name("HcomReduceScatter").Device(DEVICE_CPU).Priority(3), FakeOp);
  REGISTER_KERNEL_BUILDER(Name("HcomSend").Device(DEVICE_CPU).Priority(3), FakeOp);
  REGISTER_KERNEL_BUILDER(Name("HcomReceive").Device(DEVICE_CPU).Priority(3), FakeOp);
  REGISTER_KERNEL_BUILDER(Name("HcomRemoteRead").Device(DEVICE_CPU).Priority(3), FakeOp);
  REGISTER_KERNEL_BUILDER(Name("HcomRemoteRefRead").Device(DEVICE_CPU).Priority(3), FakeOp);
  REGISTER_KERNEL_BUILDER(Name("HcomRemoteWrite").Device(DEVICE_CPU).Priority(3), FakeOp);
  REGISTER_KERNEL_BUILDER(Name("HcomRemoteScatterWrite").Device(DEVICE_CPU).Priority(3), FakeOp);
}// namespace tensorflow