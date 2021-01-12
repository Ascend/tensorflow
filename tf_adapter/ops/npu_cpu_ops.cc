/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("EmbeddingRankId")
  .Input("addr_table: uint64")
  .Input("index: T")
  .Output("rank_id: uint64")
  .Attr("T: {int64,int32,uint64}")
  .Attr("row_memory: int = 320")
  .Attr("mode: string = 'mod' ")
  .SetAllowsUninitializedInput()
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    auto out_shape = c->MakeShape({c->Dim(c->input(1), 0), c->Dim(c->input(0), 1)});
    c->set_output(0, out_shape);
    return Status::OK();
  })
  .Doc(R"doc(
    Traverse the index calculation server and its position in the server.
    Arguments
        addr_table:    Tensors of addr_table.
        index:    Tensors of index.
    Output
        rank_id:    Tensors with the same shape as index.dim(0)*3.
    )doc");
//regist lru cahe op
REGISTER_OP("LruCache")
  .Output("cache: resource")
  .Attr("cache_size: int")
  .Attr("load_factor: float = 1.0")
  .Attr("container: string = ''")
  .Attr("shared_name: string = 'LruCache'")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape);
//regist cache add op
REGISTER_OP("CacheAdd")
  .Input("cache: resource")
  .Input("ids: T")
  .Output("swap_in_id: T")
  .Output("swap_in_idx: int64")
  .Output("swap_out_id: T")
  .Output("swap_out_idx: int64")
  .Attr("T: {int64, int32, uint64, uint32}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->Vector(c->UnknownDim()));
    c->set_output(1, c->Vector(c->UnknownDim()));
    c->set_output(2, c->Vector(c->UnknownDim()));
    c->set_output(3, c->Vector(c->UnknownDim()));
    return Status::OK();
  });
//regist cache remote index to local op
REGISTER_OP("CacheRemoteIndexToLocal")
  .Input("cache: resource")
  .Input("ids: T")
  .Output("local_idx: int64")
  .Attr("T: {int64, int32, uint32, uint64}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->Vector(c->Rank(c->input(1))));
    return Status::OK();
  });
//regist deformable offsets op
REGISTER_OP("DeformableOffsets")
  .Input("x: T")
  .Input("offsets: T")
  .Output("y: T")
  .Attr("T: {float16, float32}")
  .Attr("strides: list(int)")
  .Attr("pads: list(int)")
  .Attr("ksize: list(int)")
  .Attr("dilations: list(int) = [1,1,1,1]")
  .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
  .Attr("deformable_groups: int = 1")
  .Attr("modulated: bool = true")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    std::string dt_format;
    const std::set<std::string> kValidFormat = {"NHWC", "NCHW"};
    if (!c->GetAttr("data_format", &dt_format).ok()) {
        dt_format = "NHWC";
    }
    if (kValidFormat.find(dt_format) == kValidFormat.end()) {
        return errors::InvalidArgument("Invalid data format string: ",
                                        dt_format);
    }

    size_t pos_n = dt_format.find("N");
    size_t pos_c = dt_format.find("C");
    size_t pos_h = dt_format.find("H");
    size_t pos_w = dt_format.find("W");

    auto input_x_shape = c->input(0);
    auto input_offsets_shape = c->input(1);
    int64_t input_offsets_h = c->Value(c->Dim(input_offsets_shape, pos_h));
    int64_t input_offsets_w = c->Value(c->Dim(input_offsets_shape, pos_w));

    std::vector<int32_t> ksizes;
    TF_RETURN_IF_ERROR(c->GetAttr("ksize", &ksizes));
    if (ksizes.size() != 2) {
        return errors::InvalidArgument(
          "ksize attribute should contain 2 values, but got: ",
          ksizes.size());
    }
    const int64_t kh = ksizes[0];
    const int64_t kw = ksizes[1];

    const int32_t rank = 4;
    std::vector<DimensionHandle> out_dims(rank);
    out_dims[pos_n] = c->Dim(input_x_shape, pos_n);
    out_dims[pos_c] = c->Dim(input_x_shape, pos_c);
    out_dims[pos_h] = c->MakeDim(input_offsets_h * kh);
    out_dims[pos_w] = c->MakeDim(input_offsets_w * kw);
    c->set_output(0, c->MakeShape(out_dims));
    return Status::OK();
  });
//regist deformable offsets grad op
REGISTER_OP("DeformableOffsetsGrad")
  .Input("grad: T")
  .Input("x: T")
  .Input("offsets: T")
  .Output("grad_x: T")
  .Output("grad_offsets: T")
  .Attr("T: {float16, float32}")
  .Attr("strides: list(int)")
  .Attr("pads: list(int)")
  .Attr("ksize: list(int)")
  .Attr("dilations: list(int) = [1,1,1,1]")
  .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
  .Attr("deformable_groups: int = 1")
  .Attr("modulated: bool = true")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    auto input_x_shape = c->input(1);
    auto input_offsets_shape = c->input(2);
    c->set_output(0, input_x_shape);
    c->set_output(1, input_offsets_shape);
    return Status::OK();
  });

REGISTER_OP("RandomChoiceWithMask")
  .Input("x: bool")
  .Output("y: int32")
  .Output("mask: bool")
  .Attr("count: int = 0")
  .Attr("seed: int = 0")
  .Attr("seed2: int = 0")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    int64 count(0);
    c->GetAttr("count", &count);
    if (count >0) {
      c->set_output(0, c->Matrix(count, c->Rank(c->input(0))));
      c->set_output(1, c->Vector(count));
    } else if (count == 0) {
      c->set_output(0, c->Matrix(c->UnknownDim(), c->Rank(c->input(0))));
      c->set_output(1, c->Vector(c->UnknownDim()));
    } else {
      return errors::InvalidArgument(
              "input count must greater or equal to 0 but instead is ",
              count);
    }
    return Status::OK();
  });
}  // namespace tensorflow
