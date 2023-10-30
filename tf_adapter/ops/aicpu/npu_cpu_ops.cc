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
// regist embedding local index op
REGISTER_OP("EmbeddingLocalIndex")
  .Input("addr_table: uint64")
  .Input("index: T")
  .Output("local_idx: T")
  .Output("nums: T")
  .Output("recover_idx: T")
  .Attr("T: {int64,int32,uint64,uint32}")
  .Attr("row_memory: int = 320")
  .Attr("mode: string = 'mod' ")
  .SetAllowsUninitializedInput()
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    auto index_shape = c->input(1);
    c->set_output(0, index_shape);
    auto nums_shape = c->MakeShape({c->Dim(c->input(0), 0)});
    c->set_output(1, nums_shape);
    c->set_output(2, index_shape);
    return Status::OK();
  })
  .Doc(R"doc(
    Traverse the index calculation server and its position in the server.
    Arguments
        addr_table:    Tensors of addr_table.
        index:    Tensors of index.
    Output
        local_idx:    Local_idx sorted by rank_id.
        nums:   The number of local_idx found on each rank_id.
        recover_idx:  The sorted local_idx element corresponds to the position of
                      the original input index.
    )doc");
// regist lru cahe op
REGISTER_OP("LruCache")
  .Output("cache: resource")
  .Attr("cache_size: int")
  .Attr("load_factor: float = 1.0")
  .Attr("container: string = ''")
  .Attr("shared_name: string = 'LruCache'")
  .Attr("dtype: {uint32, uint64, int32, int64}")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape);
// regist cache add op
REGISTER_OP("CacheAdd")
  .Input("cache: resource")
  .Input("ids: T")
  .Output("swap_in_id: T")
  .Output("swap_in_idx: T")
  .Output("swap_out_id: T")
  .Output("swap_out_idx: T")
  .Attr("T: {int64, int32, uint64, uint32}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->Vector(c->UnknownDim()));
    c->set_output(1, c->Vector(c->UnknownDim()));
    c->set_output(2, c->Vector(c->UnknownDim()));
    c->set_output(3, c->Vector(c->UnknownDim()));
    return Status::OK();
  });
// regist cache remote index to local op
REGISTER_OP("CacheRemoteIndexToLocal")
  .Input("cache: resource")
  .Input("ids: T")
  .Output("local_idx: T")
  .Attr("T: {int64, int32, uint32, uint64}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->Vector(c->Rank(c->input(1))));
    return Status::OK();
  });
// regist cache all index to local op
REGISTER_OP("CacheAllIndexToLocal")
  .Input("cache: resource")
  .Output("local_idx: dtype")
  .Attr("dtype: {int64, int32, uint32, uint64}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->Vector(c->UnknownDim()));
    return Status::OK();
  });

// regist deformable offsets op
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
// regist deformable offsets grad op
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
// regist Random Choice With Mask op
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
    if (count > 0) {
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

REGISTER_OP("InitPartitionMap")
  .Input("ps_num: int32")
  .Input("ps_ids: int32")
  .Attr("partition_num: int = 65537")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("InitEmbeddingHashmap")
  .Input("table_id: int32")
  .Attr("bucket_size: int = 0")
  .Attr("value_total_len: int = 0")
  .Attr("dtype: {uint8, uint16, float32} = DT_FLOAT")
  .Attr("embedding_dim: int = 0")
  .Attr("initializer_mode: string = '' ")
  .Attr("constant_value: float = 0")
  .Attr("min: float = -2")
  .Attr("max: float = 2")
  .Attr("mu: float = 0")
  .Attr("sigma: float = 1")
  .Attr("seed: int = 0")
  .Attr("seed2: int = 0")
  .Attr("filter_mode: string = 'no_filter' ")
  .Attr("optimizer_mode: string = '' ")
  .Attr("optimizer_params: list(float) = [0.1]")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("EmbeddingTableImport")
  .Input("file_path: string")
  .Input("ps_id: int32")
  .Input("table_id: int32")
  .Attr("embedding_dim: list(int)")
  .Attr("value_total_len: list(int)")
  .Attr("only_var_flag: bool = false")
  .Attr("file_type: string = 'bin' ")
  .Attr("table_name: list(string)")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("EmbeddingTableFind")
  .Input("table_id: int32")
  .Input("keys: int64")
  .Output("values: float32")
  .Attr("embedding_dim: int = 0")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    ShapeHandle keys_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys_shape));
    int embedding_dim;
    if (!c->GetAttr("embedding_dim", &embedding_dim).ok()) {
        return errors::InvalidArgument("Invalid embedding_dim");
    }
    c->set_output(0, c->Matrix(c->Dim(keys_shape, 0), embedding_dim));
    return Status::OK();
  });

REGISTER_OP("UninitPartitionMap")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("UninitEmbeddingHashmap")
  .Input("table_id: int32")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("TableToResource")
  .Input("table_id: int32")
  .Output("table_handle: resource")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    auto data_shape = c->input(0);
    c->set_output(0, data_shape);
    return Status::OK();
  });

REGISTER_OP("EmbeddingTableFindAndInit")
  .Input("table_id: int32")
  .Input("keys: int64")
  .Output("values: float32")
  .Attr("embedding_dim: int = 0")
  .Attr("value_total_len: int = 0")
  .Attr("initializer_mode: string = 'random_uniform'")
  .Attr("constant_value: float = 0")
  .Attr("min: float = -2")
  .Attr("max: float = 2")
  .Attr("mu: float = 0")
  .Attr("sigma: float = 1")
  .Attr("seed: int = 0")
  .Attr("seed2: int = 0")
  .Attr("filter_mode: string = 'no_filter'")
  .Attr("filter_freq: int = 0")
  .Attr("default_key_or_value: bool = false")
  .Attr("default_key: int = 0")
  .Attr("default_value: float = 0")
  .Attr("optimizer_mode: string = '' ")
  .Attr("optimizer_params: list(float) = [0.1]")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    ShapeHandle keys_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys_shape));
    int embedding_dim;
    if (!c->GetAttr("embedding_dim", &embedding_dim).ok()) {
        return errors::InvalidArgument("Invalid embedding_dim");
    }
    c->set_output(0, c->Matrix(c->Dim(keys_shape, 0), embedding_dim));
    return Status::OK();
  });

REGISTER_OP("FusedRemoteLookupWithUnique")
  .Input("table_id: int32")
  .Input("keys: int64")
  .Input("key_num_input: int64")
  .Input("unique_indices: int32")
  .Output("values: float32")
  .Attr("embedding_dim: int = 0")
  .Attr("value_total_len: int = 0")
  .Attr("initializer_mode: string = 'random_uniform'")
  .Attr("constant_value: float = 0")
  .Attr("min: float = -2")
  .Attr("max: float = 2")
  .Attr("mu: float = 0")
  .Attr("sigma: float = 1")
  .Attr("seed: int = 0")
  .Attr("seed2: int = 0")
  .Attr("filter_mode: string = 'no_filter'")
  .Attr("filter_freq: int = 0")
  .Attr("default_key_or_value: bool = false")
  .Attr("default_key: int = 0")
  .Attr("default_value: float = 0")
  .Attr("optimizer_mode: string = '' ")
  .Attr("optimizer_params: list(float) = [0.1]")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    ShapeHandle keys_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &keys_shape));
    int embedding_dim;
    if (!c->GetAttr("embedding_dim", &embedding_dim).ok()) {
        return errors::InvalidArgument("Invalid embedding_dim");
    }
    c->set_output(0, c->Matrix(c->Dim(keys_shape, 0), embedding_dim));
    return Status::OK();
  });

REGISTER_OP("EmbeddingApplyAdam")
  .Input("var_handle: resource")
  .Input("beta1_power: T")
  .Input("beta2_power: T")
  .Input("lr: T")
  .Input("beta1: T")
  .Input("beta2: T")
  .Input("epsilon: T")
  .Input("grad: T")
  .Input("keys: int64")
  .Input("global_step: Tstep")
  .Output("var_handle_output: resource")
  .Attr("embedding_dim: int = 0")
  .Attr("T: {float32, float16}")
  .Attr("Tstep: {int32, int64}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    auto data_shape = c->input(0);
    c->set_output(0, data_shape);
    return Status::OK();
  });

REGISTER_OP("EmbeddingApplyAdamW")
  .Input("var_handle: resource")
  .Input("beta1_power: T")
  .Input("beta2_power: T")
  .Input("lr: T")
  .Input("weight_decay: T")
  .Input("beta1: T")
  .Input("beta2: T")
  .Input("epsilon: T")
  .Input("grad: T")
  .Input("keys: int64")
  .Input("max_grad_norm: T")
  .Output("var_handle_output: resource")
  .Attr("amsgrad: bool")
  .Attr("maximize: bool")
  .Attr("embedding_dim: int = 0")
  .Attr("T: {float32, float16}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    auto data_shape = c->input(0);
    c->set_output(0, data_shape);
    return Status::OK();
  });

REGISTER_OP("EmbeddingApplyAdaGrad")
  .Input("var_handle: resource")
  .Input("lr: T")
  .Input("grad: T")
  .Input("keys: int64")
  .Input("global_step: Tstep")
  .Output("var_handle_output: resource")
  .Attr("embedding_dim: int = 0")
  .Attr("T: {float32, float16}")
  .Attr("Tstep: {int32, int64}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    auto data_shape = c->input(0);
    c->set_output(0, data_shape);
    return Status::OK();
  });

REGISTER_OP("EmbeddingComputeVarExport")
  .Input("file_path: string")
  .Input("ps_id: int32")
  .Input("table_id: int32")
  .Attr("table_name: list(string)")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("EmbeddingComputeVarImport")
  .Input("file_path: string")
  .Input("ps_id: int32")
  .Attr("table_name: list(string)")
  .Input("table_id: int32")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("EmbeddingTableExport")
  .Input("file_path: string")
  .Input("ps_id: int32")
  .Input("table_id: int32")
  .Attr("embedding_dim: list(int)")
  .Attr("value_total_len: list(int)")
  .Attr("export_mode: {'all', 'old', 'new', 'specifiednew'} = 'all'")
  .Attr("only_var_flag: bool = false")
  .Attr("file_type: string = 'bin' ")
  .Attr("table_name: list(string)")
  .Attr("filter_export_flag: bool = false")
  .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("EmbeddingFeatureMapping")
  .Input("feature_id: int64")
  .Output("offset_id: int32")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

// regist dense image warp op
REGISTER_OP("DenseImageWarp")
  .Input("image: T")
  .Input("flow: S")
  .Output("y: T")
  .Attr("T: {float16, float32}")
  .Attr("S: {float16, float32}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    auto input_image_shape = c->input(0);
    c->set_output(0, input_image_shape);
    return Status::OK();
  });
// regist dense image warp grad op
REGISTER_OP("DenseImageWarpGrad")
  .Input("grad: T")
  .Input("image: T")
  .Input("flow: S")
  .Output("grad_image: T")
  .Output("grad_flow: S")
  .Attr("T: {float16, float32}")
  .Attr("S: {float16, float32}")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
    auto input_image_shape = c->input(1);
    auto input_flow_shape = c->input(2);
    c->set_output(0, input_image_shape);
    c->set_output(1, input_flow_shape);
    return Status::OK();
  });

  REGISTER_OP("ScatterElements")
    .Input("data: T")
    .Input("indices: indexT")
    .Input("updates: T")
    .Output("y: T")
    .Attr("axis: int = 0")
    .Attr("T: numbertype")
    .Attr("indexT: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      auto data_shape = c->input(0);
      c->set_output(0, data_shape);
      return Status::OK();
    });
  REGISTER_OP("ScatterElementsV2")
    .Input("data: T")
    .Input("indices: indexT")
    .Input("updates: T")
    .Output("y: T")
    .Attr("axis: int = 0")
    .Attr("T: numbertype")
    .Attr("indexT: {int32, int64}")
    .Attr("reduction: string = 'None'");

  REGISTER_OP("BatchEnqueue")
    .Input("x: T")
    .Input("queue_id: uint32")
    .Output("enqueue_count: int32")
    .Attr("batch_size: int = 8")
    .Attr("queue_name: string = ''")
    .Attr("queue_depth: int = 100")
    .Attr("pad_mode: {'REPLICATE', 'ZERO'} = 'REPLICATE'")
    .Attr("T: {float16, float32, float64, int8, uint8, int16, uint16, int32, uint32, int64, uint64}")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

  REGISTER_OP("OCRRecognitionPreHandle")
    .Input("imgs_data: uint8")
    .Input("imgs_offset: int32")
    .Input("imgs_size: int32")
    .Input("langs: int32")
    .Input("langs_score: T")
    .Output("imgs: uint8")
    .Output("imgs_relation: int32")
    .Output("imgs_lang: int32")
    .Output("imgs_piece_fillers: int32")
    .Attr("batch_size: int = 8")
    .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
    .Attr("pad_mode: {'REPLICATE', 'ZERO'} = 'REPLICATE'")
    .Attr("T: {float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(c->UnknownDim()));
      c->set_output(3, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

  REGISTER_OP("OCRDetectionPreHandle")
    .Input("img: uint8")
    .Output("resized_img: uint8")
    .Output("h_scale: float32")
    .Output("w_scale: float32")
    .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      std::string dt_format;
      const std::set<std::string> kVaildFormat = {"NHWC", "NCHW"};
      if (!c->GetAttr("data_format", &dt_format).ok()) {
        dt_format = "NHWC";
      }
      if (kVaildFormat.find(dt_format) == kVaildFormat.end()) {
        return errors::InvalidArgument("Invalid data format string: ", dt_format);
      }
      const int32_t kRank = 3;
      int32 imgRank = c->Rank(c->input(0));
      if (imgRank != kRank) {
        return errors::InvalidArgument("Invalid image shape: shape rank must be 3, but got ", imgRank);
      }

      size_t cPos = dt_format.find("C") - 1;
      size_t hPos = dt_format.find("H") - 1;
      size_t wPos = dt_format.find("W") - 1;
      int64 channel = c->Value(c->Dim(c->input(0), cPos));
      const int64_t kChannel = 3;
      if (channel != kChannel) {
        return errors::InvalidArgument("Invalid image shape: shape channel must be 3, but got ", channel);
      }

      const int64_t kMinSize = 480;
      const int64_t kMidSize = 960;
      const int64_t kMaxSize = 1920;
      const int64_t kLongSizeLow = 720;
      const int64_t kLongSizeHigh = 1440;
      int64_t resize;
      if (c->ValueKnown(c->Dim(c->input(0), hPos)) && c->ValueKnown(c->Dim(c->input(0), wPos))) {
        int64_t longSize = std::max(c->Value(c->Dim(c->input(0), hPos)), c->Value(c->Dim(c->input(0), wPos)));
        resize = (longSize <= kLongSizeLow) ? kMinSize : ((longSize <= kLongSizeHigh) ? kMidSize : kMaxSize);
      } else {
        resize = c->Value(c->UnknownDim());
      }
      std::vector<DimensionHandle> out_dims(kRank);
      out_dims[cPos] = c->MakeDim(channel);
      out_dims[hPos] = c->MakeDim(resize);
      out_dims[wPos] = c->MakeDim(resize);
      c->set_output(0, c->MakeShape(out_dims));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

  REGISTER_OP("OCRIdentifyPreHandle")
    .Input("imgs_data: uint8")
    .Input("imgs_offset: int32")
    .Input("imgs_size: int32")
    .Output("resized_imgs: uint8")
    .Attr("size: list(int)")
    .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      std::vector<int32_t> size;
      TF_RETURN_IF_ERROR(c->GetAttr("size", &size));
      if (size.size() != 2) {
        return errors::InvalidArgument(
          "size attribute should contain 2 values, but got: ",
          size.size());
      }
      const int64_t k1 = size[0];
      const int64_t k2 = size[1];
      std::string dt_format;
      const std::set<std::string> kVaildFormat = {"NHWC", "NCHW"};
      if (!c->GetAttr("data_format", &dt_format).ok()) {
        dt_format = "NHWC";
      }
      if (kVaildFormat.find(dt_format) == kVaildFormat.end()) {
        return errors::InvalidArgument("Invalid data format string: ",
                                       dt_format);
      }
      const int32 kImgShapeRank = 1;
      if (c->Rank(c->input(0)) != kImgShapeRank) {
        return errors::InvalidArgument("Invalid images shape: must be 1, bug got: ",
                                       c->Rank(c->input(0)));
      }
      const int32 kImgOffsetShapeRank = 1;
      if (c->Rank(c->input(1)) != kImgOffsetShapeRank) {
        return errors::InvalidArgument("Invalid images offset shape: must be 1, bug got: ",
                                       c->Rank(c->input(1)));
      }
      const int32 kImgSizeShapeRank = 2;
      if (c->Rank(c->input(2)) != kImgSizeShapeRank) {
        return errors::InvalidArgument("Invalid images size shape: must be 2, bug got: ",
                                       c->Rank(c->input(2)));
      }
      // the second dim of imgs size must be 3
      const int32 kImgSizeShape = 3;
      if (c->Value(c->Dim(c->input(2), 1)) != kImgSizeShape) {
        return errors::InvalidArgument("Invalid image size shape: must be 3, bug got: ",
                                       c->Value(c->Dim(c->input(2), 1)));
      }
      const int32_t kRank = 4;
      std::vector<DimensionHandle> out_dims(kRank);
      auto imgs_offset = c->input(1);
      out_dims[0] = c->Dim(imgs_offset, 0);
      if (dt_format == "NHWC") {
        out_dims[1] = c->MakeDim(k1);
        out_dims[2] = c->MakeDim(k2);
        out_dims[3] = c->MakeDim(3);
      } else {
        out_dims[1] = c->MakeDim(3);
        out_dims[2] = c->MakeDim(k1);
        out_dims[3] = c->MakeDim(k2);
      }
      c->set_output(0, c->MakeShape(out_dims));
      return Status::OK();
    });

REGISTER_OP("BatchDilatePolys")
         .Input("polys_data:int32")
         .Input("polys_offset:int32")
         .Input("polys_size:int32")
         .Input("score:float")
         .Input("min_border:int32")
         .Input("min_area_thr:int32")
         .Input("score_thr:float")
         .Input("expand_scale:float")
         .Output("dilated_polys_data:int32")
         .Output("dilated_polys_offset:int32")
         .Output("dilated_polys_size:int32")
         .SetShapeFn([](shape_inference::InferenceContext *c) {
           c->set_output(0, c->Vector(c->UnknownDim()));
           c->set_output(1, c->Vector(c->UnknownDim()));
           c->set_output(2, c->Vector(c->UnknownDim()));
           return Status::OK();
         });

REGISTER_OP("OCRFindContours")
         .Input("img:uint8")
         .Output("polys_data:int32")
         .Output("polys_offset:int32")
         .Output("polys_size:int32")
         .Attr("value_mode:int = 0")
         .SetShapeFn([](shape_inference::InferenceContext *c) {
           c->set_output(0, c->Vector(c->UnknownDim()));
           c->set_output(1, c->Vector(c->UnknownDim()));
           c->set_output(2, c->Vector(c->UnknownDim()));
           return Status::OK();
         });

REGISTER_OP("Dequeue")
    .Input("queue_id: uint32")
    .Output("data: output_type")
    .Attr("output_type: {float16, float32, float64, int8, uint8,"
          "int16, uint16, int32, uint32, int64, uint64} = DT_UINT8")
    .Attr("output_shape: list(int)")
    .Attr("queue_name: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      std::vector<int32_t> output_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("output_shape", &output_shape));
      int32_t rank = output_shape.size();
      std::vector<DimensionHandle> out_dims(rank);
      for (auto i = 0; i < rank; ++i) {
        out_dims[i] = c->MakeDim(output_shape[i]);
      }
      c->set_output(0, c->MakeShape(out_dims));
      return Status::OK();
    });

REGISTER_OP("OCRDetectionPostHandle")
    .Input("img: uint8")
    .Input("polys_data: int32")
    .Input("polys_offset: int32")
    .Input("polys_size: int32")
    .Output("imgs_data: uint8")
    .Output("imgs_offset: int32")
    .Output("imgs_size: int32")
    .Output("rect_points: int32")
    .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      auto data_shape = c->input(2);
      c->set_output(1, data_shape);
      c->set_output(2, c->Matrix(c->Rank(c->input(2)), 3));
      const int32_t rank = 3;
      std::vector<DimensionHandle> out_dims(rank);
      out_dims[0] = c->Dim(data_shape, 0);
      out_dims[1] = c->MakeDim(4);
      out_dims[2] = c->MakeDim(2);
      c->set_output(3, c->MakeShape(out_dims));
      return Status::OK();
    });

REGISTER_OP("WarpAffineV2")
    .Input("x: T")
    .Input("matrix: float32")
    .Input("dst_size: dstT")
    .Output("y: T")
    .Attr("interpolation: string")
    .Attr("border_type: string")
    .Attr("border_value: int = 0")
    .Attr("T: {uint8, float32}")
    .Attr("dstT: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      auto src_shape = c->input(0);
      auto dst_shape = c->input(2);
      const int32_t rank = 3;
      std::vector<DimensionHandle> out_dims(rank);
      out_dims[0] = c->Dim(dst_shape, 0);
      out_dims[1] = c->Dim(dst_shape, 1);
      out_dims[2] = c->Dim(src_shape, 2);
      c->set_output(0, c->MakeShape(out_dims));
      return Status::OK();
    });

REGISTER_OP("ResizeV2")
    .Input("x: T")
    .Input("dst_size: dstT")
    .Output("y: T")
    .Attr("interpolation: string")
    .Attr("T: {uint8, float32}")
    .Attr("dstT: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      auto src_shape = c->input(0);
      auto dst_shape = c->input(1);
      const int32_t rank = 3;
      std::vector<DimensionHandle> out_dims(rank);
      out_dims[0] = c->Dim(dst_shape, 0);
      out_dims[1] = c->Dim(dst_shape, 1);
      out_dims[2] = c->Dim(src_shape, 2);
      c->set_output(0, c->MakeShape(out_dims));
      return Status::OK();
    });

    REGISTER_OP("ResizeAndClipPolys")
    .Input("polys_data: int32")
    .Input("polys_offset: int32")
    .Input("polys_size: int32")
    .Input("h_scale: float32")
    .Input("w_scale: float32")
    .Input("img_h: int32")
    .Input("img_w: int32")
    .Output("clipped_polys_data: int32")
    .Output("clipped_polys_offset: int32")
    .Output("clipped_polys_size: int32")
    .Output("clipped_polys_num: int32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(c->UnknownDim()));
      c->set_output(3, c->Scalar());
      return Status::OK();
    });

    REGISTER_OP("NonZeroWithValueShape")
    .Input("value: T")
    .Input("index: int32")
    .Input("count: int32")
    .Output("out_value: T")
    .Output("out_index: int32")
    .Attr("T: {double, float, float16, int8, uint8, int16, uint16, int32, uint32, int64, uint64, bool}")

    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

    Status DecodeImageV3ShapeFn(InferenceContext* c) {
      ShapeHandle unused;
      int32 channels;
      bool expand_animations;
      DimensionHandle channels_dim;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->GetAttr("channels", &channels));
      TF_RETURN_IF_ERROR(c->GetAttr("expand_animations", &expand_animations));

      if (channels == 0) {
        channels_dim = c->UnknownDim();
      } else {
        if (channels < 0) {
          return errors::InvalidArgument("channels must be non-negative, got ",
                                         channels);
        }
        channels_dim = c->MakeDim(channels);
      }

      // `expand_animations` set to true will return 4-D shapes for GIF. 3-D shapes
      // will be returned for jpg, png, and bmp. `expand_animations` set to false
      // will always return 3-D shapes for all (jpg, png, bmp, gif).
      if (expand_animations) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      } else {
        c->set_output(0,
                      c->MakeShape({InferenceContext::kUnknownDim,
                                    InferenceContext::kUnknownDim, channels_dim}));
        return Status::OK();
      }
    }
    REGISTER_OP("DecodeImageV3")
    .Input("contents: string")
    // Setting `channels` to 0 means using the inherent number of channels in
    // the image.
    .Attr("channels: int = 0")
    .Attr("dtype: {uint8, uint16, float32} = DT_UINT8")
    .Output("image: dtype")
    .Attr("expand_animations: bool = true")
    .SetShapeFn(DecodeImageV3ShapeFn);
}  // namespace tensorflow
