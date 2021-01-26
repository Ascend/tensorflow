/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved. foss@huawei.com

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

namespace tensorflow {
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

namespace {
REGISTER_OP("FastGelu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("FastGeluGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(tensorflow::shape_inference::MergeBothInputsShapeFn);

REGISTER_OP("DynamicGruV2")
    .Input("x: float16")
    .Input("weight_input: float16")
    .Input("weight_hidden: float16")
    .Input("bias_input: T")
    .Input("bias_hidden: T")
    .Input("seq_length: int32")
    .Input("init_h: T")
    .Output("y: T")
    .Output("output_h: T")
    .Output("update: T")
    .Output("reset: T")
    .Output("new: T")
    .Output("hidden_new: T")
    .Attr("T: {float16, float32}")
    .Attr("direction: string")
    .Attr("cell_depth: int = 1")
    .Attr("keep_prob: float = 1.0")
    .Attr("cell_clip: float = -1.0")
    .Attr("num_proj: int = 0")
    .Attr("time_major: bool = true")
    .Attr("activation: string")
    .Attr("gate_order: string")
    .Attr("reset_after: bool = true")
    .Attr("is_training: bool = true")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      auto input_shape = c->input(0);
      auto weight_input_shape = c->input(1);
      auto weight_hidden_shape = c->input(2);
      auto num_step = c->Dim(input_shape, 0);
      auto batch_size = c->Dim(input_shape, 1);
      auto input_size = c->Dim(input_shape, 2);
      auto hidden_size = c->Dim(weight_hidden_shape, 0);
      int num_proj = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("num_proj", &num_proj));
      ShapeHandle output_y_shape;
      if (num_proj == 0) {
        output_y_shape = c->MakeShape({num_step, batch_size, hidden_size});
      } else {
        std::vector<DimensionHandle> num_projs;
        num_projs.reserve(num_proj);
        auto num_proj_shape = c->MakeShape(num_projs);
        DimensionHandle num_proj_size = c->Dim(num_proj_shape, 0);
        DimensionHandle output_hidden_size;
        TF_RETURN_IF_ERROR(c->Min(num_proj_size, hidden_size, &output_hidden_size));
        output_y_shape = c->MakeShape({num_step, batch_size, output_hidden_size});
      }          
      auto output_h_shape = c->MakeShape({num_step, batch_size, hidden_size});         
      c->set_output(0, output_y_shape);
      c->set_output(1, output_h_shape);
      c->set_output(2, c->UnknownShape());
      c->set_output(3, c->UnknownShape());
      c->set_output(4, c->UnknownShape());
      c->set_output(5, c->UnknownShape());
      return Status::OK();
    });

REGISTER_OP("DynamicGruV2Grad")
    .Input("x: float16")
    .Input("weight_input: float16")
    .Input("weight_hidden: float16")
    .Input("y: T")
    .Input("init_h: T")
    .Input("h: T")
    .Input("dy: T")
    .Input("dh: T")
    .Input("update: T")
    .Input("reset: T")
    .Input("new: T")
    .Input("hidden_new: T")
    .Output("dw_input: T")
    .Output("dw_hidden: T")
    .Output("db_input: T")
    .Output("db_hidden: T")
    .Output("dx: T")
    .Output("dh_prev: T")
    .Attr("T: {float16, float32}")
    .Attr("direction: string")
    .Attr("cell_depth: int = 1")
    .Attr("keep_prob: float = 1.0")
    .Attr("cell_clip: float = -1.0")
    .Attr("num_proj: int = 0")
    .Attr("time_major: bool = true")
    .Attr("gate_order: string")
    .Attr("reset_after: bool = true")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      auto input_shape = c->input(0);
      auto weight_hidden_shape = c->input(2);
      auto num_step = c->Dim(input_shape, 0);
      auto batch_size = c->Dim(input_shape, 1);
      auto input_size = c->Dim(input_shape, 2);
      auto hidden_size = c->Dim(weight_hidden_shape, 0);
      auto hidden_size_1 = c->Dim(weight_hidden_shape, 1);
      auto output_dw_input_shape = 
          c->MakeShape({input_size, hidden_size_1});
      auto output_dw_hidden_shape = 
          c->MakeShape({hidden_size, hidden_size_1});
      auto output_db_input_shape = 
          c->MakeShape({hidden_size_1});
      auto output_db_hidden_shape = 
          c->MakeShape({hidden_size_1});
      auto output_dx_shape = 
          c->MakeShape({num_step, batch_size, input_size});
      auto output_dh_prev_shape = 
          c->MakeShape({batch_size, hidden_size});        
      c->set_output(0, output_dw_input_shape);
      c->set_output(1, output_dw_hidden_shape);
      c->set_output(2, output_db_input_shape);
      c->set_output(3, output_db_hidden_shape);
      c->set_output(4, output_dx_shape);
      c->set_output(5, output_dh_prev_shape);
      return Status::OK();
    });

REGISTER_OP("DynamicRnn")
    .Input("x: T")
    .Input("w: T")
    .Input("b: T")
    .Input("seq_length: int32")
    .Input("init_h: T")
    .Input("init_c: T")
    .Output("y: T")
    .Output("output_h: T")
    .Output("output_c: T")
    .Output("i: T")
    .Output("j: T")
    .Output("f: T")
    .Output("o: T")
    .Output("tanhc: T")
    .Attr("T: {float16, float32}")
    .Attr("cell_type: string")
    .Attr("direction: string")
    .Attr("cell_depth: int = 1")
    .Attr("use_peephole: bool = false")
    .Attr("keep_prob: float = 1.0")
    .Attr("cell_clip: float = -1.0")
    .Attr("num_proj: int = 0")
    .Attr("time_major: bool = true")
    .Attr("activation: string")
    .Attr("forget_bias: float = 0.0")
    .Attr("is_training: bool = true")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      auto input_shape = c->input(0);
      auto num_step = c->Dim(input_shape, 0);
      auto batch_size = c->Dim(input_shape, 1);
      auto input_size = c->Dim(input_shape, 2);
      auto w = c->input(1);
      auto hidden_size_total = c->Dim(w, 0);
      DimensionHandle hidden_size;
      TF_RETURN_IF_ERROR(c->Subtract(hidden_size_total, input_size, &hidden_size));
      int num_proj = 0;
      TF_RETURN_IF_ERROR(c->GetAttr("num_proj", &num_proj));
      ShapeHandle output_y_shape;
      if (num_proj == 0) {
        output_y_shape = c->MakeShape({num_step, batch_size, hidden_size});
      } else {
        std::vector<DimensionHandle> num_projs;
        num_projs.reserve(num_proj);
        auto num_proj_shape = c->MakeShape(num_projs);
        DimensionHandle num_proj_size = c->Dim(num_proj_shape, 0);
        DimensionHandle output_hidden_size;
        TF_RETURN_IF_ERROR(c->Min(num_proj_size, hidden_size, &output_hidden_size));
        output_y_shape = c->MakeShape({num_step, batch_size, output_hidden_size});
      }          
      auto output_h_shape = 
          c->MakeShape({num_step, batch_size, hidden_size});
      auto output_c_shape = 
          c->MakeShape({num_step, batch_size, hidden_size});

      c->set_output(0, output_y_shape);
      c->set_output(1, output_h_shape);
      c->set_output(2, output_c_shape);
      c->set_output(3, c->UnknownShape());
      c->set_output(4, c->UnknownShape());
      c->set_output(5, c->UnknownShape());
      c->set_output(6, c->UnknownShape());
      c->set_output(7, c->UnknownShape());
      return Status::OK();
    });

REGISTER_OP("DynamicRnnGrad")
    .Input("x: T")
    .Input("w: T")
    .Input("b: T")
    .Input("y: T")
    .Input("init_h: T")
    .Input("init_c: T")
    .Input("h: T")
    .Input("c: T")
    .Input("dy: T")
    .Input("dh: T")
    .Input("dc: T")
    .Input("i: T")
    .Input("j: T")
    .Input("f: T")
    .Input("o: T")
    .Input("tanhc: T")
    .Output("dw: T")
    .Output("db: T")
    .Output("dx: T")
    .Output("dh_prev: T")
    .Output("dc_prev: T")
    .Attr("T: {float16, float32}")
    .Attr("cell_type: string")
    .Attr("direction: string")
    .Attr("cell_depth: int = 1")
    .Attr("use_peephole: bool = false")
    .Attr("keep_prob: float = 1.0")
    .Attr("cell_clip: float = -1.0")
    .Attr("num_proj: int = 0")
    .Attr("time_major: bool = true")
    .Attr("forget_bias: float = 0.0")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      auto input_shape = c->input(0);
      auto num_step = c->Dim(input_shape, 0);
      auto batch_size = c->Dim(input_shape, 1);
      auto input_size = c->Dim(input_shape, 2);
      auto w = c->input(1);
      auto hidden_size_total = c->Dim(w, 0);
      auto hidden_size_4 = c->Dim(w, 1);
      DimensionHandle hidden_size;
      TF_RETURN_IF_ERROR(c->Subtract(hidden_size_total, input_size, &hidden_size));

      auto output_dx_shape = 
          c->MakeShape({num_step, batch_size, input_size});
      auto output_dw_shape = 
          c->MakeShape({hidden_size_total, hidden_size_4});
      auto output_db_shape = 
          c->MakeShape({hidden_size_4});
      auto output_dh_prev_shape = 
          c->MakeShape({batch_size, hidden_size});
      auto output_dc_prev_shape = 
          c->MakeShape({batch_size, hidden_size});       
      c->set_output(0, output_dw_shape);
      c->set_output(1, output_db_shape);
      c->set_output(2, output_dx_shape);
      c->set_output(3, output_dh_prev_shape);
      c->set_output(4, output_dc_prev_shape);
      return Status::OK();
    });

REGISTER_OP("Centralization")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float16, float32}")
    .Attr("axes: list(int)")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("PRelu")
    .Input("x: T")
    .Input("weight: T")
    .Output("y: T")
    .Attr("T: {float16, float32}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("PReluGrad")
    .Input("grads: T")
    .Input("features: T")
    .Input("weights: T")
    .Output("dx: T")
    .Output("da: T")
    .Attr("T: {float16, float32}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(2));
        return Status::OK();
    });
}  // namespace
} // namespace tensorflow