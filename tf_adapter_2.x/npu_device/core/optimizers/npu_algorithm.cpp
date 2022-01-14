/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include "tensorflow/core/graph/algorithm.h"

#include "npu_device.h"
#include "npu_optimizer_manager.h"

namespace npu {
tensorflow::Status MarkGraphNodeInOutDesc(TFE_Context *context, tensorflow::Graph *graph, int num_inputs,
                                          TFE_TensorHandle **inputs) {
  tensorflow::ShapeRefiner shape_refiner(graph->versions(), npu::UnwrapCtx(context)->FuncLibDef());
  VecTensorShapes arg_shapes;
  VecTensorDataTypes arg_handle_dtyes;
  VecTensorPartialShapes arg_handle_shapes;
  for (int i = 0; i < num_inputs; i++) {
    const tensorflow::Tensor *tensor;
    NPU_REQUIRES_OK(npu::GetTensorHandleTensor(inputs[i], &tensor));
    arg_shapes.push_back({tensor->shape()});
    TensorDataTypes handle_dtyes;
    TensorPartialShapes handle_shapes;
    if (tensor->dtype() == tensorflow::DT_RESOURCE) {
      auto handle = tensor->flat<tensorflow::ResourceHandle>()(0);
      const auto &dtypes_and_shapes = handle.dtypes_and_shapes();
      for (auto &dtype_and_shape : dtypes_and_shapes) {
        handle_dtyes.push_back(dtype_and_shape.dtype);
        handle_shapes.push_back(dtype_and_shape.shape);
      }
    }
    arg_handle_dtyes.push_back(handle_dtyes);
    arg_handle_shapes.push_back(handle_shapes);
  }

  auto node_shape_inference_lambda = [&shape_refiner, num_inputs, inputs, &arg_shapes, &arg_handle_dtyes,
                                      &arg_handle_shapes](tensorflow::Node *node) {
    AssembleOpDef(node);
    if (node->IsArg() && node->attrs().Find("index")) {
      auto index = node->attrs().Find("index")->i();
      if (index < num_inputs && !node->attrs().Find("_output_shapes")) {
        node->AddAttr("_output_shapes", arg_shapes[index]);
      }
      if (index < num_inputs && tensorflow::unwrap(inputs[index])->DataType() == tensorflow::DT_RESOURCE) {
        if (!node->attrs().Find("_handle_shapes")) {
          node->AddAttr("_handle_shapes", arg_handle_shapes[index]);
        }
        if (!node->attrs().Find("_handle_dtypes")) {
          node->AddAttr("_handle_dtypes", arg_handle_dtyes[index]);
        }
      }
    }
    auto status = shape_refiner.AddNode(node);
    if (!status.ok()) {
      LOG(INFO) << "  " << node->name() << "[" << node->type_string() << "] Skip infer " << status.error_message();
      return;
    }
    auto node_ctx = shape_refiner.GetContext(node);

    DLOG() << "Shape of node " << node->DebugString();
    if (kDumpExecutionDetail) {
      TensorDataTypes input_types;
      tensorflow::InputTypesForNode(node->def(), node->op_def(), &input_types);
      TensorPartialShapes input_shapes;
      for (int i = 0; i < node_ctx->num_inputs(); ++i) {
        tensorflow::TensorShapeProto proto;
        node_ctx->ShapeHandleToProto(node_ctx->input(i), &proto);
        input_shapes.emplace_back(proto);
        LOG(INFO) << "    input " << i << ": " << tensorflow::DataTypeString(input_types[i])
                  << node_ctx->DebugString(node_ctx->input(i));
      }
    }

    TensorDataTypes input_types;
    TensorDataTypes output_types;
    tensorflow::InOutTypesForNode(node->def(), node->op_def(), &input_types, &output_types);

    if (!input_types.empty()) {
      tensorflow::AttrValue input_desc_attrs;
      bool input_desc_incomplete = false;
      for (int i = 0; i < node->num_inputs(); i++) {
        const tensorflow::Edge *edge = nullptr;
        status = node->input_edge(i, &edge);
        if (!status.ok()) {
          LOG(ERROR) << status.ToString();
          return;
        }

        auto input_attr = edge->src()->attrs().Find(kOutputDesc);
        if (input_attr == nullptr) {
          input_desc_incomplete = true;
          LOG(WARNING) << node->DebugString() << " input node " << edge->src()->DebugString()
                       << " has no desc for output " << edge->src_output();
          break;
        }
        *input_desc_attrs.mutable_list()->add_func() =
          edge->src()->attrs().Find(kOutputDesc)->list().func(edge->src_output());
      }
      if (!input_desc_incomplete) {
        node->AddAttr(kInputDesc, input_desc_attrs);
      } else {
        TensorPartialShapes input_shapes;
        for (int i = 0; i < node_ctx->num_inputs(); ++i) {
          tensorflow::TensorShapeProto proto;
          node_ctx->ShapeHandleToProto(node_ctx->input(i), &proto);
          input_shapes.emplace_back(proto);
        }
        AssembleInputDesc(input_shapes, input_types, node);
      }
    }

    if (!output_types.empty()) {
      TensorPartialShapes output_shapes;
      for (int i = 0; i < node_ctx->num_outputs(); ++i) {
        tensorflow::TensorShapeProto proto;
        node_ctx->ShapeHandleToProto(node_ctx->output(i), &proto);
        output_shapes.emplace_back(proto);
        DLOG() << "    output " << i << ": " << tensorflow::DataTypeString(output_types[i])
               << node_ctx->DebugString(node_ctx->output(i));
      }
      AssembleOutputDesc(output_shapes, output_types, node);
    }
  };
  tensorflow::ReverseDFS(*graph, {}, node_shape_inference_lambda);
  return tensorflow::Status::OK();
}
}  // namespace npu