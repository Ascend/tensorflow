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

#include "npu_parser.h"

#include "npu_env.h"
#include "npu_logger.h"
#include "tensorflow/core/graph/algorithm.h"

namespace npu {
void AssembleParserAddons(const tensorflow::FunctionLibraryDefinition *lib_def, tensorflow::Graph *graph) {
  tensorflow::ShapeRefiner shape_refiner(graph->versions(), lib_def);
  auto node_shape_inference_lambda = [&shape_refiner](tensorflow::Node *node) {
    AssembleOpDef(node);
    auto status = shape_refiner.AddNode(node);
    if (!status.ok()) {
      LOG(INFO) << "  " << node->name() << "[" << node->type_string() << "] Skip infer " << status.error_message();
      return;
    }
    auto node_ctx = shape_refiner.GetContext(node);

    TensorDataTypes input_types;
    TensorDataTypes output_types;
    (void)tensorflow::InOutTypesForNode(node->def(), node->op_def(), &input_types, &output_types);

    DLOG() << "Shape of node " << node->name() << "[" << node->type_string() << "]";
    if (kDumpExecutionDetail) {
      for (int i = 0; i < node_ctx->num_inputs(); ++i) {
        DLOG() << "    input " << i << ": " << tensorflow::DataTypeString(input_types[static_cast<size_t>(i)])
               << node_ctx->DebugString(node_ctx->input(i));
      }
    }

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
          (void)input_shapes.emplace_back(proto);
        }
        AssembleInputDesc(input_shapes, input_types, node);
      }
    }

    if (!output_types.empty()) {
      TensorPartialShapes output_shapes;
      for (int i = 0; i < node_ctx->num_outputs(); ++i) {
        tensorflow::TensorShapeProto proto;
        node_ctx->ShapeHandleToProto(node_ctx->output(i), &proto);
        (void)output_shapes.emplace_back(proto);
        DLOG() << "    output " << i << ": " << tensorflow::DataTypeString(output_types[static_cast<size_t>(i)])
               << node_ctx->DebugString(node_ctx->output(i));
      }
      AssembleOutputDesc(output_shapes, output_types, node);
    }
  };
  tensorflow::ReverseDFS(*graph, {}, node_shape_inference_lambda);
}
void AssembleParserAddons(TFE_Context *context, tensorflow::Graph *graph) {
  AssembleParserAddons(npu::UnwrapCtx(context)->FuncLibDef(), graph);
}
}  // namespace npu
