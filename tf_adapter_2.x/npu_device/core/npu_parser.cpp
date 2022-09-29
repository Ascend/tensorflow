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
/**
 * @breif: assemble desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param name: node name
 * @param ndef: tensorflow node def
 */
void AssembleDesc(TensorPartialShapes shapes, TensorDataTypes types, const std::string &name,
                  tensorflow::NodeDef &ndef) {
  tensorflow::AddNodeAttr(name, BuildDescAttr(std::move(shapes), std::move(types)), &ndef);
}

/**
 * @breif: assemble desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param name: node name
 * @param ndef: tensorflow node def
 */
void AssembleDesc(TensorShapes shapes, TensorDataTypes types, const std::string &name, tensorflow::NodeDef &ndef) {
  tensorflow::AddNodeAttr(name, BuildDescAttr(std::move(shapes), std::move(types)), &ndef);
}

/**
 * @breif: assemble input desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node ndef
 */
void AssembleInputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::NodeDef &ndef) {
  AssembleDesc(std::move(shapes), std::move(types), kInputDesc, ndef);
}

/**
 * @breif: assemble output desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node ndef
 */
void AssembleOutputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::NodeDef &ndef) {
  AssembleDesc(std::move(shapes), std::move(types), kOutputDesc, ndef);
}

/**
 * @breif: assemble input desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node def
 */
void AssembleInputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::NodeDef &ndef) {
  AssembleDesc(std::move(shapes), std::move(types), kInputDesc, ndef);
}

/**
 * @breif: assemble output desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node def
 */
void AssembleOutputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::NodeDef &ndef) {
  AssembleDesc(std::move(shapes), std::move(types), kOutputDesc, ndef);
}

/**
 * @breif: assemble input desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
void AssembleInputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::Node &n) {
  n.AddAttr(kInputDesc, BuildDescAttr(std::move(shapes), std::move(types)));
}

/**
 * @breif: assemble output desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
void AssembleOutputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::Node &n) {
  n.AddAttr(kOutputDesc, BuildDescAttr(std::move(shapes), std::move(types)));
}

/**
 * @breif: assemble input desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
void AssembleInputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node &n) {
  n.AddAttr(kInputDesc, BuildDescAttr(std::move(shapes), std::move(types)));
}

/**
 * @breif: assemble output desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
void AssembleOutputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node &n) {
  n.AddAttr(kOutputDesc, BuildDescAttr(std::move(shapes), std::move(types)));
}

/**
 * @breif: assemble op def
 * @param op_data: tensorflow op registration data
 * @param n: tensorflow node
 */
void AssembleOpDef(const tensorflow::OpRegistrationData &op_data, tensorflow::Node &n) {
  std::string serialized_op_def;
  (void)op_data.op_def.SerializeToString(&serialized_op_def);
  n.AddAttr("op_def", serialized_op_def);
}

/**
 * @breif: assemble op def
 * @param n: tensorflow node
 */
void AssembleOpDef(tensorflow::Node &n) {
  const tensorflow::OpRegistrationData *op_reg_data;
  (void)tensorflow::OpRegistry::Global()->LookUp(n.type_string(), &op_reg_data);
  std::string serialized_op_def;
  (void)op_reg_data->op_def.SerializeToString(&serialized_op_def);
  n.AddAttr("op_def", serialized_op_def);
}

void AssembleParserAddons(const tensorflow::FunctionLibraryDefinition *lib_def, tensorflow::Graph *graph) {
  tensorflow::ShapeRefiner shape_refiner(graph->versions(), lib_def);
  auto node_shape_inference_lambda = [&shape_refiner](tensorflow::Node *node) {
    AssembleOpDef(*node);
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
        AssembleInputDesc(input_shapes, input_types, *node);
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
      AssembleOutputDesc(output_shapes, output_types, *node);
    }
  };
  tensorflow::ReverseDFS(*graph, {}, node_shape_inference_lambda);
}
void AssembleParserAddons(TFE_Context *context, tensorflow::Graph *graph) {
  AssembleParserAddons(npu::UnwrapCtx(context)->FuncLibDef(), graph);
}
}  // namespace npu
