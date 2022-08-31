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

#ifndef NPU_DEVICE_CORE_NPU_PARSER_H
#define NPU_DEVICE_CORE_NPU_PARSER_H

#include <utility>

#include "npu_types.h"
#include "npu_unwrap.h"
#include "npu_utils.h"

#include "graph/types.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"

namespace {
const std::string kInputDesc = "input_tensor_desc";
const std::string kOutputDesc = "output_tensor_desc";
const std::string kFormat = "serialize_format";
const std::string kType = "serialize_datatype";
const std::string kShape = "serialize_shape";
}  // namespace

namespace npu {
template <typename T>
static inline tensorflow::AttrValue BuildDescAttr(T shapes, TensorDataTypes types) {
  tensorflow::AttrValue desc_attr;
  for (size_t i = 0; i < types.size(); i++) {
    auto desc = desc_attr.mutable_list()->add_func();
    desc->set_name(std::to_string(i));

    tensorflow::AttrValue shape_value;
    if (shapes[i].unknown_rank()) {
      const int kUnknownRankDimSize = -2;
      shape_value.mutable_list()->add_i(kUnknownRankDimSize);
    } else {
      for (int j = 0; j < shapes[i].dims(); j++) {
        shape_value.mutable_list()->add_i(shapes[i].dim_size(j));
      }
    }
    (void)desc->mutable_attr()->insert({kShape, shape_value});

    tensorflow::AttrValue type_value;
    type_value.set_i(static_cast<int64_t>(types[i]));
    (void)desc->mutable_attr()->insert({kType, type_value});

    tensorflow::AttrValue format_value;
    format_value.set_i(static_cast<int>(ge::Format::FORMAT_NHWC));
    (void)desc->mutable_attr()->insert({kFormat, format_value});
  }
  return desc_attr;
}

/**
 * @breif: assemble desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param name: node name
 * @param ndef: tensorflow node def
 */
void AssembleDesc(TensorPartialShapes shapes, TensorDataTypes types, const std::string &name,
                  tensorflow::NodeDef *ndef);

/**
 * @breif: assemble desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param name: node name
 * @param ndef: tensorflow node def
 */
void AssembleDesc(TensorShapes shapes, TensorDataTypes types, const std::string &name, tensorflow::NodeDef *ndef);

/**
 * @breif: assemble input desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node ndef
 */
TF_ATTRIBUTE_UNUSED void AssembleInputDesc(TensorPartialShapes shapes, TensorDataTypes types,
                                           tensorflow::NodeDef *ndef);

/**
 * @breif: assemble output desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node ndef
 */
TF_ATTRIBUTE_UNUSED void AssembleOutputDesc(TensorPartialShapes shapes, TensorDataTypes types,
                                            tensorflow::NodeDef *ndef);

/**
 * @breif: assemble input desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node def
 */
TF_ATTRIBUTE_UNUSED void AssembleInputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::NodeDef *ndef);

/**
 * @breif: assemble output desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param ndef: tensorflow node def
 */
TF_ATTRIBUTE_UNUSED void AssembleOutputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::NodeDef *ndef);

/**
 * @breif: assemble input desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleInputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::Node *n);

/**
 * @breif: assemble output desc
 * @param shapes: tensor shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleOutputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::Node *n);

/**
 * @breif: assemble input desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleInputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node *n);

/**
 * @breif: assemble output desc
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleOutputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node *n);

/**
 * @breif: assemble op def
 * @param op_data: tensorflow op registration data
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleOpDef(const tensorflow::OpRegistrationData *op_data, tensorflow::Node *n);

/**
 * @breif: assemble op def
 * @param n: tensorflow node
 */
TF_ATTRIBUTE_UNUSED void AssembleOpDef(tensorflow::Node *n);

/**
 * @breif: assemble op def
 * @param op_data: tensorflow op registration data
 * @param ndef: tensorflow node def
 */
TF_ATTRIBUTE_UNUSED inline void AssembleOpDef(const tensorflow::OpRegistrationData *op_data,
                                              tensorflow::NodeDef *ndef) {
  std::string serialized_op_def;
  (void)op_data->op_def.SerializeToString(&serialized_op_def);
  tensorflow::AddNodeAttr("op_def", serialized_op_def, ndef);
}

/**
 * @breif: assemble op def
 * @param ndef: tensorflow node def
 */
TF_ATTRIBUTE_UNUSED inline void AssembleOpDef(tensorflow::NodeDef *ndef) {
  const tensorflow::OpRegistrationData *op_reg_data;
  (void)tensorflow::OpRegistry::Global()->LookUp(ndef->op(), &op_reg_data);
  std::string serialized_op_def;
  (void)op_reg_data->op_def.SerializeToString(&serialized_op_def);
  tensorflow::AddNodeAttr("op_def", serialized_op_def, ndef);
}

void AssembleParserAddons(TFE_Context *context, tensorflow::Graph *graph);

void AssembleParserAddons(const tensorflow::FunctionLibraryDefinition *lib_def, tensorflow::Graph *graph);
}  // namespace npu
#endif  // NPU_DEVICE_CORE_NPU_PARSER_H
