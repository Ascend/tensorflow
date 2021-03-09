/**
* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
* Description: Common depends and micro defines for and only for data preprocess module
*/

#ifndef TENSORFLOW_NPU_PARSER_H
#define TENSORFLOW_NPU_PARSER_H

#include <utility>

#include "npu_types.h"
#include "npu_unwrap.h"
#include "npu_utils.h"

#include "graph/types.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"

namespace {
const std::string kInputDesc = "input_tensor_desc";
const std::string kOutputDesc = "output_tensor_desc";
const std::string kFormat = "serialize_format";
const std::string kType = "serialize_datatype";
const std::string kShape = "serialize_shape";
const std::string kSubGraph = "SubGraph";
}  // namespace

template <typename T>
static tensorflow::AttrValue BuildDescAttr(T shapes, TensorDataTypes types) {
  tensorflow::AttrValue desc_attr;
  for (size_t i = 0; i < types.size(); i++) {
    auto desc = desc_attr.mutable_list()->add_func();
    desc->set_name(std::to_string(i));

    tensorflow::AttrValue shape_value;
    for (int j = 0; j < shapes[i].dims(); j++) { shape_value.mutable_list()->add_i(shapes[i].dim_size(j)); }
    desc->mutable_attr()->insert({kShape, shape_value});

    tensorflow::AttrValue type_value;
    type_value.set_i(static_cast<int64_t>(types[i]));
    desc->mutable_attr()->insert({kType, type_value});

    tensorflow::AttrValue format_value;
    format_value.set_i(static_cast<int>(ge::Format::FORMAT_NHWC));
    desc->mutable_attr()->insert({kFormat, format_value});
  }
  return desc_attr;
}

static void AssembleDesc(TensorPartialShapes shapes, TensorDataTypes types, const std::string &name,
                         tensorflow::NodeDef *ndef) {
  tensorflow::AddNodeAttr(name, BuildDescAttr(std::move(shapes), std::move(types)), ndef);
}

static void AssembleDesc(TensorShapes shapes, TensorDataTypes types, const std::string &name,
                         tensorflow::NodeDef *ndef) {
  tensorflow::AddNodeAttr(name, BuildDescAttr(std::move(shapes), std::move(types)), ndef);
}

static void AssembleInputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::NodeDef *ndef) {
  AssembleDesc(std::move(shapes), std::move(types), kInputDesc, ndef);
}

static void AssembleOutputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::NodeDef *ndef) {
  AssembleDesc(std::move(shapes), std::move(types), kOutputDesc, ndef);
}

static void AssembleInputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::NodeDef *ndef) {
  AssembleDesc(std::move(shapes), std::move(types), kInputDesc, ndef);
}

static void AssembleOutputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::NodeDef *ndef) {
  AssembleDesc(std::move(shapes), std::move(types), kOutputDesc, ndef);
}

static void AssembleInputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::Node *n) {
  n->AddAttr(kInputDesc, BuildDescAttr(std::move(shapes), std::move(types)));
}

static void AssembleOutputDesc(TensorShapes shapes, TensorDataTypes types, tensorflow::Node *n) {
  n->AddAttr(kOutputDesc, BuildDescAttr(std::move(shapes), std::move(types)));
}

static void AssembleInputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node *n) {
  n->AddAttr(kInputDesc, BuildDescAttr(std::move(shapes), std::move(types)));
}

static void AssembleOutputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node *n) {
  n->AddAttr(kOutputDesc, BuildDescAttr(std::move(shapes), std::move(types)));
}

static void AssembleOpDef(const tensorflow::OpRegistrationData *op_data, tensorflow::Node *n) {
  std::string serialized_op_def;
  op_data->op_def.SerializeToString(&serialized_op_def);
  n->AddAttr("op_def", serialized_op_def);
}

static void AssembleOpDef(tensorflow::Node *n) {
  const tensorflow::OpRegistrationData *op_reg_data;
  tensorflow::OpRegistry::Global()->LookUp(n->type_string(), &op_reg_data);
  std::string serialized_op_def;
  op_reg_data->op_def.SerializeToString(&serialized_op_def);
  n->AddAttr("op_def", serialized_op_def);
}

static void AssembleOpDef(const tensorflow::OpRegistrationData *op_data, tensorflow::NodeDef *ndef) {
  std::string serialized_op_def;
  op_data->op_def.SerializeToString(&serialized_op_def);
  tensorflow::AddNodeAttr("op_def", serialized_op_def, ndef);
}

static void AssembleOpDef(tensorflow::NodeDef *ndef) {
  const tensorflow::OpRegistrationData *op_reg_data;
  tensorflow::OpRegistry::Global()->LookUp(ndef->op(), &op_reg_data);
  std::string serialized_op_def;
  op_reg_data->op_def.SerializeToString(&serialized_op_def);
  tensorflow::AddNodeAttr("op_def", serialized_op_def, ndef);
}

#endif  //TENSORFLOW_NPU_PARSER_H
