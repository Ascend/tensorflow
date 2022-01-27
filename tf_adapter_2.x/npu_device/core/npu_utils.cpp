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

#include "npu_utils.h"

#include <queue>

#include "tensorflow/core/graph/algorithm.h"

#include "npu_env.h"

namespace npu {
ScopeTensorHandleDeleter::~ScopeTensorHandleDeleter() {
  for (auto handle : handles_) {
    TFE_DeleteTensorHandle(handle);
  }
}

void ScopeTensorHandleDeleter::Guard(TFE_TensorHandle *handle) {
  if (handle != nullptr) {
    handles_.insert(handle);
  }
}

/**
 * @brief: map ge type to tf
 * @param ge_type: ge type
 * @param acl_type: tf type
 */
tensorflow::Status MapGeType2Tf(ge::DataType ge_type, tensorflow::DataType &tf_type) {
  static std::map<ge::DataType, tensorflow::DataType> kGeType2Tf = {
    {ge::DT_FLOAT, tensorflow::DT_FLOAT},           {ge::DT_DOUBLE, tensorflow::DT_DOUBLE},
    {ge::DT_INT32, tensorflow::DT_INT32},           {ge::DT_UINT8, tensorflow::DT_UINT8},
    {ge::DT_INT16, tensorflow::DT_INT16},           {ge::DT_INT8, tensorflow::DT_INT8},
    {ge::DT_STRING, tensorflow::DT_STRING},         {ge::DT_COMPLEX64, tensorflow::DT_COMPLEX64},
    {ge::DT_INT64, tensorflow::DT_INT64},           {ge::DT_BOOL, tensorflow::DT_BOOL},
    {ge::DT_QINT8, tensorflow::DT_QINT8},           {ge::DT_QUINT8, tensorflow::DT_QUINT8},
    {ge::DT_QINT32, tensorflow::DT_QINT32},         {ge::DT_QINT16, tensorflow::DT_QINT16},
    {ge::DT_QUINT16, tensorflow::DT_QUINT16},       {ge::DT_UINT16, tensorflow::DT_UINT16},
    {ge::DT_COMPLEX128, tensorflow::DT_COMPLEX128}, {ge::DT_RESOURCE, tensorflow::DT_RESOURCE},
    {ge::DT_VARIANT, tensorflow::DT_VARIANT},       {ge::DT_UINT32, tensorflow::DT_UINT32},
    {ge::DT_UINT64, tensorflow::DT_UINT64},         {ge::DT_STRING_REF, tensorflow::DT_STRING_REF},
    {ge::DT_FLOAT16, tensorflow::DT_HALF},
  };
  if (kGeType2Tf.find(ge_type) == kGeType2Tf.end()) {
    return tensorflow::errors::InvalidArgument("Unsupport ge data type enmu value ", ge_type, " by tf");
  }
  tf_type = kGeType2Tf[ge_type];
  return tensorflow::Status::OK();
}

/**
 * @brief: map tf type to ge
 * @param ge_type: tf type
 * @param acl_type: ge type
 */
tensorflow::Status MapTfType2Ge(tensorflow::DataType tf_type, ge::DataType &ge_type) {
  static std::map<tensorflow::DataType, ge::DataType> kTfType2Ge = {
    {tensorflow::DT_FLOAT, ge::DT_FLOAT},           {tensorflow::DT_DOUBLE, ge::DT_DOUBLE},
    {tensorflow::DT_INT32, ge::DT_INT32},           {tensorflow::DT_UINT8, ge::DT_UINT8},
    {tensorflow::DT_INT16, ge::DT_INT16},           {tensorflow::DT_INT8, ge::DT_INT8},
    {tensorflow::DT_STRING, ge::DT_STRING},         {tensorflow::DT_COMPLEX64, ge::DT_COMPLEX64},
    {tensorflow::DT_INT64, ge::DT_INT64},           {tensorflow::DT_BOOL, ge::DT_BOOL},
    {tensorflow::DT_QINT8, ge::DT_QINT8},           {tensorflow::DT_QUINT8, ge::DT_QUINT8},
    {tensorflow::DT_QINT32, ge::DT_QINT32},         {tensorflow::DT_QINT16, ge::DT_QINT16},
    {tensorflow::DT_QUINT16, ge::DT_QUINT16},       {tensorflow::DT_UINT16, ge::DT_UINT16},
    {tensorflow::DT_COMPLEX128, ge::DT_COMPLEX128}, {tensorflow::DT_RESOURCE, ge::DT_RESOURCE},
    {tensorflow::DT_VARIANT, ge::DT_VARIANT},       {tensorflow::DT_UINT32, ge::DT_UINT32},
    {tensorflow::DT_UINT64, ge::DT_UINT64},         {tensorflow::DT_STRING_REF, ge::DT_STRING_REF},
    {tensorflow::DT_HALF, ge::DT_FLOAT16},
  };
  if (kTfType2Ge.find(tf_type) == kTfType2Ge.end()) {
    return tensorflow::errors::InvalidArgument("Unsupport tf data type enmu value ", ge_type, " by ge");
  }
  ge_type = kTfType2Ge[tf_type];
  return tensorflow::Status::OK();
}

/**
 * @brief: map ge type to acl
 * @param ge_type: ge type
 * @param acl_type: acl type
 */
tensorflow::Status MapGeType2Acl(ge::DataType ge_type, aclDataType &acl_type) {
  static std::map<ge::DataType, aclDataType> kGeType2Acl = {
    {ge::DT_FLOAT, ACL_FLOAT},     {ge::DT_DOUBLE, ACL_DOUBLE}, {ge::DT_INT32, ACL_INT32},
    {ge::DT_UINT8, ACL_UINT8},     {ge::DT_INT16, ACL_INT16},   {ge::DT_INT8, ACL_INT8},
    {ge::DT_STRING, ACL_STRING},   {ge::DT_INT64, ACL_INT64},   {ge::DT_BOOL, ACL_BOOL},
    {ge::DT_UINT16, ACL_UINT16},   {ge::DT_UINT32, ACL_UINT32}, {ge::DT_UINT64, ACL_UINT64},
    {ge::DT_FLOAT16, ACL_FLOAT16},
  };
  if (kGeType2Acl.find(ge_type) == kGeType2Acl.end()) {
    return tensorflow::errors::InvalidArgument("Unsupport ge data type enmu value ", ge_type, " by acl");
  }
  acl_type = kGeType2Acl[ge_type];
  return tensorflow::Status::OK();
}

/**
 * @brief: map ge format to acl
 * @param ge_format: ge format
 * @param acl_format: acl format
 */
tensorflow::Status MapGeFormat2Acl(ge::Format ge_format, aclFormat &acl_format) {
  static std::map<ge::Format, aclFormat> kGeFormat2Acl = {{ge::Format::FORMAT_NCHW, ACL_FORMAT_NCHW},
                                                          {ge::Format::FORMAT_NHWC, ACL_FORMAT_NHWC},
                                                          {ge::Format::FORMAT_ND, ACL_FORMAT_ND},
                                                          {ge::Format::FORMAT_NC1HWC0, ACL_FORMAT_NC1HWC0},
                                                          {ge::Format::FORMAT_FRACTAL_Z, ACL_FORMAT_FRACTAL_Z},
                                                          {ge::Format::FORMAT_NC1HWC0_C04, ACL_FORMAT_NC1HWC0_C04},
                                                          {ge::Format::FORMAT_NDHWC, ACL_FORMAT_NDHWC},
                                                          {ge::Format::FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ},
                                                          {ge::Format::FORMAT_NCDHW, ACL_FORMAT_NCDHW},
                                                          {ge::Format::FORMAT_NDC1HWC0, ACL_FORMAT_NDC1HWC0},
                                                          {ge::Format::FORMAT_FRACTAL_Z_3D, ACL_FRACTAL_Z_3D}};
  if (kGeFormat2Acl.find(ge_format) == kGeFormat2Acl.end()) {
    return tensorflow::errors::InvalidArgument("Unsupport ge format enmu value ", ge_format, " by acl");
  }
  acl_format = kGeFormat2Acl[ge_format];
  return tensorflow::Status::OK();
}

// TODO:在GE处理中，变量名称作为唯一标识，对于shared_name是"_"开头的变量，由于tensorflow禁止变量名以"_"开头，所以无法直接将shared_name
//  作为Node的name，对于GE，则没有这个限制，因而，这个函数需要能够屏蔽这种差异。
std::string WrapResourceName(const std::string &name) {
  if (kCustomKernelEnabled) {
    return name;
  }
  return "cpu_" + name;
}

tensorflow::FunctionDefLibrary CollectGraphSubGraphs(const tensorflow::GraphDef &gdef,
                                                     const tensorflow::FunctionLibraryDefinition *lib_def) {
  tensorflow::FunctionDefLibrary fdef_lib;

  std::unordered_set<std::string> related_function_names;
  std::queue<const tensorflow::FunctionDef *> related_functions;
  for (const auto &n : gdef.node()) {
    for (const auto &attr : n.attr()) {
      if (attr.second.has_func() && related_function_names.insert(attr.second.func().name()).second) {
        const auto *f = lib_def->Find(attr.second.func().name());
        if (f != nullptr) {
          *fdef_lib.add_function() = *f;
          related_functions.push(f);
        } else {
          LOG(ERROR) << "Function " << attr.second.func().name() << " not found";
        }
      }
    }
  }

  while (!related_functions.empty()) {
    const auto *f = related_functions.front();
    related_functions.pop();
    for (const auto &n : f->node_def()) {
      for (const auto &attr : n.attr()) {
        if (attr.second.has_func() && related_function_names.insert(attr.second.func().name()).second) {
          const auto *f_inner = lib_def->Find(attr.second.func().name());
          if (f_inner != nullptr) {
            *fdef_lib.add_function() = *f_inner;
            related_functions.push(f_inner);
          } else {
            LOG(ERROR) << "Function " << attr.second.func().name() << " not found";
          }
        }
      }
    }
  }
  return fdef_lib;
}

void OptimizeStageGraphDumper::Dump(const std::string &stage, const tensorflow::GraphDef &graph_def) {
  std::string graph_name = tensorflow::strings::StrCat(graph_, ".", counter_++, ".", stage, ".pbtxt");
  LOG(INFO) << "Dump graph " << graph_name;
  WriteTextProto(tensorflow::Env::Default(), graph_name, graph_def);
}

void OptimizeStageGraphDumper::DumpWithSubGraphs(const std::string &stage, const tensorflow::GraphDef &graph_def,
                                                 const tensorflow::FunctionLibraryDefinition *lib_def) {
  tensorflow::GraphDef copied_graph_def = graph_def;
  *copied_graph_def.mutable_library() = CollectGraphSubGraphs(graph_def, lib_def);
  Dump(stage, copied_graph_def);
}

/**
 * @brief: prune function
 * @param fdef: function def
 * @param g: graph
 * @param keep_signature: if keep signature or not
 */
void PruneGraphByFunctionSignature(const tensorflow::FunctionDef &fdef, tensorflow::Graph *g, bool keep_signature) {
  std::unordered_set<tensorflow::StringPiece, tensorflow::StringPieceHasher> control_ret_nodes;
  for (const auto &control_ret : fdef.control_ret()) {
    control_ret_nodes.insert(control_ret.second);
  }

  std::unordered_set<const tensorflow::Node *> nodes;
  for (auto n : g->nodes()) {
    if (n->IsControlFlow() || n->op_def().is_stateful() ||
        (control_ret_nodes.find(n->name()) != control_ret_nodes.end())) {
      if (n->type_string() == "VarHandleOp" || n->type_string() == "IteratorV2") {
        continue;
      }
      if (!keep_signature) {
        if (n->IsArg()) {
          continue;
        }
        if (n->IsRetval() && n->attrs().Find("T")->type() == tensorflow::DT_RESOURCE) {
          continue;
        }
      }
      nodes.insert(n);
    }
  }
  bool changed = tensorflow::PruneForReverseReachability(g, std::move(nodes));
  if (changed) {
    tensorflow::FixupSourceAndSinkEdges(g);
  }
}

uint64_t NextUUID() {
  static std::atomic<uint64_t> uuid{0};
  return uuid.fetch_add(1);
}

/**
 * @brief: fix graph arg return value index
 * @param graph: graph
 */
void FixGraphArgRetvalIndex(tensorflow::Graph *graph) {
  std::map<int, tensorflow::Node *> indexed_args;
  std::map<int, tensorflow::Node *> indexed_retvals;
  for (auto node : graph->nodes()) {
    if (node->IsArg()) {
      indexed_args[node->attrs().Find("index")->i()] = node;
    }
    if (node->IsRetval()) {
      indexed_retvals[node->attrs().Find("index")->i()] = node;
    }
  }
  int current_arg_index = 0;
  for (auto indexed_arg : indexed_args) {
    indexed_arg.second->AddAttr("index", current_arg_index++);
  }

  int current_retval_index = 0;
  for (auto indexed_retval : indexed_retvals) {
    indexed_retval.second->AddAttr("index", current_retval_index++);
  }
}
}  // namespace npu