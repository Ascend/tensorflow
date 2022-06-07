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
#include <securec.h>

#include "tensorflow/core/graph/algorithm.h"

#include "npu_device.h"
#include "npu_env.h"
#include "npu_logger.h"

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
    {ge::DT_FLOAT16, tensorflow::DT_HALF},          {ge::DT_BF16, tensorflow::DT_BFLOAT16},
  };
  if (kGeType2Tf.find(ge_type) == kGeType2Tf.end()) {
    return tensorflow::errors::InvalidArgument("Unsupported ge data type enmu value ", ge_type, " by tf");
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
    {tensorflow::DT_HALF, ge::DT_FLOAT16},          {tensorflow::DT_BFLOAT16, ge::DT_BF16},
  };
  if (kTfType2Ge.find(tf_type) == kTfType2Ge.end()) {
    return tensorflow::errors::InvalidArgument("Unsupported tf type ", tensorflow::DataTypeString(tf_type), " by ge");
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

std::string WrapResourceName(const std::string &name) {
  if (name[0] == '_') {
    DLOG() << "Replace name " << name << " to npu" << name;
    return "npu" + name;
  }
  return name;
}

bool IsSubstituteNode(const tensorflow::Node *node) {
  auto attr = node->attrs().Find("_is_substitute");
  return (attr != nullptr) && attr->b();
}

bool IsSubstituteNode(const tensorflow::NodeDef &def) {
  auto attr = def.attr().find("_is_substitute");
  return attr != def.attr().end() && attr->second.b();
}

bool IsNodeHasSubgraph(const tensorflow::Node *node) {
  for (auto &attr : node->attrs()) {
    if (attr.second.has_func()) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> GetNodeSubgraphNames(const tensorflow::Node *node) {
  std::vector<std::string> subgraphs;
  for (auto &attr : node->attrs()) {
    if (attr.second.has_func()) {
      subgraphs.push_back(attr.second.func().name());
    }
  }
  return subgraphs;
}

bool IsNodeHasSubstituteInput(const tensorflow::Node *node) {
  for (auto in_node : node->in_nodes()) {
    if (IsSubstituteNode(in_node)) {
      return true;
    }
  }
  return false;
}

tensorflow::DataType EdgeDataType(const tensorflow::Edge &edge) { return edge.src()->output_type(edge.src_output()); }

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

OptimizeStageGraphDumper::OptimizeStageGraphDumper(const std::string &graph) {
  enabled_ = kDumpExecutionDetail || kDumpGraph;
  if (enabled_) {
    graph_ = graph;
    counter_ = 0;
  }
}

void OptimizeStageGraphDumper::Dump(const std::string &stage, const tensorflow::GraphDef &graph_def) {
  if (!enabled_) {
    return;
  }
  std::string graph_name = tensorflow::strings::StrCat(graph_, ".", counter_++, ".", stage, ".pbtxt");
  DLOG() << "Dump graph " << graph_name;
  WriteTextProto(tensorflow::Env::Default(), graph_name, graph_def);
}

void OptimizeStageGraphDumper::DumpWithSubGraphs(const std::string &stage, const tensorflow::GraphDef &graph_def,
                                                 const tensorflow::FunctionLibraryDefinition *lib_def) {
  if (!enabled_) {
    return;
  }
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
      if ((!keep_signature) && n->IsArg()) {
        continue;
      }
      nodes.insert(n);
    }
  }
  bool changed = tensorflow::PruneForReverseReachability(g, std::move(nodes));
  if (changed) {
    tensorflow::FixupSourceAndSinkEdges(g);
  }
}

std::set<std::string> GetNodeSubgraph(const tensorflow::Node *node) {
  std::set<std::string> fns;
  for (const auto &attr : node->attrs()) {
    if (attr.second.has_func()) {
      fns.insert(attr.second.func().name());
    }
  }
  return fns;
}

tensorflow::Status GetSubgraphUnsupportedOps(NpuDevice *device, const tensorflow::Node *node,
                                             const tensorflow::FunctionLibraryDefinition *lib_def,
                                             std::set<std::string> &unsupported_ops) {
  // For subgraphs of node A, if subgraph has node must place on cpu, A must place on cpu
  // If A placed on cpu, any subraph has node that must place on npu, will be mix partitioned
  auto related_function_names = GetNodeSubgraph(node);
  std::queue<const tensorflow::FunctionDef *> related_functions;

  for (auto &fn : related_function_names) {
    const auto *f = lib_def->Find(fn);
    NPU_REQUIRES(f != nullptr, tensorflow::errors::Internal("Function ", fn, " not found"));
    related_functions.push(f);
  }

  while (!related_functions.empty()) {
    const auto *func = related_functions.front();
    related_functions.pop();
    for (const auto &n : func->node_def()) {
      if (!device->Supported(n.op())) {
        unsupported_ops.insert(n.op());
      }
      for (const auto &attr : n.attr()) {
        auto &fn = attr.second.func().name();
        if (attr.second.has_func() && related_function_names.insert(fn).second) {
          const auto *f = lib_def->Find(attr.second.func().name());
          NPU_REQUIRES(f != nullptr, tensorflow::errors::Internal("Function ", fn, " not found"));
          related_functions.push(f);
        }
      }
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status GetGraphUnsupportedOps(NpuDevice *device, tensorflow::Graph *graph,
                                          const tensorflow::FunctionLibraryDefinition *lib_def,
                                          std::set<std::string> &unsupported_ops) {
  for (auto node : graph->op_nodes()) {
    if (!device->Supported(node->type_string())) {
      unsupported_ops.insert(node->type_string());
    }
    NPU_REQUIRES_OK(GetSubgraphUnsupportedOps(device, node, lib_def, unsupported_ops));
  }
  return tensorflow::Status::OK();
}

namespace {
bool IsNegligibleUnknownShapeOutput(const tensorflow::Node *node, int index) {
  const static std::map<std::string, std::set<int>> kNegligibleUnknownShapeOutput = {{"FusedBatchNormV3", {5}}};
  auto iter = kNegligibleUnknownShapeOutput.find(node->type_string());
  return (iter != kNegligibleUnknownShapeOutput.end()) && (iter->second.count(index) > 0U);
}

bool IsGraphHasAnyUnknownShapeNode(const tensorflow::Graph *graph, const tensorflow::FunctionLibraryDefinition *lib_def,
                                   std::queue<std::unique_ptr<tensorflow::Graph>> &q) {
  tensorflow::ShapeRefiner shape_refiner(graph->versions(), lib_def);
  std::atomic<bool> has_unknown_shape_node{false};
  auto node_shape_inference_lambda = [&q, &lib_def, &has_unknown_shape_node, &shape_refiner](tensorflow::Node *node) {
    if (has_unknown_shape_node) {
      return;
    }
    auto status = shape_refiner.AddNode(node);
    if (!status.ok()) {
      has_unknown_shape_node = true;
      LOG(ERROR) << node->name() << "[" << node->type_string() << "] infer shape failed " << status.error_message();
      return;
    }
    auto node_ctx = shape_refiner.GetContext(node);
    for (int i = 0; i < node_ctx->num_outputs(); ++i) {
      tensorflow::TensorShapeProto proto;
      node_ctx->ShapeHandleToProto(node_ctx->output(i), &proto);
      tensorflow::PartialTensorShape shape(proto);
      if ((!shape.IsFullyDefined()) && (!IsNegligibleUnknownShapeOutput(node, i))) {
        DLOG() << node->name() << "[" << node->type_string() << "] unknown shape output " << i << shape.DebugString();
        has_unknown_shape_node = true;
        return;
      }
    }
    if (node->IsWhileNode() || node->IsCaseNode() || node->IsIfNode()) {
      auto fns = GetNodeSubgraphNames(node);
      std::vector<tensorflow::PartialTensorShape> shapes;
      for (int i = (node->IsWhileNode() ? 0 : 1); i < node_ctx->num_inputs(); i++) {
        tensorflow::TensorShapeProto proto;
        node_ctx->ShapeHandleToProto(node_ctx->input(i), &proto);
        shapes.emplace_back(proto);
      }
      for (auto &fn : fns) {
        std::unique_ptr<tensorflow::FunctionBody> fbody;
        auto &fdef = *lib_def->Find(fn);
        status = FunctionDefToBodyHelper(fdef, tensorflow::AttrSlice{}, lib_def, &fbody);
        if (!status.ok()) {
          has_unknown_shape_node = true;
          LOG(ERROR) << node->name() << "[" << node->type_string()
                     << "] convert function to graph for infer shape failed " << status.error_message();
          return;
        }
        auto fg = fbody->graph->Clone();
        for (auto n : fg->op_nodes()) {
          if (!n->IsArg()) {
            continue;
          }
          n->AddAttr("_output_shapes",
                     std::vector<tensorflow::PartialTensorShape>{shapes[n->attrs().Find("index")->i()]});
        }
        q.push(std::move(fg));
      }
    }
  };
  tensorflow::ReverseDFS(*graph, {}, node_shape_inference_lambda);
  return has_unknown_shape_node;
}
}  // namespace

bool IsGraphHasAnyUnknownShapeNode(const tensorflow::Graph *graph,
                                   const tensorflow::FunctionLibraryDefinition *lib_def) {
  std::queue<std::unique_ptr<tensorflow::Graph>> q;
  if (IsGraphHasAnyUnknownShapeNode(graph, lib_def, q)) {
    return true;
  }
  while (!q.empty()) {
    auto g = std::move(q.front());
    q.pop();
    if (IsGraphHasAnyUnknownShapeNode(g.get(), lib_def, q)) {
      return true;
    }
  }
  return false;
}

bool IsGraphNeedLoop(const tensorflow::Graph *graph, tensorflow::Node **key) {
  *key = nullptr;
  std::vector<tensorflow::Node *> while_nodes;
  for (auto node : graph->op_nodes()) {
    if (!node->IsWhileNode()) {
      continue;
    }
    while_nodes.push_back(node);
    if (node->attrs().Find("_consumed_iterators") != nullptr) {
      DLOG() << "Found while node " << node->name() << " consumed iterator";
      *key = node;
    }
  }
  if (*key == nullptr || while_nodes.size() > 1) {
    DLOG() << "Skip check as " << ((*key) ? "multi" : "no") << " while nodes in graph";
    return false;
  }
  size_t reserved_nums = 0;
  const std::function<void(const tensorflow::Node *)> &enter = [&reserved_nums](const tensorflow::Node *node) {
    if (node->IsOp()) {
      reserved_nums++;
    }
  };
  tensorflow::ReverseDFSFrom(*graph, {*key}, enter, {}, {}, {});
  DLOG() << "Reserved nodes " << reserved_nums << " vs. totally " << graph->num_op_nodes();
  return static_cast<int>(reserved_nums) == graph->num_op_nodes();
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

std::string SetToString(const std::set<std::string> &vec) {
  if (vec.empty()) {
    return "[]";
  }
  std::string s = "[";
  size_t index = 0U;
  for (auto &v : vec) {
    s += v;
    if (++index != vec.size()) {
      s += ",";
    }
  }
  return s + "]";
}

void NpuCustomizedOptimizeGraph(tensorflow::FunctionLibraryRuntime *lib, std::unique_ptr<tensorflow::Graph> *g) {
  tensorflow::GraphOptimizer::Options options;
  options.cf_consider_fn = [](const tensorflow::Node *n) {
    for (const auto &output_arg : n->op_def().output_arg()) {
      if (output_arg.type() == tensorflow::DT_VARIANT) {
        return false;
      }
    }
    return true;
  };
  tensorflow::OptimizeGraph(lib, g, options);
}

tensorflow::Status LoopCopy(void *dst_ptr, void *src_ptr, size_t src_size) {
  size_t copy_size = 0UL;
  size_t org_src_size = src_size;
  do {
    size_t src_copy_size = (src_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : src_size;
    if (memcpy_s(dst_ptr, src_copy_size, src_ptr, src_copy_size) != EOK) {
      return tensorflow::errors::Internal("loop memory copy failed , dst:", dst_ptr, ", dst_size:", src_copy_size,
                                          ", src:", src_ptr, ", src_size:", src_copy_size);
    }
    copy_size += src_copy_size;
    dst_ptr += src_copy_size;
    src_ptr += src_copy_size;
    src_size -= src_copy_size;
  } while (copy_size < org_src_size);
  return tensorflow::Status::OK();
}

size_t CreateChannelCapacity(const npu::TensorPartialShapes &shapes, const npu::TensorDataTypes &types) {
  const size_t kMaxChannelCapacity = 128UL;
  const size_t kStringTypeCapacity = 64UL;
  const size_t kUnknownShapeCapacity = 3UL;
  const size_t kMinChannelCapacity = 2UL;
  const int32_t kInvalidCpacity = -1;
  constexpr size_t kDefaultDataSize = 2 * 1024 * 1024 * 1024UL;
  constexpr int64_t kSizeTMaxsize = 16 * 1024 * 1024 * 1024UL;

  size_t total_sizes = 0UL;
  for (size_t i = 0UL; i < types.size(); i++) {
    tensorflow::DataType data_type = types.at(i);
    if (data_type == tensorflow::DT_STRING) {
      return kStringTypeCapacity;
    }
    if (!shapes[i].IsFullyDefined()) {
      return kUnknownShapeCapacity;
    }
    int64_t result = 0;
    if (shapes[i].num_elements() > 0 &&
        tensorflow::DataTypeSize(data_type) > (kSizeTMaxsize / shapes[i].num_elements())) {
      return kInvalidCpacity;
    } else {
      result = shapes[i].num_elements() * tensorflow::DataTypeSize(data_type);
    }
    if (result > kSizeTMaxsize - total_sizes) {
      return kInvalidCpacity;
    }
    total_sizes += static_cast<size_t>(result);
  }
  return std::min(kMaxChannelCapacity, std::max(kMinChannelCapacity, (kDefaultDataSize / total_sizes)));
}
}  // namespace npu