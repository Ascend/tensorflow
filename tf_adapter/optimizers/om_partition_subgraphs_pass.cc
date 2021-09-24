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

#include "tf_adapter/optimizers/om_partition_subgraphs_pass.h"

#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/public/session_options.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/infershape_util.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/npu_ops_identifier.h"
#include "tf_adapter/common/graphcycles.h"

namespace tensorflow {
static const int64 kMicrosToMillis = 1000;

namespace OMSplitter {
const char *const ARG_OP = "_Arg";
const char *const RET_OP = "_Retval";
const char *const PARTITION_SUB_GRAPH_ATTR = "_subgraphs";
const std::string ATTR_NAME_FRAMEWORK_FUNC_DEF = "func_def";
const std::string ATTR_NAME_SHARED_NAME = "shared_name";
const std::string ATTR_VALUE_SHARED_NAME = "iterator_default";
const std::string ATTR_VALUE_SCOPE_NAME = "_without_npu_compile";
const int MAX_GROUP_SIZE = 100000;
const uint32_t MIN_CLUSTER_SIZE = 2;
std::atomic<bool> compile_mode(false);
mutex support_node_mu;
std::set<string> not_support_nodes;

// Graph to FunctionDef conversion.
Status OMSubGraphToFunctionDef(const Graph &graph, const string &name, FunctionDef *fdef) {
  fdef->mutable_signature()->set_name(name);
  std::unordered_map<string, string> tensorRenaming;
  std::unordered_map<string, string> returnValues;

  for (Node const *node : graph.op_nodes()) {
    REQUIRES_NOT_NULL(node);
    if (node->type_string() == ARG_OP) {
      int index;
      DataType type;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &type));
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      while (fdef->signature().input_arg_size() <= index) { fdef->mutable_signature()->add_input_arg(); }
      OpDef::ArgDef *argdef = fdef->mutable_signature()->mutable_input_arg(index);
      REQUIRES_NOT_NULL(argdef);
      argdef->set_type(type);
      argdef->set_name(node->name());
      tensorRenaming[strings::StrCat(node->name(), ":0")] = node->name();
      continue;
    }

    if (node->type_string() == RET_OP) {
      int index;
      DataType type;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &type));
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      while (fdef->signature().output_arg_size() <= index) { fdef->mutable_signature()->add_output_arg(); }
      OpDef::ArgDef *argdef = fdef->mutable_signature()->mutable_output_arg(index);
      REQUIRES_NOT_NULL(argdef);
      argdef->set_type(type);
      argdef->set_name(node->name());
      const Edge *edge = nullptr;
      TF_CHECK_OK(node->input_edge(0, &edge));
      returnValues[node->name()] = strings::StrCat(edge->src()->name(), ":", edge->src_output());
      continue;
    }

    NodeDef *nodeDef = fdef->add_node_def();
    REQUIRES_NOT_NULL(nodeDef);
    *nodeDef = node->def();
    if (!node->assigned_device_name().empty()) { nodeDef->set_device(node->assigned_device_name()); }
    nodeDef->set_name(node->name());
    // Reset input names based on graph rather than the NodeDef.
    nodeDef->clear_input();
    // Edges, indexed by dst_input.
    std::vector<const Edge *> inEdges;
    std::vector<const Edge *> ctrlEdges;
    for (Edge const *edge : node->in_edges()) {
      REQUIRES_NOT_NULL(edge);
      REQUIRES_NOT_NULL(edge->src());
      if (edge->src()->IsSource()) { continue; }

      if (edge->IsControlEdge()) {
        ctrlEdges.push_back(edge);
      } else {
        unsigned int dst_input = edge->dst_input() < 0 ? 0 : static_cast<unsigned int>(edge->dst_input());
        if (inEdges.size() <= dst_input) {
          try {
            inEdges.resize(dst_input + 1);
          } catch (...) { return errors::InvalidArgument("inEdges resize is failed, resize is %u", dst_input + 1); }
        }
        inEdges[dst_input] = edge;
      }
    }

    // Add regular inputs
    for (auto edge : inEdges) {
      REQUIRES_NOT_NULL(edge);
      REQUIRES_NOT_NULL(edge->src());
      nodeDef->add_input(strings::StrCat(edge->src()->name(), ":", edge->src_output()));
    }

    // Add control inputs
    for (const Edge *edge : ctrlEdges) {
      REQUIRES_NOT_NULL(edge);
      REQUIRES_NOT_NULL(edge->src());
      nodeDef->add_input(strings::StrCat("^", edge->src()->name()));
    }

    // Populate tensorRenaming.
    NameRangeMap outputRanges;
    TF_RETURN_IF_ERROR(NameRangesForNode(*node, node->op_def(), nullptr, &outputRanges));
    for (const auto &output : outputRanges) {
      for (int i = output.second.first; i < output.second.second; ++i) {
        const string tensorName = strings::StrCat(nodeDef->name(), ":", output.first, ":", i - output.second.first);
        tensorRenaming[strings::StrCat(node->name(), ":", i)] = tensorName;
      }
    }
  }

  // Detect missing function inputs.
  for (int i = 0; i < fdef->signature().input_arg_size(); ++i) {
    const string &inputName = fdef->signature().input_arg(i).name();
    if (inputName.empty()) { return errors::InvalidArgument("Missing input ", i, " to function ", name); }
  }

  // Remap input names.  We do this as a second pass to allow the nodes to be in
  // any order.
  for (int nIndex = 0; nIndex < fdef->node_def_size(); ++nIndex) {
    NodeDef *nodeDef = fdef->mutable_node_def(nIndex);
    for (int i = 0; i < nodeDef->input_size(); ++i) {
      if (str_util::StartsWith(nodeDef->input(i), "^")) {
        // Control input
        const string inputCtrlName = nodeDef->input(i).substr(1);
        if (inputCtrlName.empty()) {
          return errors::InvalidArgument("Could not remap control input ", i, ", '", nodeDef->input(i), "', of node '",
                                         nodeDef->name(), "' in function ", name);
        }
        *nodeDef->mutable_input(i) = strings::StrCat("^", inputCtrlName);
      } else {
        const auto iter = tensorRenaming.find(nodeDef->input(i));
        if (iter == tensorRenaming.end()) {
          return errors::InvalidArgument("Could not remap input ", i, ", '", nodeDef->input(i), "', of node '",
                                         nodeDef->name(), "' in function ", name);
        }
        *nodeDef->mutable_input(i) = iter->second;
      }
    }
  }

  // Remap return values.
  for (int r = 0; r < fdef->signature().output_arg_size(); ++r) {
    const string &retName = fdef->signature().output_arg(r).name();
    if (retName.empty()) { return errors::InvalidArgument("Missing output ", r, " to function ", name); }
    const string &returnValue = returnValues[retName];
    const auto iter = tensorRenaming.find(returnValue);
    if (iter == tensorRenaming.end()) {
      return errors::InvalidArgument("Could not remap return value ", r, ", '", retName, "', of '", returnValue,
                                     "' in function ", name);
    }
    (*fdef->mutable_ret())[retName] = iter->second;
  }

  return Status::OK();
}

struct NodeCompare {
  bool operator()(const Node *a, const Node *b) const { return a->id() < b->id(); }
};
using OrderedNodeSet = std::set<Node *, NodeCompare>;

bool EndsWith(const std::string &str, const std::string &suffix) {
  return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool IsWhiteListSupport(const string &op_name, bool mix_compile_mode, const string &node_name,
                        bool support_const = false) {
  static const std::string suffix_op = "Dataset";
  static const std::string suffix_op_v2 = "DatasetV2";

  auto identifier = NpuOpsIdentifier::GetInstance(mix_compile_mode);

  bool ans = (identifier->IsNpuSupported(op_name, node_name)) && !EndsWith(op_name, suffix_op) &&
             !EndsWith(op_name, suffix_op_v2) && (support_const || !(op_name == "Const")) &&
             !(op_name == "_Arg") && !(op_name == "_Retval") && !(op_name == "StringJoin");
  if (!ans) {
    mutex_lock lock(support_node_mu);
    auto ret = not_support_nodes.insert(op_name);
    if (ret.second) {
      ADP_LOG(INFO) << "node: " << op_name << " is not in white list, "
                    << "so currently not support";
    }
  }

  return ans;
}

Status SetIteratorShardName(Node *node) {
  if (node->type_string() != "Iterator" && node->type_string() != "IteratorV2") {
    return errors::InvalidArgument("Node op type is not Iterator.");
  }
  string shardName;
  Status s = GetNodeAttr(node->attrs(), ATTR_NAME_SHARED_NAME, &shardName);
  if (s.code() == error::Code::NOT_FOUND) {
    node->AddAttr(ATTR_NAME_SHARED_NAME, node->name());
    return Status::OK();
  } else if (!s.ok()) {
    return s;
  }
  node->ClearAttr(ATTR_NAME_SHARED_NAME);
  node->AddAttr(ATTR_NAME_SHARED_NAME, node->name());
  ADP_LOG(INFO) << "shardName is " << shardName;
  return Status::OK();
}

bool IsWithoutNpuScope(const NodeDef &node_def) {
  if (!compile_mode) { return false; }
  if (node_def.attr().count(ATTR_VALUE_SCOPE_NAME)) { return node_def.attr().at(ATTR_VALUE_SCOPE_NAME).b(); }
  return false;
}

bool IsWithoutNpuScope(Node *node) {
  return IsWithoutNpuScope(node->def());
}

// Make sure we don't recurse infinitely on recursive functions.
const int kMaxRecursionDepth = 10;

bool IsNpuSupportingFunc(const string &func_name, FunctionLibraryDefinition *func_lib, int depth) {
  if (func_lib == nullptr) {
    ADP_LOG(ERROR) << "func lib is nullptr, function name is " << func_name;
    LOG(ERROR) << "func lib is nullptr, function name is " << func_name;
    return false;
  }
  if (depth >= kMaxRecursionDepth) {
    ADP_LOG(ERROR) << "Rejecting " << func_name << ": function depth limit exceeded.";
    LOG(ERROR) << "Rejecting " << func_name << ": function depth limit exceeded.";
    return false;
  }
  const FunctionDef *func_def = func_lib->Find(func_name);
  if (func_def == nullptr) {
    return false;
  }
  for (NodeDef node_def : func_def->node_def()) {
    if (node_def.op() == "Const") {
      ADP_LOG(INFO) << "Const in func can dump";
    } else if (!IsNpuSupportingNode(node_def, compile_mode, func_lib)) {
      return false;
    }
    for (const auto &item : node_def.attr()) {
      if (item.second.has_func()) {
        if (!IsNpuSupportingFunc(item.second.func().name(), func_lib, depth + 1)) { return false; }
      }
    }
  }
  return true;
}

bool IsNpuSupportingFunc(Node *node, FunctionLibraryDefinition *func_lib, int depth) {
  for (const auto &it : node->attrs()) {
    if (it.second.has_func()) {
      string func_name = it.second.func().name();
      if (!IsNpuSupportingFunc(func_name, func_lib, depth)) { return false; }
    }
  }
  return true;
}

bool IsNpuSupportingNode(const NodeDef &node_def, bool mix_compile_mode,
                         FunctionLibraryDefinition *func_lib, bool support_const) {
  if (IsWithoutNpuScope(node_def)) { return false; }
  if (IsWhiteListSupport(node_def.op(), mix_compile_mode, node_def.name(), support_const)) { return true; }
  if (IsNpuSupportingFunc(node_def.op(), func_lib, 0)) { return true; }
  return false;
}

bool IsNpuSupportingNode(Node *node, bool mix_compile_mode, FunctionLibraryDefinition *func_lib,
                         bool support_const) {
  return IsNpuSupportingNode(node->def(), mix_compile_mode, func_lib, support_const);
}

bool IsUnSupportedResource(bool mix_compile_mode, Node* node) {
  if (!mix_compile_mode) { return false; }
  for (int32 i = 0; i < node->num_inputs(); i++) {
    const Edge *edge = nullptr;
    if (!node->input_edge(i, &edge).ok()) {
      continue;
    }
    auto dtype = node->input_type(i);
    if (dtype == DT_RESOURCE) {
      return edge->src()->type_string() != "VarHandleOp";
    }
  }
  for (int32 i = 0; i < node->num_outputs(); i++) {
    auto dtype = node->output_type(i);
    if (dtype == DT_RESOURCE) {
      return node->type_string() != "VarHandleOp";
    }
  }
  return false;
}

using NodeSet = std::set<Node *>;
using NodeMap = std::map<Node *, NodeSet>;
using NodeStack = std::vector<Node *>;
using GraphPath = std::vector<Node *>;
using GraphPaths = std::vector<GraphPath>;

int FindNodesInPaths(Node *op_head, NodeSet &ops_tail, NodeSet &ops_save) {
  if (!op_head || ops_tail.count(nullptr) > 0) { return 0; }

  static const size_t kLegalPathSize = 2;
  NodeSet empty;
  NodeSet noagain;
  NodeMap seen;
  NodeStack stack;
  GraphPath path;

  seen.insert(NodeMap::value_type(op_head, empty));
  stack.push_back(op_head);
  while (!stack.empty()) {
    Node *cur_node = stack.back();
    stack.pop_back();

    if (!cur_node) {
      seen.erase(path.back());
      if (path.size() >= kLegalPathSize) { seen[path[path.size() - kLegalPathSize]].erase(path.back()); }
      path.pop_back();
      continue;
    }
    path.push_back(cur_node);
    if (path.size() >= kLegalPathSize) { seen[path[path.size() - kLegalPathSize]].insert(path.back()); }
    stack.push_back(nullptr);

    if (ops_tail.count(cur_node) > 0 || ops_save.count(cur_node) > 0) {
      for (auto node : path) { ops_save.insert(node); }
      continue;
    }

    if (noagain.count(cur_node) > 0) {
      continue;
    }

    for (auto out_node : cur_node->out_nodes()) {
      auto num_outputs = [](Node *node) {
        unsigned int n = 0;
        for (auto out_node : node->out_nodes()) {
          if (out_node->id() > -1) {
            ++n;
          }
        }
        return n;
      };
      if (ops_save.count(out_node) > 0) {
        for (auto node : path) { ops_save.insert(node); }
      }
      auto outputs_size = num_outputs(out_node);
      if (seen.insert(NodeMap::value_type(out_node, empty)).second ||
        seen[out_node].size() < outputs_size) {
        stack.push_back(out_node);
      }
      if (seen[out_node].size() >= outputs_size) {
        noagain.insert(out_node);
      }
    }
  }
  return ops_save.size();
}

using IOP = std::pair<std::string, std::set<std::string>>;
using OneGraphIOP = std::vector<IOP>;
using AllGraphIOP = std::vector<OneGraphIOP>;

int ParseInOutPair(const std::string &in_out_pair, AllGraphIOP &all_graph_iop) {
  using Nodes = std::vector<std::string>;
  using Pair = std::vector<Nodes>;
  using Pairs = std::vector<Pair>;

  int model = 0;
  static const int kModel1 = 1;
  static const int kModel2 = 2;
  static const int kModel3 = 3;
  std::string s;
  Nodes nodes;
  Pair pair;
  Pairs pairs;
  for (char c : in_out_pair) {
      switch (c) {
      case '[':
          ++model;
          break;
      case ']':
      case ',':
          if (model == kModel1 && !pair.empty()) { pairs.emplace_back(std::move(pair)); }
          else if (model == kModel2 && !nodes.empty()) { pair.emplace_back(std::move(nodes)); }
          else if (model == kModel3 && !s.empty()) { nodes.emplace_back(std::move(s)); }
          if (c == ']') { --model; }
          break;
      case ' ':
      case '\t':
      case '\'':
          break;
      default:
          s += c;
      }
  }

  int size = 0;
  std::set<std::string> empty;
  for (auto &pair : pairs) {
    OneGraphIOP one_graph_iop;
    static const size_t kLegalSize = 2;
    if (pair.size() < kLegalSize) { continue; }
    for (auto &in : pair[0]) {
      IOP iop(in, empty);
      for (auto &out : pair[1]) {
        iop.second.insert(out);
      }
      ++size;
      one_graph_iop.push_back(iop);
    }
    all_graph_iop.push_back(one_graph_iop);
  }
  return size;
}

Status FindCandidatesByInOutPair(const Graph &graph, OrderedNodeSet *candidates,
                                 FunctionLibraryDefinition *func_lib, const std::string &in_out_pair, const std::string &in_out_pair_flag) {
  AllGraphIOP all_graph_iop;
  if (ParseInOutPair(in_out_pair, all_graph_iop) <= 0) {
    return errors::Internal("in_out_pair: ", in_out_pair, " is invalid.");
  }
  NodeSet ops_save;
  for (auto &one_graph_iop : all_graph_iop) {
    for (auto &iop : one_graph_iop) {
      static const size_t kLegalLogSize = 2;
      std::string log_out;
      for (auto &out : iop.second) {
        log_out += out + ", ";
      }
      if (log_out.size() > kLegalLogSize) {
        log_out = log_out.substr(0, log_out.size() - kLegalLogSize);
      }
      ADP_LOG(INFO) << iop.first << " -> " << log_out;

      Node *op_head = nullptr;
      NodeSet ops_tail;
      for (auto node : graph.nodes()) {
        if (node->name() == iop.first) { op_head = node; }
        if (iop.second.count(node->name()) > 0) { ops_tail.insert(node); }
      }

      if (!op_head) {
        ADP_LOG(ERROR) << iop.first << " is not in graph.";
      } else if (ops_tail.size() != iop.second.size()) {
        ADP_LOG(ERROR) << log_out << " is not all in graph.";
      } else {
        ADP_LOG(INFO) << "find " << FindNodesInPaths(op_head, ops_tail, ops_save) << " nodes in paths.";
      }
    }
  }
  if ("1" == in_out_pair_flag) {
    for (auto node : ops_save) {
      if (!IsNpuSupportingNode(node, true, func_lib, true)) {
        return errors::Internal(node->name(), " is not supported npu node.");
      } else {
        candidates->insert(node);
      }
    }
  } else {
    for (auto node : graph.op_nodes()) {
      if (ops_save.count(node) > 0) {
        ADP_LOG(INFO) << node->name() << " is excluded in candidates.";
      } else if (!IsNpuSupportingNode(node, true, func_lib, true)) {
        ADP_LOG(WARNING) << node->name() << " is not supported npu node, it will be excluded in candidates.";
      } else {
        candidates->insert(node);
      }
    }
  }
  return Status::OK();
}

Status FindNpuSupportCandidates(const Graph &graph, OrderedNodeSet *candidates, FunctionLibraryDefinition *func_lib,
                                bool enableDP, bool mix_compile_mode) {
  int64 startTime = InferShapeUtil::GetCurrentTimestap();
  compile_mode = mix_compile_mode;
  std::vector<Node *> sortedNodes;
  bool hasIteratorOp = false;
  bool hasMakeIteratorOp = false;
  bool hasOutfeedDequeueOp = false;
  bool hasStopOutfeedDequeueOp = false;
  for (Node *node : graph.op_nodes()) {
    sortedNodes.push_back(node);
    if (node->type_string().find("MakeIterator") != string::npos) {
      hasMakeIteratorOp = true;
    } else if (node->type_string() == "Iterator" || node->type_string() == "IteratorV2") {
      TF_RETURN_IF_ERROR(SetIteratorShardName(node));
      hasIteratorOp = true;
    } else if (node->type_string() == "OutfeedDequeueOp") {
      hasOutfeedDequeueOp = true;
    } else if (node->type_string() == "StopOutfeedDequeueOp") {
      hasStopOutfeedDequeueOp = true;
    }
  }

  if (hasStopOutfeedDequeueOp || hasOutfeedDequeueOp) {
    candidates->clear();
    ADP_LOG(INFO) << "hostcall subgraph will run on host.";
    return Status::OK();
  }

  std::sort(sortedNodes.begin(), sortedNodes.end(), NodeCompare());
  ADP_LOG(INFO) << "FindNpuSupportCandidates enableDP:" << enableDP << ", mix_compile_mode: " << compile_mode
            << ", hasMakeIteratorOp:" << hasMakeIteratorOp << ", hasIteratorOp:" << hasIteratorOp;

  if (hasMakeIteratorOp && hasIteratorOp) {
    candidates->clear();
    ADP_LOG(INFO) << "preprocessing subgraph will at dp_tf_ge_conversion_pass.";
    return Status::OK();
  }

  OrderedNodeSet outSet;
  for (Node *node : sortedNodes) {
    // 0 is function depth
    if (!IsNpuSupportingFunc(node, func_lib, 0)) { continue; }
    if (!node->IsOp()) {  // Ship Sink/Source nodes.
      continue;
    }
    if (enableDP
        && (node->type_string() == "Iterator" || node->type_string() == "IteratorV2"
            || node->type_string() == "IteratorGetNext" || node->type_string() == "AdpGetNext")) {
      if (node->type_string() == "IteratorGetNext") {
        for (Node *n : node->in_nodes()) {
          REQUIRES_NOT_NULL(n);
          ADP_LOG(INFO) << node->name() << " has in nodes " << n->name();
          if (n->type_string() == "Iterator" || n->type_string() == "IteratorV2") { candidates->insert(node); }
        }
      }
      if (node->type_string() == "Iterator" || node->type_string() == "IteratorV2") {
        for (Node *n : node->out_nodes()) {
          REQUIRES_NOT_NULL(n);
          ADP_LOG(INFO) << node->name() << " has in nodes " << n->name();
          if (n->type_string() == "IteratorGetNext") { candidates->insert(node); }
        }
      }
      if (node->type_string() == "AdpGetNext") { candidates->insert(node); }
    } else {
      // Const down when it need down
      if (node->type_string() == "Const") {
        int ctrlEdgeNum = 0;
        for (auto edge : node->in_edges()) {
          REQUIRES_NOT_NULL(edge);
          REQUIRES_NOT_NULL(edge->src());
          if (edge->IsControlEdge() && edge->src()->name() != "_SOURCE" &&
              IsNpuSupportingNode(edge->src(), compile_mode, func_lib)) {
            candidates->insert(node);
            ctrlEdgeNum++;
            break;
          }
        }
        if (ctrlEdgeNum >= 1) { continue; }
      }
      // normal needed down op
      if (IsNpuSupportingNode(node, compile_mode, func_lib) && !IsUnSupportedResource(compile_mode, node)) {
        candidates->insert(node);
      } else {
        outSet.insert(node);
      }
    }
  }

  if (mix_compile_mode) {
    std::vector<ControlFlowInfo> cfInfos;
    Status status = BuildControlFlowInfo(&graph, &cfInfos);
    if (!status.ok()) return status;
    std::set<std::string> unsupportedFrames;
    for (auto it : outSet) {
      auto cfInfo = cfInfos[it->id()];
      if (!cfInfo.frame_name.empty()) { unsupportedFrames.insert(cfInfo.frame_name); }
      while (!cfInfos[cfInfo.parent_frame->id()].frame_name.empty()) {
        unsupportedFrames.insert(cfInfos[cfInfo.parent_frame->id()].frame_name);
        cfInfo = cfInfos[cfInfo.parent_frame->id()];
      }
    }
    for (auto it = candidates->begin(); it != candidates->end();) {
      auto cfInfo = cfInfos[(*it)->id()];
      if (unsupportedFrames.find(cfInfo.frame_name) != unsupportedFrames.end()) {
        outSet.insert(*it);
        it = candidates->erase(it);
      } else {
        ++it;
      }
    }
  }

  // Reference edge: The reference input/output of the sinking node does not sink
  while (!outSet.empty()) {
    auto iter = outSet.begin();
    auto node = *iter;
    if (mix_compile_mode && (node->type_string() == "Where")) {
      bool isInitializedGraph = InferShapeUtil::IsInitializedGraph(node);
      if (isInitializedGraph) { candidates->insert(node); }
    }

    outSet.erase(iter);
    for (auto edge : node->out_edges()) {
      REQUIRES_NOT_NULL(edge);
      REQUIRES_NOT_NULL(edge->dst());
      if (!edge->IsControlEdge()) {
        DataType dtypeDst = edge->dst()->input_type(edge->dst_input());
        if (IsRefType(dtypeDst) && candidates->count(edge->dst()) > 0) {
          candidates->erase(edge->dst());
          outSet.insert(edge->dst());
          ADP_LOG(INFO) << "Remove node : " << edge->dst()->name() << " from candidates, because of node : "
                        << node->name() << " REF input.";
          continue;
        }
        if (dtypeDst == DT_STRING || dtypeDst == DT_RESOURCE) {
          if (edge->dst()->type_string() == "Assert") { continue; }
          if (node->type_string() == "Const") { continue; }
          if (candidates->erase(edge->dst()) > 0) { outSet.insert(edge->dst()); }
        }
      }
    }
    for (auto edge : node->in_edges()) {
      REQUIRES_NOT_NULL(edge);
      REQUIRES_NOT_NULL(edge->src());
      REQUIRES_NOT_NULL(edge->dst());
      if (!edge->IsControlEdge()) {
        DataType dtypeDst = edge->dst()->input_type(edge->dst_input());
        if (IsRefType(dtypeDst) && candidates->count(edge->src()) > 0) {
          candidates->erase(edge->src());
          outSet.insert(edge->src());
          ADP_LOG(INFO) << "Remove node : " << edge->dst()->name() << " from candidates, because of node : "
                        << node->name() << " REF Output.";
          continue;
        }
        if (dtypeDst == DT_STRING || dtypeDst == DT_RESOURCE) {
          if (candidates->erase(edge->src()) > 0) { outSet.insert(edge->src()); }
        }
      }
    }
  }
  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "TFadapter find Npu support candidates cost: [" << ((endTime - startTime) / kMicrosToMillis)
                << " ms]";
  return Status::OK();
}

bool NodeIsCandidateForClustering(Node *node, OrderedNodeSet *candidates) { return candidates->count(node) > 0; }

Status AddRelationalConst(const Graph &graph, OrderedNodeSet *candidates) {
  for (Node *node : graph.op_nodes()) {
    if (node->type_string() == "Const") {
      for (auto edge : node->out_edges()) {
        REQUIRES_NOT_NULL(edge);
        REQUIRES_NOT_NULL(edge->dst());
        if (NodeIsCandidateForClustering(edge->dst(), candidates)) {
          candidates->insert(node);
          break;
        }
      }
    }
  }
  return Status::OK();
}

bool GetNodeFuncs(const FunctionLibraryDefinition *flib_def, Node *node, std::vector<string> &nodeFuncs) {
  nodeFuncs.clear();
  for (const auto &iter : node->attrs()) {
    if (iter.second.has_func()) {
      nodeFuncs.push_back(iter.second.func().name());
      std::vector<string> funcNameStack;
      funcNameStack.clear();
      funcNameStack.push_back(iter.second.func().name());
      while (!funcNameStack.empty()) {
        string funcName = funcNameStack.back();
        funcNameStack.pop_back();
        const FunctionDef *fdef = flib_def->Find(funcName);
        if (fdef != nullptr) {
          for (const auto &ndef : fdef->node_def()) {
            for (const auto &item : ndef.attr()) {
              if (item.second.has_func()) {
                nodeFuncs.push_back(item.second.func().name());
                funcNameStack.push_back(item.second.func().name());
                continue;
              }
            }
          }
        }
      }
      continue;
    }
  }

  return !nodeFuncs.empty();
}

struct Cluster {
  int index = 0;
  std::set<Node *> nodes;
  std::set<std::string> start_nodes_name;
};

// Merges src and dst clusters of the edge
void MergeClusters(Edge *edge, std::map<Node *, std::shared_ptr<Cluster>> &cluster_map) {
  Node *src = edge->src();
  Node *dst = edge->dst();

  // Merge dst cluster into src cluster
  auto cluster_dst = cluster_map[dst];
  for (const auto &start_node_name : cluster_dst->start_nodes_name) {
    cluster_map[src]->start_nodes_name.insert(start_node_name);
  }
  for (auto node : cluster_dst->nodes) {
    cluster_map[src]->nodes.insert(node);
    cluster_map[node] = cluster_map[src];
  }
}

Status MergeSubgraphsInNewWay(std::vector<std::pair<string, int>> &sortedCluster, OrderedNodeSet &npuSupportCandidates,
                              std::map<string, std::set<string>> &clusterToMerge) {
  int64 startTime = InferShapeUtil::GetCurrentTimestap();
  if (sortedCluster.size() < MIN_CLUSTER_SIZE) { return Status::OK(); }
  // record already merged cluster
  std::set<string> mergedClusters;
  // record every cluster merge to which cluster : first now, second dst
  std::map<string, string> mergePair;
  uint32_t clusterindex = 0;
  while (mergedClusters.size() < sortedCluster.size()) {
    string dstSubgraph = sortedCluster[clusterindex].first;
    if (mergedClusters.count(dstSubgraph) < 1) {
      mergedClusters.insert(dstSubgraph);
      for (const string &toMerge : clusterToMerge[dstSubgraph]) {
        if (clusterToMerge[toMerge].count(dstSubgraph) > 0) {
          mergedClusters.insert(toMerge);
          mergePair[toMerge] = dstSubgraph;
        }
      }
    }
    clusterindex++;
    if (clusterindex >= sortedCluster.size()) { break; }
  }

  for (Node *n : npuSupportCandidates) {
    string name;
    Status s = GetNodeAttr(n->attrs(), PARTITION_SUB_GRAPH_ATTR, &name);
    if (s.code() == error::Code::NOT_FOUND) {
      continue;
    } else if (!s.ok()) {
      return s;
    }
    if (mergePair.count(name) > 0) {
      n->ClearAttr(PARTITION_SUB_GRAPH_ATTR);
      n->AddAttr(PARTITION_SUB_GRAPH_ATTR, mergePair[name]);
    }
  }
  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "TFadapter merge clusters cost: [" << ((endTime - startTime) / kMicrosToMillis) << " ms]";
  return Status::OK();
}

Status MergeSubgraphs(std::vector<std::pair<string, int>> &sortedCluster, OrderedNodeSet &npuSupportCandidates,
                      std::map<string, std::set<string>> &clusterToMerge) {
  int64 startTime = InferShapeUtil::GetCurrentTimestap();
  std::set<string> mergedClusters;
  if (sortedCluster.size() <= 1) { return Status::OK(); }
  string dstSubgraph = sortedCluster[0].first;
  mergedClusters.insert(dstSubgraph);
  for (uint32_t i = 1; i < sortedCluster.size(); i++) {
    bool canMerge = false;
    string name = sortedCluster[i].first;
    if (clusterToMerge[dstSubgraph].count(name) > 0 && clusterToMerge[name].count(dstSubgraph) > 0) {
      canMerge = true;
      for (const string &mergedName : mergedClusters) {
        // Mutual exclusion between merged groups
        if (clusterToMerge[mergedName].count(name) == 0 || clusterToMerge[name].count(mergedName) == 0) {
          canMerge = false;
          break;
        }
      }
    }
    if (canMerge && mergedClusters.count(name) == 0) { mergedClusters.insert(name); }
  }

  for (Node *n : npuSupportCandidates) {
    string name;
    Status s = GetNodeAttr(n->attrs(), PARTITION_SUB_GRAPH_ATTR, &name);
    if (s.code() == error::Code::NOT_FOUND) {
      continue;
    } else if (!s.ok()) {
      return s;
    }
    if (name == dstSubgraph) { continue; }
    if (mergedClusters.count(name) != 0) {
      n->ClearAttr(PARTITION_SUB_GRAPH_ATTR);
      n->AddAttr(PARTITION_SUB_GRAPH_ATTR, dstSubgraph);
    } else if (name != dstSubgraph) {
      n->ClearAttr(PARTITION_SUB_GRAPH_ATTR);
    }
  }
  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "TFadapter merge clusters cost: [" << ((endTime - startTime) / kMicrosToMillis) << " ms]";
  return Status::OK();
}

std::vector<string> string_split(const string &str, const string &pattern) {
  std::vector<string> resultVec;
  string::size_type pos1 = 0;
  string::size_type pos2 = str.find(pattern);
  while (pos2 != string::npos) {
    resultVec.push_back(str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + pattern.size();
    pos2 = str.find(pattern, pos1);
  }
  if (pos1 != str.length()) { resultVec.push_back(str.substr(pos1)); }
  return resultVec;
}

Status MarkForPartition(std::unique_ptr<Graph> *graphIn, int &clusterNum, bool mix_compile_mode, int graph_num,
                        FunctionLibraryDefinition *func_lib, std::map<std::string, std::string> pass_options,
                        std::map<std::string, std::string> &graph_options) {
  Graph *graph = graphIn->get();
  bool enable_dp = pass_options["enable_dp"] == "1";
  bool is_set_lazy_recompile = graph_options["dynamic_input"] == "1" &&
                               graph_options["dynamic_graph_execute_mode"] == "lazy_recompile";
  OrderedNodeSet npuSupportCandidates;
  if (!pass_options["in_out_pair"].empty()) {
    if (!mix_compile_mode) {
      return errors::Internal("mix_compile_mode must be True when use in_out_pair.");
    }
    TF_RETURN_IF_ERROR(FindCandidatesByInOutPair(*graph, &npuSupportCandidates, func_lib,
                                                 pass_options["in_out_pair"], pass_options["in_out_pair_flag"]));
  }
  if (npuSupportCandidates.empty()) {
    TF_RETURN_IF_ERROR(FindNpuSupportCandidates(*graph, &npuSupportCandidates, func_lib, enable_dp, mix_compile_mode));
    TF_RETURN_IF_ERROR(AddRelationalConst(*graph, &npuSupportCandidates));
  }

  std::map<Node *, std::shared_ptr<Cluster>> cluster_map;
  tensorflow::GraphCycles cycles;
  std::string job = pass_options["job"];

  // Initial Step: Each node is a cluster of its own
  for (auto node : graph->nodes()) {
    int new_index = cycles.NewNode();
    try {
      cluster_map[node] = std::make_shared<Cluster>();
    } catch (...) { return errors::Internal("make shared failed"); }
    cluster_map[node]->index = new_index;
    cluster_map[node]->nodes.insert(node);

    auto node_attr_value = node->attrs().Find("_StartNodeName");
    if (node_attr_value != nullptr) {
      std::vector<std::string> startNodeVec = string_split(node_attr_value->s(), ";");
      for (const auto &startNodeName : startNodeVec) { cluster_map[node]->start_nodes_name.insert(startNodeName); }
    }
  }
  // Check for existing cyclicity in the graph
  std::vector<Edge *> cycleEdges;
  for (auto edge : graph->edges()) {
    REQUIRES_NOT_NULL(edge);
    Node *src = edge->src();
    Node *dst = edge->dst();
    REQUIRES_NOT_NULL(src);
    REQUIRES_NOT_NULL(dst);
    // Skip source/sink
    if (!src->IsOp() || !dst->IsOp()) { continue; }
    // Skip NextIteration
    if (src->IsNextIteration()) {
      cycleEdges.push_back(edge);
      continue;
    }
    if (!cycles.InsertEdge(cluster_map[src]->index, cluster_map[dst]->index)) {
      ADP_LOG(ERROR) << "Failing due to cycle";
      LOG(ERROR) << "Failing due to cycle";
      return errors::Unimplemented("Input graph has a cycle (inserting an edge from ", src->DebugString(), " to ",
                                   dst->DebugString(), " would create a cycle)");
    }
  }

  for (auto edge : cycleEdges) {
    REQUIRES_NOT_NULL(edge);
    Node *src = edge->src();
    Node *dst = edge->dst();
    REQUIRES_NOT_NULL(src);
    REQUIRES_NOT_NULL(dst);
    cycles.InsertEdge(cluster_map[src]->index, cluster_map[dst]->index, true);
  }

  bool changed = false;
  do {
    changed = false;
    for (auto edge : graph->edges()) {
      REQUIRES_NOT_NULL(edge);
      Node *src = edge->src();
      Node *dst = edge->dst();
      REQUIRES_NOT_NULL(src);
      REQUIRES_NOT_NULL(dst);
      if (!src->IsOp() || !dst->IsOp()) { continue; }

      int src_index = cluster_map[src]->index;
      int dst_index = cluster_map[dst]->index;

      if (!NodeIsCandidateForClustering(src, &npuSupportCandidates)
          || !NodeIsCandidateForClustering(dst, &npuSupportCandidates)) {
        continue;
      }

      if (is_set_lazy_recompile && src->type_string() == "IteratorGetNext" && enable_dp) {
        graph_options["is_dynamic_getnext"] = "1";
        continue;
      }

      // Check if contracting the edge will lead to cycles
      // if not, MergeClusters
      if (cycles.HasEdge(src_index, dst_index) && cycles.ContractEdge(src_index, dst_index)) {
        if (job != "localhost") {
          bool find_same_start = false;
          auto cluster_src = cluster_map[src];
          auto cluster_dst = cluster_map[dst];
          for (const auto &src_start_name : cluster_src->start_nodes_name) {
            for (const auto &dst_start_name : cluster_dst->start_nodes_name) {
              if (src_start_name == dst_start_name) {
                find_same_start = true;
                ADP_LOG(INFO) << "node : " << src->name() << " and node : " << dst->name()
                              << " has same start node : " << src_start_name;
                break;
              }
            }
            if (find_same_start) { break; }
          }
          if (find_same_start) { continue; }
        }
        MergeClusters(edge, cluster_map);
        changed = true;
      }
    }
  } while (changed);

  int64 clusterSequenceNum = 0;
  std::map<int, std::pair<string, int>> clusterInfo;
  std::map<string, std::set<string>> clusterToMerge;
  std::map<int, std::set<int>> clusterIndexToMerge;
  std::set<Cluster *> seen;
  std::set<int> clusterSet;

  for (const auto &item : cluster_map) {
    auto cluster = item.second.get();
    if (seen.count(cluster) != 0) { continue; }

    bool hasSupportNode = false;
    bool hasNonSupportNode = false;

    for (auto node : cluster->nodes) {
      if (NodeIsCandidateForClustering(node, &npuSupportCandidates)) {
        hasSupportNode = true;
      } else {
        hasNonSupportNode = true;
      }
    }

    if (hasSupportNode && hasNonSupportNode) {
      ADP_LOG(INFO) << "Cluster " << cluster->index << " has both Candidate and non-Candidate nodes";
      return errors::Internal("Cluster ", cluster->index, " has both Candidate and non-Candidate nodes");
    }

    if (!hasSupportNode) {
      seen.insert(cluster);
      continue;
    }

    clusterSet.insert(cluster->index);
    string op_prefix = "GeOp";

    string name = strings::StrCat(string(op_prefix), std::to_string(graph_num), string("_"),
                                  std::to_string(clusterSequenceNum++));
    clusterInfo[cluster->index] = std::make_pair(name, cluster->nodes.size());
    for (auto node : cluster->nodes) {
      if (!NodeIsCandidateForClustering(node, &npuSupportCandidates)) {
        // attr PARTITION_SUB_GRAPH_ATTR delete later
        clusterInfo.erase(cluster->index);
        return errors::Internal("Node ", node->DebugString(),
                                " was not marked for clustering but was "
                                "placed in a cluster.");
      }
      node->AddAttr(PARTITION_SUB_GRAPH_ATTR, name);
    }
    seen.insert(cluster);
  }
  // Generate Merge possibility between clusters
  if (clusterSet.size() > 1) {
    for (int src : clusterSet) {
      for (int dst : clusterSet) {
        if (src == dst) { continue; }
        if (!cycles.IsReachableNonConst(src, dst) && !cycles.IsReachableNonConst(dst, src)) {
          if (mix_compile_mode) {
            bool canReach = false;
            for (auto cluster : clusterIndexToMerge[src]) {
              if (cycles.IsReachableNonConst(dst, cluster) || cycles.IsReachableNonConst(cluster, dst)) {
                canReach = true;
                break;
              }
            }

            if (!canReach && job != "localhost") {
              Cluster *cluster_src = nullptr;
              Cluster *cluster_dst = nullptr;
              for (const auto &cluster_item : cluster_map) {
                if (cluster_src == nullptr && src == cluster_item.second->index) {
                  cluster_src = cluster_item.second.get();
                }
                if (cluster_dst == nullptr && dst == cluster_item.second->index) {
                  cluster_dst = cluster_item.second.get();
                }
                if (cluster_src != nullptr && cluster_dst != nullptr) { break; }
              }
              bool find_same_start = false;
              for (const auto &src_start_name : cluster_src->start_nodes_name) {
                for (const auto &dst_start_name : cluster_dst->start_nodes_name) {
                  if (src_start_name == dst_start_name) {
                    find_same_start = true;
                    break;
                  }
                }
                if (find_same_start) { break; }
              }
              if (find_same_start) { canReach = true; }
            }
            if (!canReach) {
              clusterToMerge[clusterInfo[src].first].insert(clusterInfo[dst].first);
              clusterIndexToMerge[src].insert(dst);
            }
          } else {
            clusterToMerge[clusterInfo[src].first].insert(clusterInfo[dst].first);
          }
        }
      }
    }
  }

  struct ClusterCompare {
    bool operator()(const std::pair<string, int> &a, const std::pair<string, int> &b) const {
      return a.second > b.second;
    }
  };

  std::vector<std::pair<string, int>> sortedCluster;
  for (const auto &cluster : clusterInfo) { sortedCluster.push_back(cluster.second); }
  std::sort(sortedCluster.begin(), sortedCluster.end(), ClusterCompare());
  clusterNum = clusterSequenceNum;
  if (static_cast<int>(sortedCluster.size()) != clusterNum) {
    return errors::Internal("Sorted cluster size should be equal to origin subgraph num. ", "Sorted cluster size is ",
                            sortedCluster.size(), ", origin subgraph num is ", clusterNum);
  }
  ADP_LOG(INFO) << "cluster Num is " << clusterNum;
  if (clusterNum == 0) { return Status::OK(); }

  int minGroupSizeTemp = 1;
  int minGroupSize =
      (((minGroupSizeTemp > 0) && (minGroupSizeTemp < MAX_GROUP_SIZE)) ? (minGroupSizeTemp)
                                                                       : (1));  // default threshold is 10.
  ADP_LOG(INFO) << "All nodes in graph: " << graph->num_nodes() << ", max nodes count: " << sortedCluster[0].second
            << " in subgraph: " << sortedCluster[0].first << " minGroupSize: " << minGroupSize;

  bool isDateSetCluster = false;
  bool isBroadcastGraph = false;
  for (Node *n : npuSupportCandidates) {
    if (n->type_string().find("MakeIterator") != string::npos) {
      isDateSetCluster = true;
      break;
    }
    if (n->type_string().find("HcomBroadcast") != string::npos) {
      isBroadcastGraph = true;
      break;
    }
  }
  if (sortedCluster[0].second >= minGroupSize || isDateSetCluster || isBroadcastGraph) {
    if (sortedCluster[0].second == 1) {
      for (Node *n : npuSupportCandidates) {
        if (n->type_string() == "NoOp" || n->type_string() == "Identity") {
          string name;
          Status s = GetNodeAttr(n->attrs(), PARTITION_SUB_GRAPH_ATTR, &name);
          if (s.code() == error::Code::NOT_FOUND) {
            continue;
          } else if (!s.ok()) {
            return s;
          }
          n->ClearAttr(PARTITION_SUB_GRAPH_ATTR);
          ADP_LOG(INFO) << "Clear isolated NoOp from " << name;
          clusterNum -= 1;
        }
      }
    }
    if (clusterNum > 1) {
      if (mix_compile_mode || is_set_lazy_recompile) {
        TF_RETURN_IF_ERROR(MergeSubgraphsInNewWay(sortedCluster, npuSupportCandidates, clusterToMerge));
      } else {
        TF_RETURN_IF_ERROR(MergeSubgraphs(sortedCluster, npuSupportCandidates, clusterToMerge));
        clusterNum = 1;
      }
    }
  } else {
    ADP_LOG(INFO) << "Clear all node PARTITION_SUB_GRAPH_ATTR attr.";
    for (Node *n : npuSupportCandidates) { n->ClearAttr(PARTITION_SUB_GRAPH_ATTR); }
    clusterNum = 0;
  }

  return Status::OK();
}

// A node/slot pair.
struct NodeSlot {
  NodeSlot() : node(nullptr), slot(-1), dtype(DT_INVALID) {}
  NodeSlot(const Node *node, int slot) : node(node), slot(slot), dtype(DT_INVALID) {}
  NodeSlot(const Node *node, int slot, DataType dtype) : node(node), slot(slot), dtype(dtype) {}

  const Node *node;
  int slot;

  // Optional: used to record the destination type of a source NodeSlot in case
  // the source output is a Ref type that is cast to a Tensor at the
  // destination.
  DataType dtype;

  bool operator==(const NodeSlot &other) const {
    return node == other.node && slot == other.slot && dtype == other.dtype;
  }

  // Leave dtype out of the hash since there are never two NodeSlots with the
  // same node and slot and different dtypes.
  struct Hasher {
    uint64 operator()(NodeSlot const &s) const {
      return Hash64Combine(std::hash<const Node *>()(s.node), std::hash<int>()(s.slot));
    }
  };

  struct PairHasher {
    uint64 operator()(std::pair<NodeSlot, NodeSlot> const &s) const {
      return Hash64Combine(Hasher()(s.first), Hasher()(s.second));
    }
  };
};

Node *AddIdentityNode(Graph *graph, const Edge *edge, const string &srcName, int srcIndex, const string &device,
                      Status *status) {
  // edge is not nullptr
  if (edge->src() == nullptr) { return nullptr; }
  NodeDef identityDef;
  NodeDefBuilder builder(strings::StrCat(edge->src()->name(), "_dummyIdentity"), "Identity");
  DataType dtype = BaseType(edge->src()->output_type(edge->src_output()));
  builder.Attr("T", dtype);
  builder.Input(srcName, srcIndex, dtype);
  builder.Device(device);
  Status s = builder.Finalize(&identityDef);
  if (!s.ok()) {
    status->Update(s);
    return nullptr;
  }
  Node *identityNode = graph->AddNode(identityDef, &s);
  if (!s.ok() || identityNode == nullptr) {
    status->Update(s);
    return nullptr;
  }
  identityNode->set_assigned_device_name(device);

  return identityNode;
}

class OMSplitter {
 public:
  OMSplitter(string groupAttribute, Graph const *graphIn, std::map<std::string, std::string> npu_optimizer_options,
             std::map<std::string, std::string> pass_options, std::map<std::string, std::string> graph_options)
      : groupAttribute_(std::move(groupAttribute)), graphIn_(graphIn),
        npu_optimizer_options_(std::move(npu_optimizer_options)), pass_options_(std::move(pass_options)),
        graph_options_(std::move(graph_options)) {}

  ~OMSplitter() = default;
  // Find subgraphs marked with 'groupAttribute', and build a new
  // subgraph, one for each value of 'groupAttribute'.
  // 'subgraphNum' indicate how many subgraphs has been built.
  Status SplitIntoSubgraphs(uint32_t &subgraphNum);

  // Build a FunctionDef for each subgraph, and add it 'library'. The values of
  // the 'groupAttribute' annotations become the function names.
  Status BuildFunctionDefs(FunctionLibraryDefinition *library, const string &graph_format);

  // Write a copy of the input graph to 'graphOut', where the subgraphs are
  // replaced with GEOPs to the new functions.
  Status BuildOutputGraph(Graph *graphOut);

 private:
  // A subgraph of the input, all marked with a common 'groupAttribute'
  // value.
  //
  // In the following simple example, A, B, ..., E are nodes in the original
  // graph. The group attributes  g are
  // each shown as either 0 or empty.
  //
  //  A  -->  B  -->  C  -->  D  -->  E
  //  g:      g:0     g:0     g:0     g:
  //
  // The example is rewritten to one graph;
  //
  //  A  -->  GEOp_0  -->  E
  //
  // The GEOp is as follows.
  //
  //  Arg  --> B  --> C  --> D --> Retval
  //
  class Subgraph {
   public:
    // Creates a graph to build the subgraph in, if it doesn't already exist,
    // using the same op registry and versions as graphIn.
    Subgraph() : GEOpNodeInputs_(nullptr), GEOpNodeOutputs_(nullptr) {}

    ~Subgraph() = default;

    Node *MakeNodeImage(const Graph *graphIn, Node *node);

    // Returns the graph the subgraph is being built in.
    Graph *GetGraph() const;

    // Builds a FunctionDef, and adds it to 'library'. The value of the
    // 'groupAttribute' annotations becomes the function name.
    Status BuildFunctionDef(const string &name, FunctionLibraryDefinition *library, const string &graph_format);

    // Adds the GEOp node to graphOut.
    Status AddGEOpNode(const std::unordered_map<const Node *, Node *> &nodeImages, Graph *graphOut);

    // Returns the Node that inputs to the GEOp should be wired up to.
    Node *GetGEOpNodeForInputs() const;

    // Returns the Node that outputs to the GEOp should be wired up to.
    Node *GetGEOpNodeForOutputs() const;

    // Returns the index of the arg that the dst of edge should connect to.
    int GetArgIndexForEdge(const Edge *edge) const;

    // Returns the index of the result that the src of edge should connect to.
    int GetResultIndexForEdge(const Edge *edge) const;

    // Creates an _Arg node for the src node of edge, and add its index to
    // argsBySrc_, if none exists yet. Also adds its index to argsByDst_,
    // and adds the edge within the subgraph from the _Arg node to the image of
    // the dst node.
    Status RecordArg(const Edge *edge, const std::unordered_map<const Node *, Node *> &nodeImages,
                     std::vector<std::pair<const Node *, Node *>> *srcArgPairs);

    // Creates a _Retval node for the src node of edge, and add it to results_,
    // if none exists yet. If a new _Retval node is created, also adds the edge
    // within the subgraph from the src to the _Retval node.
    Status RecordResult(const Edge *edge, const std::unordered_map<const Node *, Node *> &nodeImages);

    // Indicates if the subgraph does not have any input or output
    bool isIsolatedSubgraph();

    Status SetOptions(std::map<std::string, std::string> npu_optimizer_options,
                      std::map<std::string, std::string> pass_options,
                      std::map<std::string, std::string> graph_options);

    // GEOp node(s) in the output graph. Not owned.
    // both point to the function call node.
    Node *GEOpNodeInputs_;
    Node *GEOpNodeOutputs_;

   private:
    // The subgraph extracted from the input graph, suitable for being turned
    // into a FunctionDef. Inputs are fed by _Arg nodes, and outputs are
    // returned by _Retval nodes.
    std::unique_ptr<Graph> graph_;

    // Which device are these nodes on
    string device_;

    // NodeDef for the GEOp node.
    NodeDef GEOpNodeDef_;

    // Name that is used for the GEOp node.
    string functionDefName_;

    // Maps from source (producer node/slot) and destination
    // (consumer node/slot) tensors in the input graph to _Arg numbers in
    // the subgraph.
    std::unordered_map<NodeSlot, int, NodeSlot::Hasher> argsBySrc_;
    std::unordered_map<NodeSlot, int, NodeSlot::Hasher> argsByDst_;

    // The _Arg nodes in the subgraph, in order by argument number.
    std::vector<Node *> args_;

    // Map from source tensor in the input graph to result #.
    std::unordered_map<NodeSlot, int, NodeSlot::Hasher> results_;

    DataTypeVector argDatetypes_;
    DataTypeVector resultDatetypes_;

    std::map<std::string, std::string> npu_optimizer_options_;
    std::map<std::string, std::string> pass_options_;
    std::map<std::string, std::string> graph_options_;
  };

  // Returns the key attribute  associated with a node in attr, Sets
  // either result to the empty string if the respective attribute is not
  // found.
  Status GetSubgraphIdAttr(Node const *node, string *attr) const;

  // Copies edges local to a subgraph. Adds _Arg and _Retval nodes to
  // subgraphs for data edges that cross subgraph boundaries.
  Status CopySubgraphEdges(const std::unordered_map<const Node *, Node *> &nodeImages,
                           std::vector<std::pair<const Node *, Node *>> *srcArgPairs);

  // Copies all marked nodes to a subgraph. Does nothing for unmarked nodes
  Status CopySubgraphNodes(std::unordered_map<const Node *, Node *> *nodeImages);

  // Copies all nodes that aren't in a subgraph to the output graph.
  Status CopyNodesToOutputGraph(Graph *graphOut, std::unordered_map<const Node *, Node *> *nodeImages);

  // Adds GEOp nodes for each subgraph.
  Status AddGEOpNodes(const std::unordered_map<const Node *, Node *> &nodeImages, Graph *graphOut);

  // Finds the image of an edge source in the output graph. If the edge crosses
  // a subgraph boundary it is the output of a GEOp node, otherwise it is a node
  // in the output graph.
  Status FindOutputImageOfEdgeSrc(const string &srcSubgraphId, const string &dstSubgraphId,
                                  const std::unordered_map<const Node *, Node *> &nodeImages,
                                  const Node *originalSrcNode, Node **srcImage);

  // Finds an edge source slot in the output graph. If the edge crosses a
  // subgraph boundary it is a slot on the output of a GEOp node , otherwise
  // it is a slot on a node in the output graph.
  int FindOutputSlotOfEdgeSrc(const string &srcSubgraphId, const string &dstSubgraphId, const Edge *edge);

  // Finds the image of an edge destination in the output graph. If the edge
  // crosses a subgraph boundary it is the input of a GEOp node , otherwise
  // it is a node in the output graph.
  Status FindOutputImageOfEdgeDst(const string &srcSubgraphId, const string &dstSubgraphId,
                                  const std::unordered_map<const Node *, Node *> &nodeImages,
                                  const Node *originalDstNode, Node **dstImage);

  // Finds an edge destination slot in the output graph. If the edge crosses a
  // subgraph boundary it is a slot on the input of a GEOp node, otherwise
  // it is a slot on a node in the output graph.
  int FindOutputSlotOfEdgeDst(const string &srcSubgraphId, const string &dstSubgraphId, const Edge *edge);

  // Copies a single edge to the output graph. The edge is either entirely
  // within the output graph, or crosses into or out of a subgraph.
  Status CopyEdgeToOutputGraph(const Edge *edge, const string &srcSubgraphId, const string &dstSubgraphId,
                               const std::unordered_map<const Node *, Node *> &nodeImages, Graph *graphOut,
                               std::unordered_set<std::pair<NodeSlot, NodeSlot>, NodeSlot::PairHasher> *edges_added);

  // Adds all edges to the output graph.
  Status AddEdgesToOutputGraph(const std::unordered_map<const Node *, Node *> &nodeImages, Graph *graphOut);

  const string groupAttribute_;
  const Graph *graphIn_;
  std::vector<const Edge *> refIn_;
  uint64_t ID_NUM = 3;

  std::unordered_map<string, Subgraph> subgraphs_;
  std::map<std::string, std::string> npu_optimizer_options_;
  std::map<std::string, std::string> pass_options_;
  std::map<std::string, std::string> graph_options_;

  TF_DISALLOW_COPY_AND_ASSIGN(OMSplitter);
};

Node *OMSplitter::Subgraph::GetGEOpNodeForInputs() const { return GEOpNodeInputs_; }

Node *OMSplitter::Subgraph::GetGEOpNodeForOutputs() const { return GEOpNodeOutputs_; }

int OMSplitter::Subgraph::GetArgIndexForEdge(const Edge *edge) const {
  return argsByDst_.at(NodeSlot(edge->dst(), edge->dst_input()));
}

int OMSplitter::Subgraph::GetResultIndexForEdge(const Edge *edge) const {
  return results_.at(NodeSlot(edge->src(), edge->src_output()));
}

Node *OMSplitter::Subgraph::MakeNodeImage(const Graph *graphIn, Node *node) {
  if (graph_ == nullptr) {
    graph_.reset(new (std::nothrow) Graph(graphIn->op_registry()));
    if (graph_ == nullptr) {
      ADP_LOG(ERROR) << "graph new failed";
      LOG(ERROR) << "graph new failed";
      return nullptr;
    }
    graph_->set_versions(graphIn->versions());
  }

  std::string job = pass_options_["job"];
  int task_index = std::atoi(pass_options_["task_index"].c_str());

  if (device_.empty()) {
    if (job != "localhost" && job != "ps" && job != "default") {
      string device_name = std::string("/job:") + std::string(job) + std::string("/replica:0/task:")
          + std::to_string(task_index) + std::string("/device:CPU:0");
      device_ = device_name;
    } else if (job == "localhost") {
      device_ = string("/job:localhost/replica:0/task:0/device:CPU:0");
    } else {
      ADP_LOG(ERROR) << "job type is : " << job << " not support. ";
      LOG(ERROR) << "job type is : " << job << " not support. ";
      return nullptr;
    }
  }
  Node *nodeOut = graph_->CopyNode(node);
  if (nodeOut == nullptr) {
    ADP_LOG(ERROR) << "copy node failed";
    LOG(ERROR) << "copy node failed";
    return nullptr;
  }
  nodeOut->set_assigned_device_name(device_);
  return nodeOut;
}

Graph *OMSplitter::Subgraph::GetGraph() const { return graph_.get(); }

Status OMSplitter::Subgraph::RecordArg(const Edge *edge, const std::unordered_map<const Node *, Node *> &nodeImages,
                                       std::vector<std::pair<const Node *, Node *>> *srcArgPairs) {
  Node *srcNode = edge->src();
  int srcSlot = edge->src_output();
  std::unordered_map<NodeSlot, int, NodeSlot::Hasher>::iterator iter;
  bool inserted = false;
  std::tie(iter, inserted) = argsBySrc_.emplace(NodeSlot(srcNode, srcSlot), argsBySrc_.size());
  int argIndex = iter->second;
  if (inserted) {
    NodeDef argNodeDef;
    NodeDefBuilder builder(strings::StrCat(srcNode->name(), "_", srcSlot, "_arg"), ARG_OP);
    DataType dtype = edge->dst()->input_type(edge->dst_input());
    builder.Attr("T", dtype);
    builder.Attr("index", argIndex);
    Status s = builder.Finalize(&argNodeDef);
    if (!s.ok()) { return s; }

    Node *arg = graph_->AddNode(argNodeDef, &s);
    if (!s.ok()) { return s; }

    srcArgPairs->push_back({srcNode, arg});
    args_.push_back(arg);
    argDatetypes_.push_back(dtype);
  }
  if (srcNode->name().find("_arg_") != std::string::npos) {
    graph_options_["placeholder_index"] += std::to_string(argIndex);
  }
  Node *dstNode = edge->dst();
  Node *dstImage = nodeImages.at(dstNode);
  int dstSlot = edge->dst_input();
  argsByDst_[NodeSlot(dstNode, dstSlot)] = argIndex;
  graph_->AddEdge(args_[argIndex], 0, dstImage, dstSlot);
  return Status::OK();
}

Status OMSplitter::Subgraph::RecordResult(const Edge *edge,
                                          const std::unordered_map<const Node *, Node *> &nodeImages) {
  Node *srcNode = edge->src();
  Node *srcImage = nodeImages.at(srcNode);
  REQUIRES_NOT_NULL(srcImage);
  int srcSlot = edge->src_output();
  std::unordered_map<NodeSlot, int, NodeSlot::Hasher>::iterator iter;
  bool inserted = false;
  std::tie(iter, inserted) = results_.emplace(NodeSlot(srcNode, srcSlot), results_.size());
  int retIndex = iter->second;
  if (inserted) {
    NodeDef retNodeDef;
    NodeDefBuilder builder(strings::StrCat(srcNode->name(), "_", srcSlot, "_retval"), RET_OP);
    DataType dtype = BaseType(srcNode->output_type(srcSlot));
    builder.Attr("T", dtype);
    builder.Attr("index", retIndex);
    builder.Input(srcImage->name(), srcSlot, dtype);
    Status s = builder.Finalize(&retNodeDef);
    if (!s.ok()) { return s; }
    Node *ret = graph_->AddNode(retNodeDef, &s);
    if (!s.ok()) { return s; }
    resultDatetypes_.push_back(dtype);
    // src --> dst has ref input/output, add identity node: src --> identity --> dst
    if (IsRefType(edge->src()->output_type(edge->src_output()))
        || IsRefType(edge->dst()->input_type(edge->dst_input()))) {
      Status addStatus;
      Node *identityNode =
          AddIdentityNode(graph_.get(), edge, srcImage->name(), srcSlot, srcImage->assigned_device_name(), &addStatus);
      if (!addStatus.ok()) { return addStatus; }
      graph_->AddEdge(srcImage, srcSlot, identityNode, 0);
      graph_->AddEdge(identityNode, 0, ret, 0);
    } else {
      graph_->AddEdge(srcImage, srcSlot, ret, 0);
    }
  }
  return Status::OK();
}

Status OMSplitter::Subgraph::BuildFunctionDef(const string &name, FunctionLibraryDefinition *library,
                                              const string &graph_format) {
  GEOpNodeDef_.set_op("GeOp");

  GEOpNodeDef_.set_name(name);
  GEOpNodeDef_.set_device(device_);

  for (auto node : graph_->nodes()) {
    std::vector<string> nodeFuncs;
    if (GetNodeFuncs(library, node, nodeFuncs)) {
      std::unique_ptr<FunctionDefLibrary> func_def_lib(new (std::nothrow) FunctionDefLibrary());
      REQUIRES_NOT_NULL(func_def_lib);
      ADP_LOG(INFO) << "Node [" << node->name() << "] has funcs:";
      for (const auto &func : nodeFuncs) {
        ADP_LOG(INFO) << func;
        FunctionDef *fdef = func_def_lib->add_function();
        REQUIRES_NOT_NULL(fdef);
        REQUIRES_NOT_NULL(library->Find(func));
        *fdef = *(library->Find(func));
      }
      string funcdefStr;
      func_def_lib->SerializeToString(&funcdefStr);
      node->AddAttr(ATTR_NAME_FRAMEWORK_FUNC_DEF, funcdefStr);
    }
  }

  functionDefName_ = name;
  FunctionDef fdef;
  TF_RETURN_IF_ERROR(OMSubGraphToFunctionDef(*graph_, name, &fdef));

  NameAttrList function;
  function.set_name(functionDefName_);
  *function.mutable_attr() = GEOpNodeDef_.attr();
  AddNodeAttr("function", function, &GEOpNodeDef_);
  AddNodeAttr("Tin", argDatetypes_, &GEOpNodeDef_);
  AddNodeAttr("Tout", resultDatetypes_, &GEOpNodeDef_);
  AddNodeAttr("data_format", graph_format, &GEOpNodeDef_);

  std::string attr_name;
  for (const auto &option : npu_optimizer_options_) {
    attr_name = std::string("_") + option.first;
    AddNodeAttr(attr_name, option.second, &GEOpNodeDef_);
  }
  for (const auto &option : graph_options_) {
    attr_name = std::string("_") + option.first;
    AddNodeAttr(attr_name, option.second, &GEOpNodeDef_);
  }
  AddNodeAttr("_NpuOptimizer", "NpuOptimizer", &GEOpNodeDef_);

  if (library->Find(name) == nullptr) { TF_RETURN_IF_ERROR(library->AddFunctionDef(fdef)); }
  return Status::OK();
}

Status OMSplitter::Subgraph::AddGEOpNode(const std::unordered_map<const Node *, Node *> &nodeImages, Graph *graphOut) {
  Status s;
  GEOpNodeInputs_ = graphOut->AddNode(GEOpNodeDef_, &s);
  if (!s.ok()) { return s; }

  // Copy the assigned device and the key_annotation over.
  REQUIRES_NOT_NULL(GEOpNodeInputs_);
  GEOpNodeInputs_->set_assigned_device_name(device_);
  GEOpNodeOutputs_ = GEOpNodeInputs_;

  return Status::OK();
}

bool OMSplitter::Subgraph::isIsolatedSubgraph() { return false; }

Status OMSplitter::Subgraph::SetOptions(std::map<std::string, std::string> npu_optimizer_options,
                                        std::map<std::string, std::string> pass_options,
                                        std::map<std::string, std::string> graph_options) {
  npu_optimizer_options_ = std::move(npu_optimizer_options);
  pass_options_ = std::move(pass_options);
  graph_options_ = std::move(graph_options);
  return Status::OK();
}

Status OMSplitter::GetSubgraphIdAttr(Node const *node, string *attr) const {
  Status s = GetNodeAttr(node->attrs(), groupAttribute_, attr);
  if (s.code() == error::Code::NOT_FOUND) {
    // Return empty attr if there's no groupAttribute.
    attr->clear();
  } else if (!s.ok()) {
    return s;
  }
  return Status::OK();
}

bool IsInSubgraph(const string &subgraphId) { return !subgraphId.empty(); }

Status OMSplitter::CopySubgraphNodes(std::unordered_map<const Node *, Node *> *nodeImages) {
  for (Node *node : graphIn_->op_nodes()) {
    string subgraphId;
    TF_RETURN_IF_ERROR(GetSubgraphIdAttr(node, &subgraphId));
    if (!IsInSubgraph(subgraphId)) { continue; }

    Subgraph &subgraph = subgraphs_[subgraphId];
    Status s = subgraph.SetOptions(npu_optimizer_options_, pass_options_, graph_options_);
    if (s != Status::OK()) {
      ADP_LOG(INFO) << "Subgraph Id: " << subgraphId << "set npu optimizer error.";
      return s;
    }
    Node *image = subgraph.MakeNodeImage(graphIn_, node);
    REQUIRES_NOT_NULL(image);
    image->ClearAttr(groupAttribute_);
    (*nodeImages)[node] = image;
  }
  return Status::OK();
}

Status OMSplitter::CopySubgraphEdges(const std::unordered_map<const Node *, Node *> &nodeImages,
                                     std::vector<std::pair<const Node *, Node *>> *srcArgPairs) {
  for (const Edge *edge : graphIn_->edges()) {
    REQUIRES_NOT_NULL(edge);
    REQUIRES_NOT_NULL(edge->src());
    REQUIRES_NOT_NULL(edge->dst());
    string srcSubgraphId;
    TF_RETURN_IF_ERROR(GetSubgraphIdAttr(edge->src(), &srcSubgraphId));
    string dstSubgraphId;
    TF_RETURN_IF_ERROR(GetSubgraphIdAttr(edge->dst(), &dstSubgraphId));
    Node *srcImage = gtl::FindWithDefault(nodeImages, edge->src(), nullptr);
    Node *dstImage = gtl::FindWithDefault(nodeImages, edge->dst(), nullptr);
    // Copy edges that are local to a subgraph.
    if (IsInSubgraph(srcSubgraphId) && IsInSubgraph(dstSubgraphId) && srcSubgraphId == dstSubgraphId) {
      Graph *g = subgraphs_[srcSubgraphId].GetGraph();
      REQUIRES_NOT_NULL(g);
      REQUIRES_NOT_NULL(srcImage);
      REQUIRES_NOT_NULL(dstImage);
      if (edge->IsControlEdge()) {
        g->AddControlEdge(srcImage, dstImage);
      } else {
        g->AddEdge(srcImage, edge->src_output(), dstImage, edge->dst_input());
      }
      continue;
    }

    // Record 'src' as an output of its subgraph.
    if (IsInSubgraph(srcSubgraphId)) {
      if (!edge->IsControlEdge()) {
        DataType dtypeDst = edge->dst()->input_type(edge->dst_input());
        if (IsRefType(dtypeDst)) {
          return errors::InvalidArgument("Ref Tensors (e.g., Variables) are not supported as results: "
                                         "tensor ",
                                         edge->src()->name(), ":", edge->src_output(), ", dst is ",
                                         edge->dst()->name());
        }
      }

      Subgraph &srcSubgraph = subgraphs_[srcSubgraphId];

      // Ignore control edges leaving the subgraph. We will lift them onto the
      // enclosing GEOps in BuildOutputGraph().
      if (!edge->IsControlEdge()) { TF_RETURN_IF_ERROR(srcSubgraph.RecordResult(edge, nodeImages)); }
    }

    // Record 'dst' as an input of its subgraph.
    if (IsInSubgraph(dstSubgraphId)) {
      // Look at the type of the destination not the source, since Ref output
      // Tensors can be automatically cast to non-Ref Tensors at the
      // destination.
      if (!edge->IsControlEdge()) {
        DataType dtypeDst = edge->dst()->input_type(edge->dst_input());
        if (IsRefType(dtypeDst)) {
          return errors::InvalidArgument("Ref Tensors (e.g., Variables) are not supported as args: "
                                         "tensor ",
                                         edge->src()->name(), ":", edge->src_output(), ", dst is ",
                                         edge->dst()->name());
        }
      }

      Subgraph &dstSubgraph = subgraphs_[dstSubgraphId];

      // Ignore control edges entering the subgraph. We will lift them onto
      // the enclosing GEOps in BuildOutputGraph().
      if (!edge->IsControlEdge()) {
        if (IsRefType(edge->src()->output_type(edge->src_output()))) { refIn_.push_back(edge); }
        TF_RETURN_IF_ERROR(dstSubgraph.RecordArg(edge, nodeImages, srcArgPairs));
      }
    }
  }
  return Status::OK();
}

Status OMSplitter::SplitIntoSubgraphs(uint32_t &subgraphNum) {
  // Map from input graph nodes to subgraph nodes.
  std::unordered_map<const Node *, Node *> nodeImages;

  // Each entry of srcArgPairs is a pair whose first element is a node in the
  // original graph that has an output edge in the subgraph, and whose second
  // element is the arg node in the subgraph that it sends to. The vector will
  // be filled in below in AddArgs.
  std::vector<std::pair<const Node *, Node *>> srcArgPairs;
  subgraphNum = 0;

  TF_RETURN_IF_ERROR(CopySubgraphNodes(&nodeImages));
  TF_RETURN_IF_ERROR(CopySubgraphEdges(nodeImages, &srcArgPairs));

  std::vector<string> allSubgraphNames;
  for (auto &entry : subgraphs_) { allSubgraphNames.push_back(entry.first); }
  for (string &subgraphName : allSubgraphNames) {
    Subgraph &subgraph = subgraphs_[subgraphName];
    if (subgraph.isIsolatedSubgraph()) {
      ADP_LOG(INFO) << "IsolatedSubgraph: " << subgraphName;
      subgraphs_.erase(subgraphName);
      for (Node *node : graphIn_->op_nodes()) {
        string subgraphId;
        TF_RETURN_IF_ERROR(GetSubgraphIdAttr(node, &subgraphId));
        if (IsInSubgraph(subgraphId) && subgraphId == subgraphName) { node->ClearAttr(groupAttribute_); }
      }
    }
  }

  subgraphNum = subgraphs_.size();
  ADP_LOG(INFO) << "subgraphNum: " << subgraphNum;

  return Status::OK();
}

Status OMSplitter::BuildFunctionDefs(FunctionLibraryDefinition *library, const string &graph_format) {
  for (auto &subgraphEntry : subgraphs_) {
    string name = subgraphEntry.first;
    Subgraph &subgraph = subgraphEntry.second;
    TF_RETURN_IF_ERROR(subgraph.BuildFunctionDef(name, library, graph_format));
  }
  return Status::OK();
}

Status OMSplitter::CopyNodesToOutputGraph(Graph *graphOut, std::unordered_map<const Node *, Node *> *nodeImages) {
  for (Node *node : graphIn_->op_nodes()) {
    string subgraphId;
    TF_RETURN_IF_ERROR(GetSubgraphIdAttr(node, &subgraphId));

    if (IsInSubgraph(subgraphId)) { continue; }
    Node *image = graphOut->CopyNode(node);
    REQUIRES_NOT_NULL(image);
    (*nodeImages)[node] = image;
  }
  (*nodeImages)[graphIn_->source_node()] = graphOut->source_node();
  (*nodeImages)[graphIn_->sink_node()] = graphOut->sink_node();
  return Status::OK();
}

Status OMSplitter::AddGEOpNodes(const std::unordered_map<const Node *, Node *> &nodeImages, Graph *graphOut) {
  for (auto &subgraphEntry : subgraphs_) { TF_RETURN_IF_ERROR(subgraphEntry.second.AddGEOpNode(nodeImages, graphOut)); }
  return Status::OK();
}

Status OMSplitter::FindOutputImageOfEdgeSrc(const string &srcSubgraphId, const string &dstSubgraphId,
                                            const std::unordered_map<const Node *, Node *> &nodeImages,
                                            const Node *originalSrcNode, Node **srcImage) {
  if (IsInSubgraph(srcSubgraphId)) {
    // The edge is from a subgraph to a regular node in the output graph so
    // use the GEOp node output.
    *srcImage = subgraphs_.at(srcSubgraphId).GetGEOpNodeForOutputs();
  } else {
    // The source of the edge is in the output graph so use the node image in
    // the output graph.
    *srcImage = nodeImages.at(originalSrcNode);
  }
  return Status::OK();
}

int OMSplitter::FindOutputSlotOfEdgeSrc(const string &srcSubgraphId, const string &dstSubgraphId, const Edge *edge) {
  if (IsInSubgraph(srcSubgraphId)) {
    const Subgraph &srcSubgraph = subgraphs_.at(srcSubgraphId);
    // 'src' is in a subgraph and 'dst' is a regular node in the output
    // graph. Use the corresponding GEOp output instead.
    return srcSubgraph.GetResultIndexForEdge(edge);
  } else {
    // The source of the edge is in the output graph so use the regular edge
    // slot.
    return edge->src_output();
  }
}

Status OMSplitter::FindOutputImageOfEdgeDst(const string &srcSubgraphId, const string &dstSubgraphId,
                                            const std::unordered_map<const Node *, Node *> &nodeImages,
                                            const Node *originalDstNode, Node **dstImage) {
  if (IsInSubgraph(dstSubgraphId)) {
    // The edge is to a subgraph from a regular node in the output graph so
    // use the GEOp node input.
    *dstImage = subgraphs_.at(dstSubgraphId).GetGEOpNodeForInputs();
  } else {
    // The destination of the edge is in the output graph so use the node image
    // in the output graph.
    *dstImage = nodeImages.at(originalDstNode);
  }
  return Status::OK();
}

int OMSplitter::FindOutputSlotOfEdgeDst(const string &srcSubgraphId, const string &dstSubgraphId, const Edge *edge) {
  if (IsInSubgraph(dstSubgraphId)) {
    const Subgraph &dstSubgraph = subgraphs_.at(dstSubgraphId);
    // 'dst' is in a subgraph and 'src' is a regular node in the output
    // graph. Use the corresponding GEOp input instead.
    return dstSubgraph.GetArgIndexForEdge(edge);
  } else {
    // The destination of the edge is in the output graph so use the regular
    // edge slot.
    return edge->dst_input();
  }
}

Status OMSplitter::CopyEdgeToOutputGraph(
    const Edge *edge, const string &srcSubgraphId, const string &dstSubgraphId,
    const std::unordered_map<const Node *, Node *> &nodeImages, Graph *graphOut,
    std::unordered_set<std::pair<NodeSlot, NodeSlot>, NodeSlot::PairHasher> *edges_added) {
  Node *srcImage = nullptr;
  TF_RETURN_IF_ERROR(FindOutputImageOfEdgeSrc(srcSubgraphId, dstSubgraphId, nodeImages, edge->src(), &srcImage));
  Node *dstImage = nullptr;
  TF_RETURN_IF_ERROR(FindOutputImageOfEdgeDst(srcSubgraphId, dstSubgraphId, nodeImages, edge->dst(), &dstImage));

  // If this is a control edge then copy it and return. Lift control edges onto
  // the enclosing GEOp.
  if (edge->IsControlEdge()) {
    // Add the control edge, if we have not already added it, using the images
    // determined above.
    if (edges_added->emplace(NodeSlot(srcImage, -1), NodeSlot(dstImage, -1)).second) {
      graphOut->AddControlEdge(srcImage, dstImage);
    }
    return Status::OK();
  }

  int srcOutput = FindOutputSlotOfEdgeSrc(srcSubgraphId, dstSubgraphId, edge);
  int dstInput = FindOutputSlotOfEdgeDst(srcSubgraphId, dstSubgraphId, edge);
  // Add the edge, if we have not already added it.
  if (edges_added->emplace(NodeSlot(srcImage, srcOutput), NodeSlot(dstImage, dstInput)).second) {
    if (std::find(refIn_.begin(), refIn_.end(), edge) != refIn_.end()) {
      Status status;
      Node *identityNode =
          AddIdentityNode(graphOut, edge, srcImage->name(), srcOutput, srcImage->assigned_device_name(), &status);
      if (!status.ok()) { return status; }
      graphOut->AddEdge(srcImage, srcOutput, identityNode, 0);
      graphOut->AddEdge(identityNode, 0, dstImage, dstInput);
    } else {
      graphOut->AddEdge(srcImage, srcOutput, dstImage, dstInput);
    }
  }
  return Status::OK();
}

Status OMSplitter::AddEdgesToOutputGraph(const std::unordered_map<const Node *, Node *> &nodeImages, Graph *graphOut) {
  // Set of edges already added to the output graph, represented as (src, dst)
  // pairs. We use the set to deduplicate edges; multiple edges in the input
  // graph may map to one edge in the output graph.
  std::unordered_set<std::pair<NodeSlot, NodeSlot>, NodeSlot::PairHasher> edges_added;

  for (const Edge *edge : graphIn_->edges()) {
    REQUIRES_NOT_NULL(edge);
    string srcSubgraphId;
    TF_RETURN_IF_ERROR(GetSubgraphIdAttr(edge->src(), &srcSubgraphId));
    string dstSubgraphId;
    TF_RETURN_IF_ERROR(GetSubgraphIdAttr(edge->dst(), &dstSubgraphId));

    // Ignore edges that are strictly contained within one subgraph.
    if (IsInSubgraph(srcSubgraphId) && IsInSubgraph(dstSubgraphId) && srcSubgraphId == dstSubgraphId) { continue; }

    TF_RETURN_IF_ERROR(CopyEdgeToOutputGraph(edge, srcSubgraphId, dstSubgraphId, nodeImages, graphOut, &edges_added));
  }

  return Status::OK();
}

Status OMSplitter::BuildOutputGraph(Graph *graphOut) {
  // Map from nodes in the input graph to nodes in the output graph.
  std::unordered_map<const Node *, Node *> nodeImages;

  TF_RETURN_IF_ERROR(CopyNodesToOutputGraph(graphOut, &nodeImages));
  TF_RETURN_IF_ERROR(AddGEOpNodes(nodeImages, graphOut));
  TF_RETURN_IF_ERROR(AddEdgesToOutputGraph(nodeImages, graphOut));

  return Status::OK();
}

Status OMPartitionSubgraphsInFunctions(string groupAttribute, std::unique_ptr<Graph> *graph, const string &graph_format,
                                       FunctionLibraryDefinition *flib_def,
                                       std::map<std::string, std::string> npu_optimizer_options,
                                       std::map<std::string, std::string> pass_options,
                                       std::map<std::string, std::string> graph_options) {
  Graph *graphIn = graph->get();
  FunctionLibraryDefinition *const library = flib_def;

  OMSplitter omsplitter(std::move(groupAttribute), graphIn, std::move(npu_optimizer_options),
                        std::move(pass_options), std::move(graph_options));
  uint32_t subgraphNum = 0;
  TF_RETURN_IF_ERROR(omsplitter.SplitIntoSubgraphs(subgraphNum));

  if (subgraphNum == 0) {
    ADP_LOG(INFO) << "No Subgraph has been built.";
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(omsplitter.BuildFunctionDefs(library, graph_format));

  FunctionLibraryDefinition libraryOut(*library);
  std::unique_ptr<Graph> out(new (std::nothrow) Graph(libraryOut));
  REQUIRES_NOT_NULL(out);
  out->set_versions(graphIn->versions());
  TF_RETURN_IF_ERROR(omsplitter.BuildOutputGraph(out.get()));
  *graph = std::move(out);

  return Status::OK();
}
}  // namespace OMSplitter
static std::atomic<int> graph_run_num(1);
static mutex graph_num_mutex(LINKER_INITIALIZED);
Status OMPartitionSubgraphsPass::Run(const GraphOptimizationPassOptions &options) {
  if ((options.graph == nullptr && options.partition_graphs == nullptr) || options.flib_def == nullptr) {
    return Status::OK();
  }

  Status s = Status::OK();
  if (options.graph != nullptr) {
    std::unique_ptr<Graph> *graph = options.graph;
    FunctionLibraryDefinition *func_lib = options.flib_def;
    s = ProcessGraph(graph, func_lib, OptimizationPassRegistry::POST_REWRITE_FOR_EXEC);
    if (s != Status::OK()) { return s; }
  } else if (options.partition_graphs != nullptr) {
    for (auto &pg : *options.partition_graphs) {
      std::unique_ptr<Graph> *graph = &pg.second;
      FunctionLibraryDefinition *func_lib = options.flib_def;
      s = ProcessGraph(graph, func_lib, OptimizationPassRegistry::POST_PARTITIONING);
      if (s != Status::OK()) { return s; }
    }
  }

  return Status::OK();
}

void OMPartitionSubgraphsPass::ParseInputShapeRange(std::string dynamic_inputs_shape_range, bool enable_dp,
                                                    std::map<std::string, std::string> &graph_options) {
  static const size_t kLegalShapeVecSize = 2;
  std::vector<std::string> inputsVec;
  std::vector<std::string> shapesVec;
  Split(dynamic_inputs_shape_range, inputsVec, ";");
  std::sort(inputsVec.begin(), inputsVec.end());
  for (auto tmp : inputsVec) {
    std::vector<std::string> shapeVec;
    Split(tmp, shapeVec, ":");
    if (shapeVec.size() != kLegalShapeVecSize) {
      ADP_LOG(FATAL) << "dynamic_inputs_shape_range style is invalid, example:'data:[1,2];getnext:[2,3]'";
      LOG(FATAL) << "dynamic_inputs_shape_range style is invalid, example:'data:[1,2];getnext:[2,3]'";
    } else if (shapeVec[0] != "data" && shapeVec[0] != "getnext") {
      ADP_LOG(FATAL) << "dynamic_inputs_shape_range style is invalid, example:'data:[1,2];getnext:[2,3]'";
      LOG(FATAL) << "dynamic_inputs_shape_range style is invalid, example:'data:[1,2];getnext:[2,3]'";
    } else {
      shapesVec.push_back(shapeVec[1]);
    }
  }
  if (shapesVec.empty() || shapesVec.size() > kLegalShapeVecSize) {
    ADP_LOG(FATAL) << "dynamic_inputs_shape_range style is invalid, more than 2 input styles.";
    LOG(FATAL) << "dynamic_inputs_shape_range style is invalid, more than 2 input styles.";
  } else if (shapesVec.size() == 1) {
    if (!enable_dp) {
      graph_options["data_inputs_shape_range"] = shapesVec[0];
    } else {
      graph_options["getnext_inputs_shape_range"] = shapesVec[0];
    }
  } else {
    if (!enable_dp) {
      graph_options["data_inputs_shape_range"] = shapesVec[1] + "," + shapesVec[0];
    } else {
      graph_options["data_inputs_shape_range"] = shapesVec[0];
      graph_options["getnext_inputs_shape_range"] = shapesVec[1];
    }
  }
}

void OMPartitionSubgraphsPass::GetGraphDynamicExecConfig(Node *node, bool enable_dp,
    std::map<std::string, std::string> &graph_options) {
  // get attr from graph_options
  auto node_attrs = node->def().attr();
  const std::string kDynamicInput = "_graph_dynamic_input";
  const std::string kDynamicGraphExecuteMode = "_graph_dynamic_graph_execute_mode";
  const std::string kDynamicInputsShapeRange = "_graph_dynamic_inputs_shape_range";
  const std::string kIsTrainGraph = "_is_train_graph";
  if (node_attrs.find(kDynamicInput) != node_attrs.end()) {
    bool dynamic_input = node_attrs.at(kDynamicInput).b();
    graph_options["dynamic_input"] = std::to_string(dynamic_input);
    if (dynamic_input) {
      std::string graph_execute_mode = node_attrs.at(kDynamicGraphExecuteMode).s();
      graph_options["dynamic_graph_execute_mode"] = graph_execute_mode;
      std::string dynamic_inputs_shape_range = node_attrs.at(kDynamicInputsShapeRange).s();
      if (graph_execute_mode == "dynamic_execute" && !dynamic_inputs_shape_range.empty()) {
        ParseInputShapeRange(dynamic_inputs_shape_range, enable_dp, graph_options);
      }
    }
  }
  if (node_attrs.find(kIsTrainGraph) != node_attrs.end()) {
    bool is_train_graph = node_attrs.at(kIsTrainGraph).b();
    graph_options["train_graph"] = std::to_string(is_train_graph);
  }
}

Status OMPartitionSubgraphsPass::ProcessGetNext(Node *node, std::string enable_dp,
        std::vector<Node*> &remove_nodes, Graph *graphIn) {
  for (auto output_type: node->output_types()) {
    if (output_type == DT_STRING && enable_dp == "0") {
      ADP_LOG(WARNING) << "Dataset outputs have string output_type, please set enable_data_pre_proc=True.";
      LOG(WARNING) << "Dataset outputs have string output_type, please set enable_data_pre_proc=True.";
    }
  }
  // fuse Iterator and IteratorGetNext, build new node AdpGetNext.
  if (enable_dp == "1") {
    std::string queue_name;
    Node *iterator_node = nullptr;
    for (const Edge *e : node->in_edges()) {
      if (e->src()->type_string() == "Iterator" || e->src()->type_string() == "IteratorV2") {
        queue_name = e->src()->name();
        iterator_node = e->src();
      }
    }
    if (NpuAttrs::GetUseAdpStatus(queue_name)) {
      Node *adp_getnext_node = nullptr;
      std::string adp_getnext_name = "AdpGetNext_fuse_" + node->name();
      auto getnext_attrs = node->def().attr();
      uint32_t device_id = 0;
      (void)GetEnvDeviceID(device_id);
      TF_CHECK_OK(NodeBuilder(adp_getnext_name, "AdpGetNext")
                              .Device(node->def().device())
                              .Attr("queue_name", "device" + std::to_string(device_id) + "_" + queue_name)
                              .Attr("output_types", getnext_attrs["output_types"])
                              .Attr("output_shapes", getnext_attrs["output_shapes"])
                              .Finalize(&*graphIn, &adp_getnext_node));
      for (const Edge *e : node->out_edges()) {
        graphIn->AddEdge(adp_getnext_node, e->src_output(), e->dst(), e->dst_input());
      }
      remove_nodes.push_back(node);
      remove_nodes.push_back(iterator_node);
      ADP_LOG(INFO) << "Build AdpGetNext success.";
    }
  }
  return Status::OK();
}

Status OMPartitionSubgraphsPass::ProcessGraph(std::unique_ptr<Graph> *graph, FunctionLibraryDefinition *func_lib,
                                              const OptimizationPassRegistry::Grouping pass_group_value) {
  int graph_num = graph_run_num++;

  if (graph == nullptr) { return Status::OK(); }

  int64 startTime = InferShapeUtil::GetCurrentTimestap();

  for (Node *n : graph->get()->nodes()) {
    REQUIRES_NOT_NULL(n);
    if (n->attrs().Find("_NoNeedOptimize")) {
      ADP_LOG(INFO) << "Found mark of noneed optimize on node [" << n->name() << "], skip OMPartitionSubgraphsPass.";
      return Status::OK();
    }
  }

  std::map<std::string, std::string> all_options;
  std::map<std::string, std::string> pass_options;
  pass_options = NpuAttrs::GetDefaultPassOptions();
  for (Node *n : graph->get()->nodes()) {
    REQUIRES_NOT_NULL(n);
    if (n->attrs().Find("_NpuOptimizer")) {
      pass_options = NpuAttrs::GetPassOptions(n->attrs());
      all_options = NpuAttrs::GetAllAttrOptions(n->attrs());
      break;
    }
  }
  ADP_LOG(INFO) << "all options:";
  NpuAttrs::LogOptions(all_options);

  std::string job = pass_options["job"];
  if (job == "ps" || job == "default") {
    ADP_LOG(INFO) << "job is " << job << " Skip the optimizer : OMPartitionSubgraphsPass.";
    return Status::OK();
  }
  if (job == "localhost" && pass_group_value != OptimizationPassRegistry::POST_REWRITE_FOR_EXEC) {
    return Status::OK();
  }
  if (job != "localhost" && pass_group_value != OptimizationPassRegistry::POST_PARTITIONING) { return Status::OK(); }

  bool use_off_line = pass_options["use_off_line"] == "1";
  bool mix_compile_mode = pass_options["mix_compile_mode"] == "1";
  int iterations_per_loop = std::atoi(pass_options["iterations_per_loop"].c_str());
  int task_index = std::atoi(pass_options["task_index"].c_str());
  if (!iterations_per_loop) {
    ADP_LOG(FATAL) << "iterator_per_loop should be int and must >= 1";
    LOG(FATAL) << "iterator_per_loop should be int and must >= 1";
  }
  if (task_index < 0) {
    ADP_LOG(FATAL) << "task_index should be int and must >= 0";
    LOG(FATAL) << "task_index should be int and must >= 0";
  }
  bool do_npu_optimizer = pass_options["do_npu_optimizer"] == "1";
  if (do_npu_optimizer) {
    if (!use_off_line) {
      ADP_LOG(INFO) << "Run online process and skip the optimizer";
      return Status::OK();
    }
  } else {
    return Status::OK();
  }

  ADP_LOG(EVENT) << "OMPartition subgraph_" << std::to_string(graph_num) << " begin.";
  ADP_LOG(INFO) << "mix_compile_mode is " << (mix_compile_mode ? "True" : "False");
  ADP_LOG(INFO) << "iterations_per_loop is " << iterations_per_loop;
  ADP_LOG(INFO) << "input_shape: " << all_options["input_shape"]
                << "dynamic_dims: " << all_options["dynamic_dims"];
  bool is_set_dynamic_config = !all_options["input_shape"].empty() &&
                               !all_options["dynamic_dims"].empty();
  if (is_set_dynamic_config && mix_compile_mode) {
    ADP_LOG(FATAL) << "dynamic config can not use with mix compile.";
    LOG(FATAL) << "dynamic config can not use with mix compile.";
  }

  char *need_print = getenv("PRINT_MODEL");

  if (need_print != nullptr && strcmp("1", need_print) == 0) {
    GraphDef ori_graph_def;
    graph->get()->ToGraphDef(&ori_graph_def);
    string ori_model_path = GetDumpPath() + "BeforeSubGraph_";
    string omodel_path = ori_model_path + std::to_string(graph_num) + ".pbtxt";
    Status status_out = WriteTextProto(Env::Default(), omodel_path, ori_graph_def);
  }

  string graph_format_value;
  Graph *graphIn = graph->get();
  int getnext_node_count = 0;
  bool include_getnext = false;
  std::vector<Node*> remove_nodes;
  for (Node *node : graphIn->op_nodes()) {
    if (node->type_string() == "NPUInit") {
      std::string attr_name;
      for (const auto &option : all_options) {
        attr_name = std::string("_") + option.first;
        node->AddAttr(attr_name, option.second);
      }
      node->AddAttr("_NpuOptimizer", "NpuOptimizer");
    }

    if (node->type_string() == "IteratorGetNext") {
      include_getnext = true;
      if (is_set_dynamic_config) { getnext_node_count++; }
      // fuse Iterator and IteratorGetNext, build new node AdpGetNext.
      TF_CHECK_OK(ProcessGetNext(node, pass_options["enable_dp"], remove_nodes, graphIn));
    } else if (node->type_string() == "_UnaryOpsComposition") {
      ADP_LOG(INFO) << "begin split _UnaryOpsComposition.";
      Node *pre_node = nullptr;
      if (node->in_edges().size() != 1) {
        ADP_LOG(INFO) << "edge size if not 1, not support in _UnaryOpsComposition.";
        continue;
      }
      pre_node = (*node->in_edges().begin())->src();
      auto attr_map = node->def().attr();
      auto node_list = attr_map["op_names"].list();
      Node *unary_node = nullptr;
      for (int i = 0; i < node_list.s_size(); i++) {
        const string &node_name = node_list.s(i);
        string op_name = node->name() + "_" + std::to_string(i) + "_" + node_name;
        ADP_LOG(INFO) << "op_names node_list: " << i << " is node: " << node_name;
        TF_CHECK_OK(NodeBuilder(op_name, node_name)
                        .Input(pre_node, 0)
                        .Device(pre_node->def().device())
                        .Finalize(&*graphIn, &unary_node));
        REQUIRES_NOT_NULL(unary_node);
        ADP_LOG(INFO) << unary_node->type_string() << " has built success.";
        pre_node = unary_node;
      }
      for (auto out_edge : node->out_edges()) {
        ADP_LOG(INFO) << "begin add edge ";
        graphIn->AddEdge(unary_node, 0, out_edge->dst(), out_edge->dst_input());
      }
      graphIn->RemoveNode(node);
    } else if (node->type_string() == "DestroyTemporaryVariable" && node->name().find("AccumulateNV2") != std::string::npos) {
      TF_RETURN_IF_ERROR(AccumulateNFusion(graphIn, node));
    }
  }
  for (Node *node : remove_nodes) {
    ADP_LOG(INFO) << "Remove node: " << node->name();
    graphIn->RemoveNode(node);
  }
  if (getnext_node_count > 1) {
    ADP_LOG(FATAL) << "dynamic dims func can not support graph with "
                   << getnext_node_count << " IteratorGetNext node.";
    LOG(FATAL) << "dynamic dims func can not support graph with "
               << getnext_node_count << " IteratorGetNext node.";
  }
  std::map<std::string, std::string> graph_options;
  bool enable_dp = (pass_options["enable_dp"] == "1") && include_getnext;
  for (Node *node : graphIn->op_nodes()) {
    if (node->type_string() == "OneShotIterator" && iterations_per_loop != 1) {
      ADP_LOG(FATAL) << "iterator_per_loop only support 1 when using OneShotIterator";
      LOG(FATAL) << "iterator_per_loop only support 1 when using OneShotIterator";
    }
    // get attr from graph options.
    GetGraphDynamicExecConfig(node, enable_dp, graph_options);
    if (graph_options["dynamic_input"] == "1" && iterations_per_loop > 1) {
      ADP_LOG(FATAL) << "iterations_per_loop only support 1 in dynamic input mode.";
      LOG(FATAL) << "iterations_per_loop only support 1 in dynamic input mode.";
    }
    string device_name;
    if (job != "localhost" && job != "ps" && job != "default") {
      device_name = std::string("/job:") + std::string(job) + std::string("/replica:0/task:")
          + std::to_string(task_index) + std::string("/device:CPU:0");
    } else if (job == "localhost") {
      device_name = string("/job:localhost/replica:0/task:0/device:CPU:0");
    } else {
      return errors::InvalidArgument("job type is : ", job, " not support. ");
    }

    node->set_assigned_device_name(device_name);

    string node_format_value;
    Status status = GetNodeAttr(node->def(), "data_format", &node_format_value);
    if (status.ok() && !node_format_value.empty()) {
      if (graph_format_value.empty()) { graph_format_value = node_format_value; }
    }
  }
  // get attr from pass_options
  if (graph_options["dynamic_input"].empty()) {
    std::string dynamic_graph_execute_mode = pass_options["dynamic_graph_execute_mode"];
    graph_options["dynamic_input"] = pass_options["dynamic_input"];
    graph_options["dynamic_graph_execute_mode"] = dynamic_graph_execute_mode;
    std::string dynamic_inputs_shape_range = pass_options["dynamic_inputs_shape_range"];
    if (dynamic_graph_execute_mode == "dynamic_execute" && !dynamic_inputs_shape_range.empty()) {
      ParseInputShapeRange(dynamic_inputs_shape_range, enable_dp, graph_options);
    }
  }

  if (graph_format_value.empty()) {
    graph_format_value = "NHWC";  // default value
  }

  int subgraphNum = 0;
  TF_RETURN_IF_ERROR(
      OMSplitter::MarkForPartition(graph, subgraphNum, mix_compile_mode, graph_num,
                                   func_lib, pass_options, graph_options));
  ADP_LOG(EVENT) << "OMPartition subgraph_" << std::to_string(graph_num) << " markForPartition success.";
  if (subgraphNum < 1) {
    ADP_LOG(INFO) << "subgraphNum is " << subgraphNum;
    return Status::OK();
  }
  if (mix_compile_mode) {
    std::vector<const Edge *> varEdges;
    for (Node *node : graphIn->op_nodes()) {
      if (node->type_string() == "VariableV2" || node->type_string() == "VarHandleOp") {
        for (auto out_edge : node->out_edges()) { varEdges.push_back(out_edge); }
      }
    }
    for (auto varEdge : varEdges) {
      REQUIRES_NOT_NULL(varEdge);
      REQUIRES_NOT_NULL(varEdge->src());
      REQUIRES_NOT_NULL(varEdge->dst());
      string srcSubgraphId;
      const string partitionAttr = OMSplitter::PARTITION_SUB_GRAPH_ATTR;
      Status s = GetNodeAttr(varEdge->src()->attrs(), partitionAttr, &srcSubgraphId);
      if (s.code() != error::Code::NOT_FOUND) {
        if (!s.ok()) { return s; }
      }

      DataType dtypeDst = varEdge->dst()->input_type(varEdge->dst_input());
      string dstSubgraphId;
      s = GetNodeAttr(varEdge->dst()->attrs(), partitionAttr, &dstSubgraphId);
      if (s.code() == error::Code::NOT_FOUND) {
        if (!(IsRefType(dtypeDst) || dtypeDst == DT_RESOURCE)) {
          continue;
        } else {
          return errors::InvalidArgument("Ref Tensors (e.g., Variables) output: ", varEdge->dst()->name(),
                                         " is not in white list");
        }
      } else if (!s.ok()) {
        return s;
      }
      if ((IsRefType(dtypeDst) || dtypeDst == DT_RESOURCE) && srcSubgraphId != dstSubgraphId) {
        Node *nodeCopy = graphIn->AddNode(varEdge->src()->def(), &s);
        if (!s.ok()) { return s; }
        nodeCopy->ClearAttr(partitionAttr);
        nodeCopy->AddAttr(partitionAttr, dstSubgraphId);
        graphIn->AddEdge(nodeCopy, varEdge->src_output(), varEdge->dst(), varEdge->dst_input());
        graphIn->RemoveEdge(varEdge);
      }
    }
  }

  if (mix_compile_mode) {
    auto nodes = graphIn->op_nodes();
    for (Node *node : nodes) {
      if (node->IsConstant()) {
        std::map<string, Node *> copiedConsts;
        string srcSubgraphId;
        const string partitionAttr = OMSplitter::PARTITION_SUB_GRAPH_ATTR;
        Status s = GetNodeAttr(node->attrs(), partitionAttr, &srcSubgraphId);
        if (s.code() != error::Code::NOT_FOUND) {
          if (!s.ok()) { return s; }
        }
        std::vector<const Edge *> edges;
        for (auto edge : node->out_edges()) { edges.push_back(edge); }
        for (auto edge : edges) {
          REQUIRES_NOT_NULL(edge);
          REQUIRES_NOT_NULL(edge->src());
          REQUIRES_NOT_NULL(edge->dst());
          string dstSubgraphId;
          s = GetNodeAttr(edge->dst()->attrs(), partitionAttr, &dstSubgraphId);
          if (s.code() == error::Code::NOT_FOUND) {
            continue;
          } else if (!s.ok()) {
            return s;
          }
          if (srcSubgraphId != dstSubgraphId) {
            if (copiedConsts.find(dstSubgraphId) == copiedConsts.end()) {
              Node *nodeCopy = graphIn->AddNode(edge->src()->def(), &s);
              if (!s.ok()) { return s; }
              nodeCopy->set_name(node->name() + "_copied");
              nodeCopy->ClearAttr(partitionAttr);
              nodeCopy->AddAttr(partitionAttr, dstSubgraphId);
              copiedConsts[dstSubgraphId] = nodeCopy;
              ADP_LOG(INFO) << "Copy const node:" << node->name();
            }
            Node *nodeCopy = copiedConsts[dstSubgraphId];
            graphIn->AddEdge(nodeCopy, edge->src_output(), edge->dst(), edge->dst_input());
            graphIn->RemoveEdge(edge);
            graphIn->AddControlEdge(edge->src(), nodeCopy, false);
          }
        }
      }
    }
  }
  TF_RETURN_IF_ERROR(OMSplitter::OMPartitionSubgraphsInFunctions(
      OMSplitter::PARTITION_SUB_GRAPH_ATTR, graph, graph_format_value, func_lib, all_options, pass_options, graph_options));
  ADP_LOG(EVENT) << "OMPartition subgraph_" << std::to_string(graph_num) << " SubgraphsInFunctions success.";
  FixupSourceAndSinkEdges(graph->get());

  if (need_print != nullptr && strcmp("1", need_print) == 0) {
    GraphDef omg_graph_def;
    graph->get()->ToGraphDef(&omg_graph_def);
    string tmpmodel_path = GetDumpPath() + "AfterSubGraph_";
    string tmodel_path = tmpmodel_path + std::to_string(graph_num) + ".pbtxt";
    Status status_o = WriteTextProto(Env::Default(), tmodel_path, omg_graph_def);
  }
  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(EVENT) << "OMPartition subgraph_" << std::to_string(graph_num) << " success. ["
                 << ((endTime - startTime) / kMicrosToMillis) << " ms]";
  return Status::OK();
}

Status OMPartitionSubgraphsPass::AccumulateNFusion(Graph *graphIn, Node *node) {
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

  std::vector<tensorflow::NodeBuilder::NodeOut> input_nodes;
  std::vector<Node *> delete_nodes;
  TensorShapeProto variable_shape;
  for (auto input_edge : node->in_edges()) {
    if (input_edge->src()->type_string() == "AssignAdd") {
      for (auto assignadd_inedge : input_edge->src()->in_edges()) {
        if (assignadd_inedge->src()->type_string() != "Assign") {
          input_nodes.emplace_back(assignadd_inedge->src());
        }
      }
      delete_nodes.emplace_back(input_edge->src());
    } else {
      for (auto assign_inedge : input_edge->src()->in_edges()) {
        if (assign_inedge->src()->type_string() == "Const" && !assign_inedge->IsControlEdge()) {
          variable_shape = assign_inedge->src()->def().attr().at("value").tensor().tensor_shape();
          delete_nodes.emplace_back(assign_inedge->src());
        } else if (assign_inedge->src()->type_string() == "TemporaryVariable") {
          delete_nodes.emplace_back(assign_inedge->src());
        } else {
          ADP_LOG(INFO) << "unsupport node in accumulate_n sub graph.";
        }
      }
      delete_nodes.emplace_back(input_edge->src());
    }
  }
  delete_nodes.emplace_back(node);

  Node *accumulate_node = nullptr;
  TF_CHECK_OK(NodeBuilder(node->name(), "AccumulateNV2")
                  .Input(input_nodes)
                  .Attr("T", dtype)
                  .Attr("shape", variable_shape)
                  .Device(node->def().device())
                  .Finalize(&*graphIn, &accumulate_node));
  REQUIRES_NOT_NULL(accumulate_node);

  for (auto out_edge : node->out_edges()) {
    if (out_edge->IsControlEdge()) {
      graphIn->AddControlEdge(accumulate_node, out_edge->dst());
    } else {
      graphIn->AddEdge(accumulate_node, 0, out_edge->dst(), out_edge->dst_input());
    }
  }
  for (auto delete_node : delete_nodes) {
    graphIn->RemoveNode(delete_node);
  }
  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 3, OMPartitionSubgraphsPass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 101, OMPartitionSubgraphsPass);
}  // namespace tensorflow
