/**
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

#include "tf_adapter/optimizers/dp_tf_ge_conversion_pass.h"

#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <algorithm>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/public/session_options.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/infershape_util.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/util.h"

namespace tensorflow {
static const int64 kMicrosToMillis = 1000;
static std::atomic<int64> g_channel_index(1);
// GE ops white list
const static std::vector<std::string> GE_OPS_WHITELIST = {
    "MapDataset",     "ParallelMapDataset",   "BatchDataset", "MapAndBatchDataset", "DeviceQueueDataset",
    "BatchDatasetV2", "MapAndBatchDatasetV2", "ModelDataset", "OptimizeDataset"};

// Customize dataset list
const static std::vector<std::string> CUSTOMIZE_DATASET_LIST = {"BatchDataset",       "BatchDatasetV2",
                                                                "MapAndBatchDataset", "MapAndBatchDatasetV2",
                                                                "ParallelMapDataset", "MakeIterator"};
// Skip dataset list
const static std::vector<std::string> SKIP_DATASET_LIST = {"ModelDataset", "OptimizeDataset"};

// Need add option attr
const static std::vector<std::string> NEED_OPTION_ATTR_DATASET_LIST = {"NpuMapDataset", "NpuMapAndBatchDataset"};

// GE fun black list
const static std::vector<std::string> GE_FUN_BLACKLIST = {"PyFunc",
                                                          "SaveV2",
                                                          "RestoreV2",
                                                          "MergeV2Checkpoints",
                                                          "Save",
                                                          "SaveSlices",
                                                          "Restore",
                                                          "RestoreSlice",
                                                          "ShardedFilename",
                                                          "ShardedFilespec",
                                                          "WholeFileReader",
                                                          "WholeFileReaderV2",
                                                          "TextLineReader",
                                                          "TextLineReaderV2",
                                                          "FixedLengthRecordReader",
                                                          "FixedLengthRecordReaderV2",
                                                          "LMDBReader",
                                                          "IdentityReader",
                                                          "IdentityReaderV2",
                                                          "ReaderRead",
                                                          "ReaderReadV2",
                                                          "ReaderReadUpTo",
                                                          "ReaderReadUpToV2",
                                                          "ReaderNumRecordsProduced",
                                                          "ReaderNumRecordsProducedV2",
                                                          "ReaderNumWorkUnitsCompleted",
                                                          "ReaderNumWorkUnitsCompletedV2",
                                                          "ReaderSerializeState",
                                                          "ReaderSerializeStateV2",
                                                          "ReaderRestoreState",
                                                          "ReaderRestoreStateV2",
                                                          "ReaderReset",
                                                          "ReaderResetV2",
                                                          "ReadFile",
                                                          "WriteFile",
                                                          "MatchingFiles",
                                                          "TFRecordReader",
                                                          "TFRecordReaderV2",
                                                          "MutableHashTable",
                                                          "MutableHashTableV2",
                                                          "MutableDenseHashTable",
                                                          "MutableDenseHashTableV2",
                                                          "InitializeTable",
                                                          "InitializeTableV2",
                                                          "InitializeTableFromTextFile",
                                                          "InitializeTableFromTextFileV2",
                                                          "MutableHashTableOfTensors",
                                                          "MutableHashTableOfTensorsV2",
                                                          "HashTable",
                                                          "HashTableV2",
                                                          "LookupTableInsert",
                                                          "LookupTableInsertV2",
                                                          "LookupTableExport",
                                                          "LookupTableExportV2",
                                                          "LookupTableImport",
                                                          "LookupTableImportV2",
                                                          "LookupTableFind",
                                                          "LookupTableFindV2"};
// Mark string for iterator_name
const static std::string DP_ITERATOR_MARK = "_iterator_name";
// Mark string for dp_init graph
const static std::string DP_INIT_GRAPH_MARK = "MakeIterator";
const static std::string DP_INIT_NOOP_GRAPH_MARK = "NoOp";
// Mark string for iterator node
const static std::string DP_INIT_ITERATOR_MARK = "Iterator";
// Mark string for device node
const static std::string DP_INIT_DEVICEQUEUE_MARK = "DeviceQueueDataset";
// Mark string for queue node
const static std::string DP_INIT_QUEUE_MARK = "QueueDataset";
// Used for 0-input NodeDefBuilder
const static std::vector<NodeDefBuilder::NodeOut> EMPTY_DEF_INPUT;
// Used for 0-input NodeBuilder
const static std::vector<NodeBuilder::NodeOut> EMPTY_INPUT;
// Used for 0-type Node(Def)Builder
const static DataTypeVector EMPTY_TYPE;
// Used for 0-shape Node(Def)Builder
const static std::vector<PartialTensorShape> EMPTY_SHAPE;

class DpTfToGEConversionPassImpl {
 public:
  explicit DpTfToGEConversionPassImpl() : graph_run_num_(0), graph_(nullptr), flib_def_(nullptr){};

  ~DpTfToGEConversionPassImpl() = default;
  Status Run(const GraphOptimizationPassOptions &options);

 private:
  Status ProcessGraph(const std::unique_ptr<Graph> *graph, FunctionLibraryDefinition *func_lib,
                      const OptimizationPassRegistry::Grouping pass_group_value);
  bool RunPass(const std::unique_ptr<Graph> *g, FunctionLibraryDefinition *flib,
               const std::map<std::string, std::string> &all_options);
  inline bool IsMakeIteratorNode(const Node *n) const;
  inline bool IsAddOptionAttrNode(const Node *n) const;
  inline bool IsIteratorNode(const Node *n) const;
  inline bool IsSkipDataset(const Node *n) const;
  inline bool IsGeSupportDataset(const Node *n) const;
  inline std::string GetEdgeName(const Edge *e) const;
  inline std::string GetRandomName(const std::string &prefix) const;
  std::string GetRandomName() const;
  inline bool EndsWith(const std::string &str, const std::string &suffix) const;
  inline bool CheckNode(const std::string &op) const;
  inline bool IsDeviceSupportedOp(const NodeDef &n) const;
  inline bool IsDeviceSupportedFunc(const std::string &fn) const;
  inline Status GetSplitEdges(const Node *n, std::vector<const Edge *> &split_edges, const Edge *last_edge);
  inline void RemoveSplitEdges(Node *topo_end);
  Status InsertChannelQueue(Node *topo_end, std::string &host_queue_name, std::string &device_queue_name,
                            const std::map<std::string, std::string> &all_options) const;
  bool GetNodeFuncs(const FunctionLibraryDefinition *flib_def, const Node *node, std::vector<string> &node_funcs) const;
  bool RemoveIsolatedNode(Graph *g, std::unordered_set<Node *> visited) const;
  Status RemoveNotSupportDataset(Graph *g, const std::string &device_queue_dataset,
                                 const std::string &make_iterator) const;
  Status AddDataTransDatasets(Node *topo_end, std::string &host_channel_name, std::string &device_channel_name,
                              const std::map<std::string, std::string> &all_options);
  void GetTopoEndsNodes(std::vector<Node *> &topo_ends) const;
  Status BuildDeviceDpGraph(const Node *topo_end, Graph *device_graph, const std::string &device_channel_name) const;
  Status AddAttr2DeviceNodes(const Node *topo_end, const Graph *device_graph) const;
  Status AddGeopNodeFunctionDef(FunctionDefLibrary &fdeflib, const std::string &fn_geop, const std::string &fn_dpop,
                                const string &default_device) const;
  Status AddGeopDatasetFunctionDef(FunctionDefLibrary &fdeflib, const std::string &fn_geop,
                                   const std::string &fn_geop_dataset, const string &default_device,
                                   const std::map<std::string, std::string> &all_options) const;
  Status BuildGeOpDatasetFunction(FunctionDefLibrary &fdeflib, Graph *device_graph, const std::string &fn_geop_dataset,
                                  const string &default_device,
                                  const std::map<std::string, std::string> &all_options) const;
  Status AddGeOpDatasetFunctionLibrary(FunctionLibraryDefinition *flib, const Node *topo_end,
                                       const std::string &device_channel_name, const std::string &fn_geop_dataset,
                                       const std::map<std::string, std::string> &all_options);
  Status AddGeOpDatasetAndDpGroupDataset(const Node *topo_end, const std::string &fn_geop_dataset,
                                         const std::string &host_channel_name,
                                         const std::string &device_channel_name) const;
  void AddOptionAttr(std::vector<Node *> nodes, const std::map<std::string, std::string> &all_options);

  // graph num
  int graph_run_num_;
  // All split edges, split edges means edges that combine A and B in this case
  // 1) A = a node that can only run on tensorflow python host and,
  // 2) B = a node that can run on GE device and all nodes followed B can run on
  // GE device,
  std::unordered_map<Node *, std::vector<const Edge *>> split_edges_;
  // Input graph, not owned
  Graph *graph_;
  // Input flib, not owned
  const FunctionLibraryDefinition *flib_def_;
};

inline bool DpTfToGEConversionPassImpl::IsMakeIteratorNode(const Node *n) const {
  return str_util::StartsWith(n->type_string(), DP_INIT_GRAPH_MARK);
}

inline bool DpTfToGEConversionPassImpl::IsAddOptionAttrNode(const Node *n) const {
  return std::find(NEED_OPTION_ATTR_DATASET_LIST.begin(), NEED_OPTION_ATTR_DATASET_LIST.end(),
                   n->type_string()) != NEED_OPTION_ATTR_DATASET_LIST.end();
}

inline bool DpTfToGEConversionPassImpl::IsIteratorNode(const Node *n) const {
  return str_util::StartsWith(n->type_string(), DP_INIT_ITERATOR_MARK);
}

inline bool DpTfToGEConversionPassImpl::IsSkipDataset(const Node *n) const {
  return std::find(SKIP_DATASET_LIST.begin(), SKIP_DATASET_LIST.end(), n->type_string()) != SKIP_DATASET_LIST.end();
}

inline bool DpTfToGEConversionPassImpl::IsGeSupportDataset(const Node *n) const {
  return std::find(GE_OPS_WHITELIST.begin(), GE_OPS_WHITELIST.end(), n->type_string()) != GE_OPS_WHITELIST.end();
}

inline std::string DpTfToGEConversionPassImpl::GetEdgeName(const Edge *e) const {
  if (e == nullptr || e->src() == nullptr || e->dst() == nullptr) {
    return "invalid_edge";
  }
  return npu::CatStr("Edge_from_", e->src()->name(), "_out", e->src_output(), "_To_", e->dst()->name(), "_in",
                     e->dst_input());
}

inline std::string DpTfToGEConversionPassImpl::GetRandomName(const std::string &prefix) const {
  return npu::CatStr(prefix, "_", GetRandomName());
}

std::string DpTfToGEConversionPassImpl::GetRandomName() const {
  random::PhiloxRandom philox(random::New64(), random::New64());
  random::SimplePhilox rnd(&philox);
  const static size_t RANDOM_LEN = 11;
  const static uint32_t CHARACTER_SETS_HEAD = 2;
  const static uint32_t CHARACTER_SETS = 3;
  const static uint32_t CHARACTER_SET_SIZE[] = {26, 26, 10};  // a-z A-Z 0-9
  const static uint32_t ASCII_UP_A = 65;                      // Ascii of 'A'
  const static uint32_t ASCII_LO_A = 97;                      // Ascii of 'a'
  const static uint32_t ASCII_0 = 48;                         // Ascii of '0'
  const static uint32_t ASCII_BASE[] = {ASCII_UP_A, ASCII_LO_A, ASCII_0};
  string x;
  uint32_t setIdx = 0;
  for (size_t i = 0; i < RANDOM_LEN; i++) {
    if (i == 0) {  // Character must not start with 0-9
      setIdx = rnd.Uniform(CHARACTER_SETS_HEAD);
    } else {
      setIdx = rnd.Uniform(CHARACTER_SETS);
    }
    uint32_t asciiIdx = rnd.Uniform(CHARACTER_SET_SIZE[setIdx]);
    x += char(ASCII_BASE[setIdx] + asciiIdx);
  }
  return x;
}

inline bool DpTfToGEConversionPassImpl::EndsWith(const std::string &str, const std::string &suffix) const {
  return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool DpTfToGEConversionPassImpl::CheckNode(const std::string &op) const {
  std::string suffix_op = "Dataset";
  std::string suffix_op_v2 = "DatasetV2";
  if (EndsWith(op, suffix_op) || EndsWith(op, suffix_op_v2)) {
    if (std::find(GE_OPS_WHITELIST.begin(), GE_OPS_WHITELIST.end(), op) != GE_OPS_WHITELIST.end()) {
      return true;
    } else {
      return false;
    }
  } else {
    if (std::find(GE_FUN_BLACKLIST.begin(), GE_FUN_BLACKLIST.end(), op) == GE_FUN_BLACKLIST.end()) {
      return true;
    } else {
      return false;
    }
  }
}

inline bool DpTfToGEConversionPassImpl::IsDeviceSupportedOp(const NodeDef &n) const {
  const OpRegistrationData *op_reg_data = nullptr;
  // Tenorflow original op
  if (OpRegistry::Global() == nullptr) {
    ADP_LOG(ERROR) << "OpRegistry global is nullptr";
    LOG(ERROR) << "OpRegistry global is nullptr";
    return false;
  }
  if (OpRegistry::Global()->LookUp(n.op(), &op_reg_data).ok()) {
    // Node in GE supported
    if (!CheckNode(n.op())) {
      ADP_LOG(INFO) << "Node [" << n.name() << "] op [" << n.op() << "] not supported by GE";
      return false;
    } else {  // Top node supported by GE, check its owned function
      for (auto &attr : n.attr()) {
        if (attr.second.has_func()) {
          if (!IsDeviceSupportedFunc(attr.second.func().name())) {
            ADP_LOG(INFO) << "Node [" << n.name() << "] function [" << attr.second.func().name()
                          << "] not supported by GE";
            return false;
          }
        }
      }
    }
  } else {  // Not tenorflow original op, this must be a function node
    if (!IsDeviceSupportedFunc(n.op())) {
      ADP_LOG(INFO) << "Node [" << n.name() << "] op [" << n.op() << "] is not a supported function by GE";
      return false;
    }
  }
  return true;
}

inline bool DpTfToGEConversionPassImpl::IsDeviceSupportedFunc(const std::string &fn) const {
  const FunctionDef *fdef = flib_def_->Find(fn);
  // Node contains not found function
  if (fdef == nullptr) {
    ADP_LOG(ERROR) << "Function [" << fn << "] not found";
    LOG(ERROR) << "Function [" << fn << "] not found";
    return false;
  }
  // Recursive check function node
  auto iter = std::find_if(fdef->node_def().begin(), fdef->node_def().end(),
                           [this](const NodeDef &node) { return !IsDeviceSupportedOp(node); });
  if (iter != fdef->node_def().end()) {
    ADP_LOG(INFO) << "Function [" << fn << "] node [" << iter->name() << "] not supported by GE";
    return false;
  }
  return true;
}

inline Status DpTfToGEConversionPassImpl::GetSplitEdges(const Node *n, std::vector<const Edge *> &split_edges,
                                                        const Edge *last_edge) {
  if (IsMakeIteratorNode(n)) {
    for (const Edge *e : n->in_edges()) {
      REQUIRES_NOT_NULL(e);
      if (!IsIteratorNode(e->src())) {
        last_edge = e;
        ADP_LOG(INFO) << npu::CatStr("last edge", GetEdgeName(last_edge));
      }
    }
  }
  // GE supported node, continue find
  if (IsDeviceSupportedOp(n->def())) {
    for (const Edge *e : n->in_edges()) {
      REQUIRES_NOT_NULL(e);
      REQUIRES_NOT_NULL(e->src());
      REQUIRES_NOT_NULL(e->dst());
      if (e->IsControlEdge() && !e->src()->IsSource()) {
        return errors::InvalidArgument("Graph contains control edges witch not from _SOURCE, will not try "
                                       "optimize");
      }
      // GE supported node, continue find
      if (kIsHeterogeneous) {
        if (!IsIteratorNode(e->src())) {
          split_edges.push_back(last_edge);
        }
      } else if (IsDeviceSupportedOp(e->src()->def())) {
        Status s = GetSplitEdges(e->src(), split_edges, last_edge);
        if (!s.ok()) {
          return s;
        }
      } else {  // GE unsupported node, this is a split edge
        ADP_LOG(INFO) << npu::CatStr("Split_", GetEdgeName(e));
        ADP_LOG(INFO) << "Begin check split edge.";
        if (IsSkipDataset(e->dst())) {
          ADP_LOG(INFO) << "ADD last edge " << GetEdgeName(last_edge);
          split_edges.push_back(last_edge);
        } else {
          ADP_LOG(INFO) << "ADD last edge " << GetEdgeName(e);
          split_edges.push_back(e);
        }
      }
    }
  }
  return Status::OK();
}

Status DpTfToGEConversionPassImpl::InsertChannelQueue(Node *topo_end, std::string &host_queue_name,
                                                      std::string &device_queue_name,
                                                      const std::map<std::string, std::string> &all_options) const {
  ADP_LOG(INFO) << "Start to insert HostQueueDataset and DeviceQueueDataset.";
  REQUIRES_NOT_NULL(topo_end);
  const Node *iterator_node = nullptr;
  if (IsMakeIteratorNode(topo_end)) {
    (void) topo_end->input_node(1, &iterator_node);
  }

  for (const Edge *e : split_edges_.at(topo_end)) {
    REQUIRES_NOT_NULL(e);
    REQUIRES_NOT_NULL(e->src());
    REQUIRES_NOT_NULL(e->dst());
    bool need_add_device_dataset = false;
    if (kIsHeterogeneous) {
      need_add_device_dataset = false;
    } else if ((!NpuAttrs::GetNewDataTransferFlag()) || (IsGeSupportDataset(e->dst()))) {
      need_add_device_dataset = true;
    } else {
      need_add_device_dataset = false;
    }

    std::string local_rank_id = all_options.at("local_rank_id");
    std::string local_device_list = all_options.at("local_device_list");
    std::string channel_name;
    if (local_rank_id == "-1") {
      REQUIRES_NOT_NULL(iterator_node);
      if (!need_add_device_dataset) {
        channel_name = iterator_node->name();
      } else {
        channel_name = "Queue_" + GetEdgeName(e) + "_" + GetRandomName();
      }
    } else {
      channel_name = npu::CatStr(e->src()->name(), "_index_", std::to_string(g_channel_index));
      g_channel_index += 1;
    }
    host_queue_name = "HostQueue_" + channel_name;
    ADP_LOG(INFO) << "Add_" << host_queue_name;
    // Host and Device queue should save type and shape
    auto m_src = e->src()->def().attr();
    bool type_status = false;
    string::size_type idx = SummarizeAttrValue(m_src["output_types"]).find("Unknown AttrValue");
    if (idx == string::npos) {
      type_status = true;
    }
    Node *queue_node_host = nullptr;
    // Make sure that 'channel_name' of host and device queue be same
    TF_CHECK_OK(NodeBuilder(host_queue_name, "HostQueueDataset")
                    .Input(e->src(), e->src_output())  // Will be replaced by GEOPDataset later
                    .Input(e->src(), e->src_output())
                    .Device(e->src()->def().device())
                    .Attr("channel_name", channel_name)
                    .Attr("output_types", type_status ? m_src["output_types"] : m_src["Toutput_types"])
                    .Attr("output_shapes", m_src["output_shapes"])
                    .Attr("_local_rank_id", local_rank_id)
                    .Attr("_local_device_list", local_device_list)
                    .Finalize(graph_, &queue_node_host));
    REQUIRES_NOT_NULL(queue_node_host);

    if (!need_add_device_dataset) {
      return Status::OK();
    }

    device_queue_name = "DeviceQueue_" + channel_name;
    ADP_LOG(INFO) << "Add_" << device_queue_name;
    Node *queue_node_device = nullptr;
    // Make sure that 'channel_name' of host and device queue be same
    TF_CHECK_OK(NodeBuilder(device_queue_name, "DeviceQueueDataset")
                    .Device(e->dst()->def().device())
                    .Attr("channel_name", channel_name)
                    .Attr("output_types", type_status ? m_src["output_types"] : m_src["Toutput_types"])
                    .Attr("output_shapes", m_src["output_shapes"])
                    .Finalize(graph_, &queue_node_device));
    REQUIRES_NOT_NULL(queue_node_device);
    // 0 means the the 0th output of queue_node_device
    REQUIRES_NOT_NULL(graph_->AddEdge(queue_node_device, 0, e->dst(), e->dst_input()));
  }
  return Status::OK();
}

Status DpTfToGEConversionPassImpl::RemoveNotSupportDataset(Graph *g, const std::string &device_queue_dataset,
                                                           const std::string &make_iterator) const {
  ADP_LOG(INFO) << "Begin RemoveSplitDataset.";
  // find device_queue_dataset and make_iterator
  Node *node = nullptr;
  Node *topo_end = nullptr;
  for (Node *n : g->op_nodes()) {
    REQUIRES_NOT_NULL(n);
    if (n->type_string() == "DeviceQueueDataset" && n->name() == device_queue_dataset) {
      ADP_LOG(INFO) << "device queue dataset node is " << n->name();
      node = n;
    }
    if (n->type_string() == "MakeIterator" && n->name() == make_iterator) {
      ADP_LOG(INFO) << "make iterator node is " << n->name();
      topo_end = n;
    }
  }
  REQUIRES_NOT_NULL(node);
  Node *end_dataset = node;
  std::vector<Node *> delete_nodes;
  while (!IsMakeIteratorNode(node)) {
    if (IsSkipDataset(node)) {
      delete_nodes.push_back(node);
    } else {
      end_dataset = node;
    }
    if (node->num_outputs() != 1) {
      ADP_LOG(ERROR) << "Invalid node " << node->name() << ", op is" << node->type_string();
      LOG(ERROR) << "Invalid node " << node->name() << ", op is" << node->type_string();
      return errors::InvalidArgument("RemoveSplitDataset: find invalid node.");
    }
    const Edge *edge = nullptr;
    for (const Edge *e : node->out_edges()) {
      edge = e;
    }
    REQUIRES_NOT_NULL(edge);
    REQUIRES_NOT_NULL(edge->dst());
    node = edge->dst();
  }
  if (delete_nodes.empty()) {
    ADP_LOG(INFO) << "all sink datasets are supported.";
    return Status::OK();
  }
  for (Node *n : delete_nodes) {
    ADP_LOG(INFO) << "ready to remove node " << n->name();
    g->RemoveNode(n);
  }
  ADP_LOG(INFO) << "end dataset node is " << end_dataset->name();
  REQUIRES_NOT_NULL(g->AddEdge(end_dataset, 0, topo_end, 0));
  return Status::OK();
}

inline void DpTfToGEConversionPassImpl::RemoveSplitEdges(Node *topo_end) {
  ADP_LOG(INFO) << "Start to remove split edges";
  for (const Edge *e : split_edges_.at(topo_end)) {
    ADP_LOG(INFO) << "Remove_" << GetEdgeName(e);
    graph_->RemoveEdge(e);
  }
}

bool DpTfToGEConversionPassImpl::GetNodeFuncs(const FunctionLibraryDefinition *flib_def, const Node *node,
                                              std::vector<string> &node_funcs) const {
  node_funcs.clear();
  for (auto iter = node->attrs().begin(); iter != node->attrs().end(); ++iter) {
    if (iter->second.has_func()) {
      node_funcs.push_back(iter->second.func().name());
      std::vector<string> func_name_stack;
      func_name_stack.clear();
      func_name_stack.push_back(iter->second.func().name());
      while (!func_name_stack.empty()) {
        string func_name = func_name_stack.back();
        func_name_stack.pop_back();
        const FunctionDef *fdef = flib_def->Find(func_name);
        if (fdef == nullptr) {
          continue;
        }
        for (const NodeDef &ndef : fdef->node_def()) {
          for (auto &item : ndef.attr()) {
            if (item.second.has_func()) {
              node_funcs.push_back(item.second.func().name());
              func_name_stack.push_back(item.second.func().name());
              continue;
            }
          }
        }
      }
      continue;
    }
  }
  return !node_funcs.empty();
}

void DpTfToGEConversionPassImpl::GetTopoEndsNodes(std::vector<Node *> &topo_ends) const {
  for (Node *node : graph_->op_nodes()) {
    if (IsMakeIteratorNode(node) && !IsWithoutNpuScope(node)) {
      auto iter = std::find_if(node->in_nodes().begin(), node->in_nodes().end(),
                               [this](const Node *in_node) { return IsIteratorNode(in_node); });
      if (iter != node->in_nodes().end()) {
        topo_ends.push_back(node);
        ADP_LOG(INFO) << "Insert topo end node " << node->name();
      }
    }
  }
}

Status DpTfToGEConversionPassImpl::AddDataTransDatasets(Node *topo_end, std::string &host_channel_name,
                                                        std::string &device_channel_name,
                                                        const std::map<std::string, std::string> &all_options) {
  const Edge *tmp_edge = nullptr;
  Status ret = GetSplitEdges(topo_end, split_edges_[topo_end], tmp_edge);
  if (!ret.ok()) {
    return ret;
  }

  // Start optimize graph
  // Insert Host and Device queue
  ADP_LOG(INFO) << "Start to add host and device queue on split edges";
  ret = InsertChannelQueue(topo_end, host_channel_name, device_channel_name, all_options);
  if (!ret.ok()) {
    return ret;
  }
  ADP_LOG(INFO) << "host queue name is " << host_channel_name << ", device queue name is " << device_channel_name;

  if (!device_channel_name.empty()) {
    RemoveSplitEdges(topo_end);
  }
  return ret;
}

Status DpTfToGEConversionPassImpl::BuildDeviceDpGraph(const Node *topo_end, Graph *device_graph,
                                                      const std::string &device_channel_name) const {
  // Make a copy of graph for pruned GE
  ADP_LOG(INFO) << "Start to prune GE graph";
  CopyGraph(*graph_, device_graph);
  // Prune visiable GE graph
  std::unordered_set<const Node *> visiable_ge;
  auto iter =
      std::find_if(device_graph->op_nodes().begin(), device_graph->op_nodes().end(),
                   [this, &topo_end](const Node *n) { return IsMakeIteratorNode(n) && n->name() == topo_end->name(); });
  if (iter != device_graph->op_nodes().end()) {
    (void) visiable_ge.emplace(*iter);
  }
  Status ret = RemoveNotSupportDataset(device_graph, device_channel_name, topo_end->name());
  if (!ret.ok()) {
    return ret;
  }

  ADP_LOG(INFO) << "Start to to PruneForReverseReachability.";
  (void) PruneForReverseReachability(device_graph, visiable_ge);
  return ret;
}

Status DpTfToGEConversionPassImpl::AddAttr2DeviceNodes(const Node *topo_end, const Graph *device_graph) const {
  std::string iterator_name;
  for (auto in_node : topo_end->in_nodes()) {
    REQUIRES_NOT_NULL(in_node);
    ADP_LOG(INFO) << "in_node name is " << in_node->name();
    if (IsIteratorNode(in_node)) {
      iterator_name = in_node->name();
      ADP_LOG(INFO) << "iterator name is " << iterator_name;
      break;
    }
  }
  if (iterator_name.empty()) {
    ADP_LOG(ERROR) << "There is no connection between MakeIteraotr and IteratorV2";
    return errors::Internal("There is no connection between MakeIteraotr and IteratorV2");
  }
  // Add dp custom kernel label
  std::string mark_name("mark_name_" + GetRandomName());
  for (auto node : device_graph->nodes()) {
    REQUIRES_NOT_NULL(node);
    if (node->type_string() == "DeviceQueueDataset") {
      node->AddAttr(DP_ITERATOR_MARK, mark_name);
    }
    if (std::find(CUSTOMIZE_DATASET_LIST.begin(), CUSTOMIZE_DATASET_LIST.end(), node->type_string()) !=
        CUSTOMIZE_DATASET_LIST.end()) {
      ADP_LOG(INFO) << node->name() << " is " << node->type_string() << ", need to add label.";
      node->AddAttr("_kernel", "dp");
      node->AddAttr(DP_ITERATOR_MARK, mark_name);
    }
  }
  return Status::OK();
}

Status DpTfToGEConversionPassImpl::AddGeopNodeFunctionDef(FunctionDefLibrary &fdeflib, const std::string &fn_geop,
                                                          const std::string &fn_dpop,
                                                          const string &default_device) const {
  // Add DPOP node(visable only by function of geop)
  string func_def_str;
  (void) fdeflib.SerializeToString(&func_def_str);

  // DPOP node should created by function of geop
  ADP_LOG(INFO) << "Start to convert dpop node to geop function";
  FunctionDef *fd = fdeflib.add_function();
  REQUIRES_NOT_NULL(fd);
  REQUIRES_NOT_NULL(fd->mutable_signature());
  fd->mutable_signature()->set_name(fn_geop);
  NodeDef *n = fd->add_node_def();
  REQUIRES_NOT_NULL(n);
  NameAttrList f_attr;
  f_attr.set_name(fn_dpop);
  *f_attr.mutable_attr() = n->attr();
  TF_CHECK_OK(NodeDefBuilder(fn_dpop, "DPOP")
                  .Input(EMPTY_DEF_INPUT)  // No partition dp_init graph on GE
                  .Device(default_device)
                  .Attr("function", f_attr)  // dpop funcion
                  .Attr("func_def", func_def_str)
                  .Attr("Tin", EMPTY_TYPE)
                  .Attr("Tout", EMPTY_TYPE)
                  .Attr("Tout", EMPTY_TYPE)
                  .Finalize(n));  // n is created by function of geop function
  return Status::OK();
}

Status
DpTfToGEConversionPassImpl::AddGeopDatasetFunctionDef(FunctionDefLibrary &fdeflib, const std::string &fn_geop,
                                                      const std::string &fn_geop_dataset, const string &default_device,
                                                      const std::map<std::string, std::string> &all_options) const {
  // GEOP node should created by function of geopDataset
  ADP_LOG(INFO) << "Start to convert geop node to geopdataset function";
  FunctionDef *fd = fdeflib.add_function();
  REQUIRES_NOT_NULL(fd);
  REQUIRES_NOT_NULL(fd->mutable_signature());
  fd->mutable_signature()->set_name(fn_geop_dataset);

  NodeDef *n = fd->add_node_def();
  REQUIRES_NOT_NULL(n);
  NameAttrList f_attr;
  f_attr.set_name(fn_geop);
  *f_attr.mutable_attr() = n->attr();
  TF_CHECK_OK(NodeDefBuilder(GetRandomName("GeOp"), "GeOp")
                  .Input(EMPTY_DEF_INPUT)  // No partition dp_init graph on GE
                  .Device(default_device)
                  .Attr("function", f_attr)  // geop funcion
                  .Attr("Tin", EMPTY_TYPE)
                  .Attr("Tout", EMPTY_TYPE)
                  .Attr("Tout", EMPTY_TYPE)
                  .Attr("_enableDP", true)
                  .Finalize(n));  // n is created by function of geopDataset function
  std::string attr_name;
  for (auto option : all_options) {
    attr_name = std::string("_") + option.first;
    AddNodeAttr(attr_name, option.second, n);
  }
  AddNodeAttr("_NpuOptimizer", "NpuOptimizer", n);
  return Status::OK();
}

Status
DpTfToGEConversionPassImpl::BuildGeOpDatasetFunction(FunctionDefLibrary &fdeflib, Graph *device_graph,
                                                     const std::string &fn_geop_dataset, const string &default_device,
                                                     const std::map<std::string, std::string> &all_options) const {
  // Convert GE graph to GEOP function body
  Status ret;
  std::string fn_dpop = GetRandomName("dpop_function");
  {
    ADP_LOG(INFO) << "Start to convert GE graph to geop function";
    FunctionDef *fd = fdeflib.add_function();
    ret = GraphToFunctionDef(*device_graph, fn_dpop, fd);
    if (!ret.ok()) {
      ADP_LOG(ERROR) << "GraphToFunctionDef failed:" << ret.error_message();
      return ret;
    }
  }
  std::string fn_geop = GetRandomName("geop_function");
  ret = AddGeopNodeFunctionDef(fdeflib, fn_geop, fn_dpop, default_device);
  if (!ret.ok()) {
    return ret;
  }
  ret = AddGeopDatasetFunctionDef(fdeflib, fn_geop, fn_geop_dataset, default_device, all_options);
  if (!ret.ok()) {
    return ret;
  }
  return ret;
}

Status DpTfToGEConversionPassImpl::AddGeOpDatasetFunctionLibrary(
    FunctionLibraryDefinition *flib, const Node *topo_end, const std::string &device_channel_name,
    const std::string &fn_geop_dataset, const std::map<std::string, std::string> &all_options) {
  FunctionDefLibrary fdeflib;
  if (device_channel_name.empty()) {
    // GEOP node should created by function of geopDataset
    ADP_LOG(INFO) << "No Dataset node can be computed in device, GeOpDataset func is null.";
    FunctionDef *fd = fdeflib.add_function();
    REQUIRES_NOT_NULL(fd);
    REQUIRES_NOT_NULL(fd->mutable_signature());
    fd->mutable_signature()->set_name(fn_geop_dataset);
  } else {
    // Make a copy of graph for pruned GE
    ADP_LOG(INFO) << "Start to prune GE graph";
    std::unique_ptr<Graph> device_graph(new (std::nothrow) Graph(OpRegistry::Global()));
    REQUIRES_NOT_NULL(device_graph);
    Status ret = BuildDeviceDpGraph(topo_end, device_graph.get(), device_channel_name);
    if (!ret.ok()) {
      return ret;
    }

    // add function_def begin
    ADP_LOG(INFO) << "Start to add function_def for GEOP's func";
    for (auto node : device_graph->nodes()) {
      std::vector<string> node_funcs;
      if (GetNodeFuncs(flib, node, node_funcs)) {
        REQUIRES_NOT_NULL(flib);
        ADP_LOG(INFO) << "Node [" << node->name() << "] has func:";
        for (const auto &func : node_funcs) {
          FunctionDef *fdef = fdeflib.add_function();
          REQUIRES_NOT_NULL(flib->Find(func));
          *fdef = *(flib->Find(func));
        }
      }
    }
    ret = AddAttr2DeviceNodes(topo_end, device_graph.get());
    if (!ret.ok()) {
      return ret;
    }

    const string kDefaultDevice = topo_end->def().device();
    ret = BuildGeOpDatasetFunction(fdeflib, device_graph.get(), fn_geop_dataset, kDefaultDevice, all_options);
    if (!ret.ok()) {
      return ret;
    }
  }

  // Update graph function libray
  ADP_LOG(INFO) << "Start to add geop and geopdataset function in graph library";
  // Not a must, just for Tensorbord viewing convenience
  (void) graph_->AddFunctionLibrary(fdeflib);
  (void) flib->AddLibrary(fdeflib);

  return Status::OK();
}

Status DpTfToGEConversionPassImpl::AddGeOpDatasetAndDpGroupDataset(const Node *topo_end,
                                                                   const std::string &fn_geop_dataset,
                                                                   const std::string &host_channel_name,
                                                                   const std::string &device_channel_name) const {
  // Add GEOPDataset node to graph_
  std::vector<const Edge *> topo_end_input_edges(topo_end->in_edges().begin(), topo_end->in_edges().end());

  ADP_LOG(INFO) << "Start to add geopdataset node in graph";
  const Node *iterator_node = nullptr;
  for (const Edge *e : topo_end_input_edges) {
    REQUIRES_NOT_NULL(e);
    if (IsIteratorNode(e->src())) {
      iterator_node = e->src();
    }
  }

  // Combine all host queue dataset with GEOPDataset
  std::vector<NodeBuilder::NodeOut> inputs;
  std::unordered_set<Node *> isolated_nodes;
  for (Node *n : graph_->op_nodes()) {
    REQUIRES_NOT_NULL(n);
    // host tf makeiterator add dp label
    if (IsMakeIteratorNode(n)) {
      n->AddAttr("_kernel", "dp");
    }
    if (n->type_string() == "HostQueueDataset" && n->name() == host_channel_name) {
      // 0: Host queue always generate one dataset
      ADP_LOG(INFO) << "inputs add node : name is " << n->name() << ", op is " << n->type_string();
      inputs.emplace_back(NodeBuilder::NodeOut(n, 0));
    }
    if (n->type_string() == "DeviceQueueDataset" && n->name() == device_channel_name) {
      (void) isolated_nodes.insert(n);
    }
  }

  Node *dpgroup_dataset_node = nullptr;
  REQUIRES_NOT_NULL(iterator_node);
  auto m_src = iterator_node->def().attr();
  TF_CHECK_OK(NodeBuilder(GetRandomName("DPGroupDataset"), "DPGroupDataset")
                  .Input(inputs)  // All host queue flow into geopDataset for driver
                  .Device(iterator_node->def().device())
                  .Attr("output_types", m_src["output_types"])
                  .Attr("output_shapes", m_src["output_shapes"])
                  .Finalize(graph_,
                            &dpgroup_dataset_node));  // Finalize geopDataset in graph_

  NameAttrList f_attr;
  f_attr.set_name(fn_geop_dataset);
  Node *geop_dataset_node = nullptr;
  TF_CHECK_OK(NodeBuilder(GetRandomName("GeopDataset"), "GEOPDataset")
                  .Device(iterator_node->def().device())
                  .Attr("f", f_attr)  // geopDataset function
                  .Finalize(graph_,
                            &geop_dataset_node));  // Finalize geopDataset in graph_

  for (Node *n : graph_->op_nodes()) {
    if (n->type_string() == "HostQueueDataset" && n->name() == host_channel_name) {
      graph_->RemoveEdge(*(n->in_edges().begin()));
      (void) graph_->AddEdge(geop_dataset_node, 0, n, 0);
    }
  }
  // Remove all edges flow to MakeIterator except the one from IteratorV2
  ADP_LOG(INFO) << "Start to combine geopdataset with iterator node and remove "
                   "orignal edges";

  // We must copy all topoend input edges as we can't modify it when combine
  // geopdataset an topoend
  for (const Edge *e : topo_end_input_edges) {
    ADP_LOG(INFO) << "node:" << topo_end->name() << ", input node is:" << e->src()->name();
    if (!IsIteratorNode(e->src())) {
      (void) CHECK_NOTNULL(graph_->AddEdge(dpgroup_dataset_node, 0, e->dst(), e->dst_input()));
      ADP_LOG(INFO) << "Remove_" << GetEdgeName(e);
      graph_->RemoveEdge(e);
    }
  }

  // Prune for the final optimized graph
  ADP_LOG(INFO) << "Start to prune final optimized graph.";

  (void) RemoveIsolatedNode(graph_, isolated_nodes);
  ADP_LOG(INFO) << "Start to assign unassigned node on default device.";
  // We do pass after assign, so we must assign all new added nodes
  for (Node *n : graph_->op_nodes()) {
    if (n->assigned_device_name().empty()) {
      // Use device of MakeIterator node as default
      n->set_assigned_device_name(iterator_node->def().device());
      ADP_LOG(INFO) << "Assigned node [" << n->name() << "] on device [" << n->assigned_device_name() << "]";
    }
  }
  return Status::OK();
}

void DpTfToGEConversionPassImpl::AddOptionAttr(std::vector<Node *> nodes, const std::map<std::string, std::string> &all_options) {
  std::string attr_name;
  for (auto node : nodes) {
    ADP_LOG(INFO) << "Node[" << node->name() << "] add options attr.";
    for (const auto &option : all_options) {
      attr_name = std::string("_") + option.first;
      node->AddAttr(attr_name, option.second);
    }
    node->AddAttr("_NpuOptimizer", "NpuOptimizer");
  }
}

bool DpTfToGEConversionPassImpl::RunPass(const std::unique_ptr<Graph> *g, FunctionLibraryDefinition *flib,
                                         const std::map<std::string, std::string> &all_options) {
  ADP_LOG(INFO) << ">>>> DpTfToGEConversionPassImpl::RunPass <<<<";
  // Convert just for convenient access
  split_edges_.clear();
  graph_ = &**g;
  flib_def_ = &(*g)->flib_def();

  // Find split edges from subgraphs, which MakeIterator connect to Itearator op
  std::vector<Node *> topo_ends;
  GetTopoEndsNodes(topo_ends);
  // After traversal, topo_ends should store MakeIterator Nodes.
  if (topo_ends.empty()) {
    ADP_LOG(INFO) << "Do not find MakeIterator <- IteratorV2 connects in the graph,"
                  << " pass datapreprocess pass.";
    return true;
  }

  if (kDumpGraph) {
    GraphDef before_graphdef;
    (*g)->ToGraphDef(&before_graphdef);
    string pre_model_path = GetDumpPath() + "BeforeSubGraph_dp_";
    string pmodel_path = pre_model_path + std::to_string(graph_run_num_) + ".pbtxt";
    TF_DO_CHECK_OK(WriteTextProto(Env::Default(), pmodel_path, before_graphdef), ERROR);
  }

  ADP_LOG(INFO) << "Start to optimize dp_init topological graph.";
  for (Node *topo_end : topo_ends) {
    // Get all edges that should be replace with HostQueue->DeviceQueue
    ADP_LOG(INFO) << "Start to find split edges, topo_end node is : " << topo_end->name() << ", op is "
                  << topo_end->type_string();
    std::string host_channel_name;
    std::string device_channel_name;
    TF_DO_CHECK_OK(AddDataTransDatasets(topo_end, host_channel_name, device_channel_name, all_options), ERROR);
    std::string fn_geop_dataset = GetRandomName("geopdataset_function");
    TF_DO_CHECK_OK(AddGeOpDatasetFunctionLibrary(flib, topo_end, device_channel_name, fn_geop_dataset, all_options),
                   ERROR);
    TF_DO_CHECK_OK(AddGeOpDatasetAndDpGroupDataset(topo_end, fn_geop_dataset, host_channel_name, device_channel_name),
                   ERROR);
  }

  ADP_LOG(INFO) << "End optimize dp_init topological graph";
  if (kDumpGraph) {
    GraphDef after_graphdef;
    (*g)->ToGraphDef(&after_graphdef);
    string suffix_model_path = GetDumpPath() + "AfterSubGraph_dp_";
    string smodel_path = suffix_model_path + std::to_string(graph_run_num_) + ".pbtxt";
    TF_DO_CHECK_OK(WriteTextProto(Env::Default(), smodel_path, after_graphdef), ERROR);
  }
  return true;
}

bool DpTfToGEConversionPassImpl::RemoveIsolatedNode(Graph *g, std::unordered_set<Node *> visited) const {
  // Compute set of nodes that we need to traverse in order to reach
  // the nodes in "nodes" by performing a breadth-first search from those
  // nodes, and accumulating the visited nodes.
  std::deque<Node *> queue;
  for (Node *n : visited) {
    ADP_LOG(INFO) << "Reverse reach init: " << n->name();
    queue.push_back(n);
  }
  while (!queue.empty()) {
    Node *n = queue.front();
    queue.pop_front();
    for (Node *out : n->out_nodes()) {
      if (visited.insert(out).second) {
        queue.push_back(out);
        ADP_LOG(INFO) << "Reverse reach : " << n->name() << " from " << out->name();
      }
    }
  }

  // Make a pass over the graph to remove nodes in "visited"
  std::vector<Node *> all_nodes;
  all_nodes.reserve(static_cast<size_t>(g->num_nodes()));
  for (Node *n : g->nodes()) {
    all_nodes.push_back(n);
  }

  bool any_removed = false;
  for (Node *n : all_nodes) {
    if (visited.count(n) != 0) {
      g->RemoveNode(n);
      any_removed = true;
    }
  }

  return any_removed;
}

Status DpTfToGEConversionPassImpl::Run(const GraphOptimizationPassOptions &options) {
  if ((options.graph == nullptr && options.partition_graphs == nullptr) || options.flib_def == nullptr) {
    return Status::OK();
  }

  Status s = Status::OK();
  if (options.graph != nullptr) {
    std::unique_ptr<Graph> *graph = options.graph;
    FunctionLibraryDefinition *func_lib = options.flib_def;
    s = ProcessGraph(graph, func_lib, OptimizationPassRegistry::POST_REWRITE_FOR_EXEC);
    if (s != Status::OK()) {
      return s;
    }
  } else if (options.partition_graphs != nullptr) {
    for (auto &pg : *options.partition_graphs) {
      std::unique_ptr<Graph> *graph = &pg.second;
      FunctionLibraryDefinition *func_lib = options.flib_def;
      s = ProcessGraph(graph, func_lib, OptimizationPassRegistry::POST_PARTITIONING);
      if (s != Status::OK()) {
        return s;
      }
    }
  }

  return Status::OK();
}

static std::atomic<int> graph_run_num(1);
static mutex graph_num_mutex(LINKER_INITIALIZED);
Status DpTfToGEConversionPass::Run(const GraphOptimizationPassOptions &options) {
  return DpTfToGEConversionPassImpl().Run(options);
}

Status DpTfToGEConversionPassImpl::ProcessGraph(const std::unique_ptr<Graph> *graph,
                                                FunctionLibraryDefinition *func_lib,
                                                const OptimizationPassRegistry::Grouping pass_group_value) {
  int64 startTime = InferShapeUtil::GetCurrentTimestap();

  graph_run_num_ = graph_run_num++;

  if (graph == nullptr) {
    return Status::OK();
  }

  std::string channel_name;
  for (Node *n : graph->get()->nodes()) {
    REQUIRES_NOT_NULL(n);
    if (n->type_string() == "Iterator" || n->type_string() == "IteratorV2") {
      channel_name = n->name();
    }
    if (n->attrs().Find("_NoNeedOptimize")) {
      ADP_LOG(INFO) << "Found mark of noneed optimize on node [" << n->name() << "], skip DpTfToGEConversionPass.";
      return Status::OK();
    }
  }
  NpuAttrs::SetUseAdpStatus(channel_name, false);

  std::map<std::string, std::string> all_options;
  std::map<std::string, std::string> pass_options;
  pass_options = NpuAttrs::GetDefaultPassOptions();
  std::vector<Node *> add_options_nodes;

  for (Node *n : graph->get()->nodes()) {
    REQUIRES_NOT_NULL(n);
    if (n->type_string() == "DvppDataset") {
      uint32_t device_id = 0;
      (void) GetEnvDeviceID(device_id);
      n->AddAttr("queue_name", "device" + std::to_string(device_id) + "_" + channel_name);
      NpuAttrs::SetUseAdpStatus(channel_name, true);
      ADP_LOG(INFO) << "The graph include DvppDataset, set channel_name:" << channel_name
                    << ", skip DpTfToGEConversionPass.";
      return Status::OK();
    }
    if (n->attrs().Find("_NpuOptimizer")) {
      pass_options = NpuAttrs::GetPassOptions(n->attrs());
      all_options = NpuAttrs::GetAllAttrOptions(n->attrs());
    }

    if (IsAddOptionAttrNode(n)) {
      add_options_nodes.push_back(n);
    }
  }

  AddOptionAttr(add_options_nodes, all_options);

  std::string job = pass_options["job"];
  if (job == "ps" || job == "default") {
    ADP_LOG(INFO) << "job is " << job << " Skip the optimizer : DpTfToGEConversionPass.";
    return Status::OK();
  }
  if (job == "localhost" && pass_group_value != OptimizationPassRegistry::POST_REWRITE_FOR_EXEC) {
    return Status::OK();
  }
  if (job != "localhost" && pass_group_value != OptimizationPassRegistry::POST_PARTITIONING) {
    return Status::OK();
  }

  bool enableDP = (pass_options["enable_dp"] == "1");
  bool use_off_line = (pass_options["use_off_line"] == "1");
  bool do_npu_optimizer = (pass_options["do_npu_optimizer"] == "1");
  if (do_npu_optimizer) {
    if (!use_off_line) {
      ADP_LOG(INFO) << "Run online process and skip the optimizer";
      return Status::OK();
    }
  } else {
    return Status::OK();
  }

  if (!enableDP) {
    ADP_LOG(INFO) << "DpTfToGEConversionPassImpl::RunPass, enable data preproc is false";
    return Status::OK();
  }
  auto process_graph = [this](const std::unique_ptr<Graph> *g, FunctionLibraryDefinition *flib,
                              const std::map<std::string, std::string> &all_options) {
    (void) RunPass(g, flib, all_options);
  };

  // For any pre-partitioning phase, graph is stored in options.graph.
  process_graph(graph, func_lib, all_options);
  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "DpTfToGEConversionPassImpl Run success. [" << ((endTime - startTime) / kMicrosToMillis) << " ms].";

  return Status::OK();
}

// We register DpTfToGE insertion for phase 102 in POST_PARTITIONING grouping
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 4, DpTfToGEConversionPass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 102, DpTfToGEConversionPass);
}  // namespace tensorflow
