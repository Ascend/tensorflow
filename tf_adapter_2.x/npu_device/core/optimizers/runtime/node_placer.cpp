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
#include "npu_utils.h"
#include "op_executors/npu_concrete_graph.h"
#include "optimizers/runtime/node_placer.h"

namespace npu {
const std::string kSharedGroup = "_shared_group_id";

Cluster::Cluster(const NodePlacer *placer, tensorflow::Node *node, Placement place)
    : id(node->id()), placement(place), placer_(placer) {
  static std::atomic_int64_t index{0};
  name = kPlacementString[placement] + "_cluster_" + std::to_string(index.fetch_add(1));
  DLOG() << "Create cluster " << name << " for " << node->name();
  (void)Merge(node);
}

bool Cluster::Merge(tensorflow::Node *node) {
  if (!nodes.insert(node).second) {
    return false;
  }
  DLOG() << "Place node " << node->name() << " in cluster " << this->name;
  (void)in_nodes.erase(node);
  (void)out_nodes.erase(node);
  for (auto n : node->in_nodes()) {
    if (nodes.count(n) == 0) {
      (void)in_nodes.insert(n);
    }
  }
  for (auto n : node->out_nodes()) {
    if (nodes.count(n) == 0) {
      (void)out_nodes.insert(n);
    }
  }
  return true;
}

void Cluster::Merge(const std::shared_ptr<Cluster> other) {
  for (auto node : other->nodes) {
    (void)Merge(node);
  }
}

tensorflow::Status NodePlacer::Apply(size_t depth) {
  const static size_t kMaxRecursionDepth = 16;
  NPU_REQUIRES(depth <= kMaxRecursionDepth, tensorflow::errors::Unimplemented(
                                              "Recursion depth exceed 16 when assign subgraph node device placement"));
  NPU_REQUIRES_OK(CopyShareableNode());
  InitNodeTopo();
  NPU_REQUIRES_OK(DeterminedSurelyNodes());  // Determine surely node placement
  NPU_REQUIRES_OK(SpreadCpuNode());          // Place node can never place on npu
  NPU_REQUIRES_OK(BuildConcreteCluster());   // Mark concrete nodes
  NPU_REQUIRES_OK(SpreadNpuNode());          // Place node on npu
  NPU_REQUIRES_OK(MergeCopiedSharedNodes());
  NPU_REQUIRES_OK(BuildNpuOp());
  NPU_REQUIRES_OK(PlaceCpuNodeSubgraphs(depth));
  return tensorflow::Status::OK();
}

void NodePlacer::InitNodeTopo() {
  uint64_t topo = 0U;
  auto leave = [this, &topo](const tensorflow::Node *node) { node_topo_[node] = topo++; };
  tensorflow::ReverseDFS(*graph_, {}, leave);
}

void NodePlacer::ResetNodeMask() {
  node_mask_.resize(graph_->num_node_ids());
  (void)memset(node_mask_.data(), 0, node_mask_.size() * sizeof(uint8_t));
}

bool NodePlacer::FetchSetMask(const NodeOrCluster &node_or_cluster) {
  auto &mask = node_mask_[node_or_cluster.Id()];
  if (mask == 0U) {
    mask = 1U;
    return false;
  }
  return true;
}

bool NodePlacer::FetchClearMask(const NodeOrCluster &node_or_cluster) {
  auto &mask = node_mask_[node_or_cluster.Id()];
  if (mask != 0U) {
    mask = 0U;
    return true;
  }
  return false;
}

tensorflow::Status NodePlacer::CopyShareableNode() {
  std::vector<tensorflow::Node *> shared_nodes;
  for (auto node : graph_->nodes()) {
    if (node->IsConstant() || IsSubstituteNode(*node)) {
      shared_nodes.emplace_back(node);
    }
  }

  int shared_id = 0;
  for (auto node : shared_nodes) {
    std::vector<const tensorflow::Edge *> edges;
    int64_t i = 0;
    for (auto edge : node->out_edges()) {
      if (!edge->dst()->IsOp()) continue;
      if (i++ > 0) {
        edges.push_back(edge);
      }
    }
    if (edges.empty()) {
      continue;
    }
    node->AddAttr(kSharedGroup, shared_id++);
    for (auto edge : edges) {
      tensorflow::Status status = tensorflow::Status::OK();
      DLOG() << "Copy node " << node->name() << " for colocate with " << edge->dst()->name();
      auto copy = graph_->AddNode(node->def(), &status);
      NPU_REQUIRES_OK(status);
      (void)graph_->AddEdge(copy, edge->src_output(), edge->dst(), edge->dst_input());
      graph_->RemoveEdge(edge);
      for (auto in_edge : node->in_edges()) {
        (void)graph_->AddEdge(in_edge->src(), in_edge->src_output(), copy, in_edge->dst_input());
      }
    }
  }
  return tensorflow::Status::OK();
}

std::set<std::shared_ptr<Cluster>> NodePlacer::GetNpuClusters() {
  std::set<std::shared_ptr<Cluster>> clusters;
  for (auto node : graph_->op_nodes()) {
    auto iter = npu_clusters_.find(node);
    if (iter == npu_clusters_.end()) {
      continue;
    }
    (void)clusters.insert(iter->second);
  }
  return clusters;
}

bool NodePlacer::IsNpuMeaningLessNode(const tensorflow::Node *node) {
  const static std::unordered_set<std::string> kNpuMeaningLessNodes{"Identity", "NoOp", "Const"};
  return kNpuMeaningLessNodes.count(node->type_string()) != 0U;
}

tensorflow::Status NodePlacer::BuildNpuOp() {
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context_)->FuncLibDef();
  auto clusters = GetNpuClusters();
  for (auto &cluster : clusters) {
    if (!IsClusterMustPlaceOnNpu(*cluster)) {
      const static size_t kMinClusterNodeNum = 1U;
      if (cluster->nodes.size() < kMinClusterNodeNum) {
        DLOG() << "Skip build npu op for cluster " << cluster->name << " as node num " << cluster->nodes.size() << " < "
               << kMinClusterNodeNum;
        continue;
      } else if (std::all_of(cluster->nodes.begin(), cluster->nodes.end(), IsNpuMeaningLessNode)) {
        DLOG() << "Skip build npu op for cluster " << cluster->name << " as all node is npu meaningless";
        continue;
      }
    }
    DLOG() << "Start building npu op for cluster " << cluster->name;
    std::set<tensorflow::Node *, StableNodeCompartor> control_inputs;
    std::set<tensorflow::Node *, StableNodeCompartor> control_outputs;

    std::vector<tensorflow::NodeBuilder::NodeOut> inputs;
    std::vector<tensorflow::DataType> input_types;
    std::vector<tensorflow::DataType> output_types;

    std::vector<const tensorflow::Edge *> input_edges;
    std::vector<const tensorflow::Edge *> output_edges;

    std::vector<tensorflow::Node *> nodes(cluster->nodes.cbegin(), cluster->nodes.cend());
    std::sort(nodes.begin(), nodes.end(), StableNodeCompartor{});

    auto cluster_graph = std::make_unique<tensorflow::Graph>(tensorflow::OpRegistry::Global());
    std::map<tensorflow::Node *, tensorflow::Node *> node_map;
    for (auto &node : nodes) {
      tensorflow::Status status;
      node_map[node] = cluster_graph->AddNode(node->def(), &status);
      NPU_REQUIRES_OK(status);
    }
    for (auto &node : nodes) {
      for (auto edge : node->in_edges()) {
        if (cluster->nodes.count(edge->src()) == 0) {
          if (edge->IsControlEdge()) {
            DLOG() << "Collect control input " << edge->src()->name() << " of cluster " << cluster->name;
            (void)control_inputs.insert(edge->src());
          } else {
            DLOG() << "Collect input edge " << edge->DebugString() << " of cluster " << cluster->name;
            input_edges.push_back(edge);
            inputs.emplace_back(tensorflow::NodeBuilder::NodeOut{edge->src(), edge->src_output()});
            input_types.emplace_back(EdgeDataType(*edge));
          }
        } else {
          auto e = cluster_graph->AddEdge(node_map[edge->src()], edge->src_output(), node_map[node], edge->dst_input());
          DLOG() << "Add cluster inner edge " << e->DebugString() << " of cluster " << cluster->name;
        }
      }
      for (auto edge : node->out_edges()) {
        if (cluster->nodes.count(edge->dst()) == 0) {
          if (edge->IsControlEdge()) {
            DLOG() << "Collect control output " << edge->src()->name() << " of cluster " << cluster->name;
            (void)control_outputs.insert(edge->dst());
          } else {
            DLOG() << "Collect output edge " << edge->DebugString() << " of cluster " << cluster->name;
            output_edges.push_back(edge);
            output_types.emplace_back(EdgeDataType(*edge));
          }
        }
      }
    }

    std::string fn = "npu_f_" + cluster->name;
    tensorflow::NameAttrList func;
    func.set_name(fn);
    tensorflow::Node *npu_op;
    NPU_REQUIRES_OK(tensorflow::NodeBuilder(cluster->name, "NpuCall")
                      .ControlInputs(std::vector<tensorflow::Node *>(control_inputs.begin(), control_inputs.end()))
                      .Input(inputs)
                      .Attr("Tin", input_types)
                      .Attr("Tout", output_types)
                      .Attr("f", func)
                      .Attr("device", device_->device_id)
                      .Finalize(graph_, &npu_op));
    DLOG() << "Add npu op " << npu_op->name() << " for cluster " << cluster->name;

    for (auto node : control_outputs) {
      DLOG() << "Add control edge from " << npu_op->name() << " to " << node->name() << " of root graph of "
             << cluster->name;
      (void)graph_->AddControlEdge(npu_op, node);
    }

    for (size_t i = 0U; i < input_edges.size(); i++) {
      auto &edge = input_edges[i];
      tensorflow::Node *arg;
      NPU_REQUIRES_OK(tensorflow::NodeBuilder(tensorflow::strings::StrCat(cluster->name, "_input", i), "_Arg")
                        .Attr("index", int64_t(i))
                        .Attr("T", input_types[i])
                        .Finalize(cluster_graph.get(), &arg));
      auto e = cluster_graph->AddEdge(arg, 0, node_map[edge->dst()], edge->dst_input());
      DLOG() << "Add input edge " << e->DebugString() << " of cluster graph of " << cluster->name;
    }
    for (size_t i = 0U; i < output_edges.size(); i++) {
      auto &edge = output_edges[i];
      tensorflow::Node *ret;
      NPU_REQUIRES_OK(tensorflow::NodeBuilder(tensorflow::strings::StrCat(cluster->name, "_output", i), "_Retval")
                        .Input(node_map[edge->src()], edge->src_output())
                        .Attr("index", int64_t(i))
                        .Attr("T", output_types[i])
                        .Finalize(cluster_graph.get(), &ret));
      auto e = graph_->AddEdge(npu_op, static_cast<int32_t>(i), edge->dst(), edge->dst_input());
      DLOG() << "Add output edge " << e->DebugString() << " of root graph of " << cluster->name;
    }

    for (auto node : nodes) {
      graph_->RemoveNode(node);
    }

    (void)tensorflow::FixupSourceAndSinkEdges(cluster_graph.get());
    tensorflow::FunctionDefLibrary flib;
    OptimizeStageGraphDumper dumper(cluster->name + "." + fn);
    dumper.DumpWithSubGraphs("NPU_FUNCTION", cluster_graph->ToGraphDefDebug(), *lib_def);
    NPU_REQUIRES_OK(tensorflow::GraphToFunctionDef(*cluster_graph, fn, flib.add_function()));
    NPU_REQUIRES_OK(lib_def->AddLibrary(flib));
  }

  auto merged = MergeCopiedSharedNodes({graph_->op_nodes().begin(), graph_->op_nodes().end()});
  for (auto node : merged) {
    DLOG() << "Remove copied shared node " << node->name() << " in root graph";
    graph_->RemoveNode(node);
  }

  (void)tensorflow::FixupSourceAndSinkEdges(graph_);
  return tensorflow::Status::OK();
}

tensorflow::Status NodePlacer::PlaceCpuNodeSubgraphs(size_t depth) const {
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context_)->FuncLibDef();
  for (auto node : graph_->op_nodes()) {
    if (node->type_string() == "NpuCall") {  // Nodes placed on cpu except npu call
      continue;
    }
    auto fns = GetNodeSubgraph(*node);
    for (auto &fn : fns) {
      DLOG() << "Device placement for subgraph " << fn << " of cpu node " << node->name();
      std::unique_ptr<tensorflow::FunctionBody> fbody;
      auto &fdef = *lib_def->Find(fn);
      NPU_REQUIRES_OK(FunctionDefToBodyHelper(fdef, tensorflow::AttrSlice{}, lib_def, &fbody));
      PruneGraphByFunctionSignature(fdef, *(fbody->graph), true);
      NPU_REQUIRES_OK(NodePlacer(context_, fbody->graph, device_).Apply(depth + 1));
      tensorflow::FunctionDefLibrary flib;
      OptimizeStageGraphDumper dumper(fn);
      dumper.DumpWithSubGraphs("MIX_FUNCTION", fbody->graph->ToGraphDefDebug(), *lib_def);
      NPU_REQUIRES_OK(tensorflow::GraphToFunctionDef(*fbody->graph, fn, flib.add_function()));
      NPU_REQUIRES_OK(lib_def->RemoveFunction(fn));
      NPU_REQUIRES_OK(lib_def->AddLibrary(flib));
    }
  }
  return tensorflow::Status::OK();
}

bool NodePlacer::IsClusterMustPlaceOnNpu(const Cluster &cluster) {
  for (auto node : cluster.nodes) {
    auto iter = node_placement_.find(node);
    if (iter != node_placement_.end() && iter->second == Placement::NPU) {
      DLOG() << cluster.name << " must place on npu as has determined npu node " << node->name();
      return true;
    }
    if (!GetNodeSubgraph(*node).empty()) {
      DLOG() << cluster.name << " prefer place on npu as has subgraph";
      return true;
    }
  }
  return false;
}

std::vector<tensorflow::Node *> NodePlacer::MergeCopiedSharedNodes(std::vector<tensorflow::Node *> all_nodes) const {
  std::vector<tensorflow::Node *> merged_nodes;
  std::map<uint64_t, std::unordered_set<tensorflow::Node *>> equal_nodes;
  for (auto node : all_nodes) {
    auto attr = node->attrs().Find(kSharedGroup);
    if (attr != nullptr) {
      (void)equal_nodes[attr->i()].insert(node);
    }
  }
  for (auto &item : equal_nodes) {
    auto &nodes = item.second;
    if (nodes.size() == 1U) {
      continue;
    }
    auto keep = *nodes.begin();
    DLOG() << keep->name() << " has " << nodes.size() << " copied nodes";
    for (auto iter = ++nodes.begin(); iter != nodes.end(); (void)(iter++)) {
      auto &node = *iter;
      for (auto edge : node->out_edges()) {
        // Ignore existed edge
        (void)graph_->AddEdge(keep, edge->src_output(), edge->dst(), edge->dst_input());
      }
      merged_nodes.push_back(node);
    }
  }
  return merged_nodes;
}

tensorflow::Status NodePlacer::MergeCopiedSharedNodes() {
  auto clusters = GetNpuClusters();
  for (auto cluster : clusters) {
    auto merged = MergeCopiedSharedNodes(std::vector<tensorflow::Node *>(cluster->nodes.begin(), cluster->nodes.end()));
    for (auto &node : merged) {
      DLOG() << "Merge and remove copied node " << node->name() << " in cluster " << cluster->name;
      (void)npu_clusters_.erase(node);
      graph_->RemoveNode(node);
      (void)cluster->nodes.erase(node);
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status NodePlacer::DeterminedSurelyNodes() {
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context_)->FuncLibDef();
  for (auto node : graph_->nodes()) {
    if (node->IsRetval() || node->IsArg() || !node->IsOp()) {
      DLOG() << "Determined cpu as frame node " << node->name() << " " << node->type_string();
      node_placement_[node] = Placement::CPU;
    } else if (!device_->Supported(node->type_string())) {
      DLOG() << "Determined cpu " << node->name() << " " << node->type_string() << " as npu unsupported";
      node_placement_[node] = Placement::CPU;
    } else if (device_->IsNpuSpecificOp(node->type_string())) {
      DLOG() << "Determined npu " << node->name() << " " << node->type_string() << " as npu specific op";
      node_placement_[node] = Placement::NPU;
    } else if (IsSubstituteNode(*node)) {
      DLOG() << "Determined npu " << node->name() << " " << node->type_string() << " as npu resource";
      node_placement_[node] = Placement::NPU;
    } else {
      std::set<std::string> unsupported_ops;
      NPU_REQUIRES_OK(GetSubgraphUnsupportedOps(*device_, *node, *lib_def, unsupported_ops));
      if (!unsupported_ops.empty()) {
        DLOG() << "Determined cpu " << node->name() << " " << node->type_string()
               << " as npu unsupported subgraph node " << SetToString(unsupported_ops);
        node_placement_[node] = Placement::CPU;
      }
    }
  }
  DLOG() << "Determined " << node_placement_.size() << " nodes, " << GetNodesPlacedOn(Placement::NPU).size() << " npu, "
         << GetNodesPlacedOn(Placement::CPU).size() << " cpu";
  return tensorflow::Status::OK();
}

void NodePlacer::Concrete(tensorflow::Node *src, tensorflow::Node *dst) {
  auto target = GetOrCreateConcreteCluster(dst);
  DLOG() << "Concrete node " << src->name() << " with " << dst->name() << " to cluster " << target->name;

  auto iter = concrete_clusters_.find(src);
  if (iter != concrete_clusters_.end() && iter->second == target) {
    DLOG() << "Node " << src->name() << " has already concrete with " << dst->name() << " in cluster " << target->name;
    return;
  }

  auto visitor = [&target](tensorflow::Node *node) {
    (void)target->Merge(node);
    return true;
  };
  (void)VisitPathNodes(src, dst, visitor);

  if (iter != concrete_clusters_.end()) {
    auto merged_cluster = iter->second;  // Must copy
    target->Merge(merged_cluster);
    for (auto &item : concrete_clusters_) {
      if (item.second == merged_cluster) {
        DLOG() << "Change concrete cluster of " << item.first->name() << " from " << iter->second->name << " to "
               << target->name;
        item.second = target;
      }
    }
  } else {
    (void)target->Merge(src);
    concrete_clusters_[src] = target;
  }
}

tensorflow::Status NodePlacer::BuildConcreteCluster() {
  std::vector<tensorflow::Node *> starts;
  for (auto node : graph_->op_nodes()) {
    if (IsNodePlacedOn(node, Placement::CPU)) {
      continue;
    }
    if (std::any_of(node->out_edges().begin(), node->out_edges().end(),
                    [this](const tensorflow::Edge *edge) { return !IsSupportedNpuBound(*edge); }) &&
        std::all_of(node->in_edges().begin(), node->in_edges().end(),
                    [this](const tensorflow::Edge *edge) { return IsSupportedNpuBound(*edge); })) {
      DLOG() << "Need concrete for start node " << node->name();
      starts.push_back(node);
    }
  }
  for (auto &start : starts) {
    DLOG() << "Concrete from start " << start->name();
    const auto enter = [](tensorflow::Node *node) { DLOG() << "Concrete reach " << node->name(); };
    tensorflow::DFSFrom(*graph_, {start}, enter, {}, {}, [this](const tensorflow::Edge &edge) {
      if (IsSupportedNpuBound(edge)) {
        return false;
      }
      DLOG() << "Forward concrete " << tensorflow::DataTypeString(EdgeDataType(edge)) << " edge " << edge.DebugString();
      Concrete(edge.dst(), edge.src());
      return true;
    });
  }

  std::set<std::shared_ptr<Cluster>> vst_cluster;
  for (auto &item : concrete_clusters_) {
    auto &cluster = item.second;
    if (!vst_cluster.insert(cluster).second) {
      continue;
    }
    std::queue<std::shared_ptr<Cluster>> q;
    for (auto &node : cluster->nodes) {
      auto iter = concrete_clusters_.find(node);
      if (iter != concrete_clusters_.end() && iter->second != cluster) {
        q.push(iter->second);
      }
    }
    std::set<std::shared_ptr<Cluster>> vst{cluster};
    while (!q.empty()) {
      auto path_cluster = q.front();
      q.pop();
      if (!vst.insert(path_cluster).second) {
        continue;
      }
      DLOG() << "Concrete path cluster " << path_cluster->name << " of " << cluster->name;
      for (auto &node : path_cluster->nodes) {
        (void)cluster->Merge(node);
        auto iter = concrete_clusters_.find(node);
        if (iter != concrete_clusters_.end() && iter->second != path_cluster) {
          q.push(iter->second);
        }
      }
    }
  }

  for (auto &start : starts) {
    auto iter = concrete_clusters_.find(start);
    if (iter == concrete_clusters_.end()) {
      continue;
    }
    auto cluster = iter->second;
    auto found = std::find_if(cluster->nodes.cbegin(), cluster->nodes.cend(),
                              [this](tensorflow::Node *node) { return !IsNodeCanPlacedOn(node, Placement::NPU); });
    if (found != cluster->nodes.end()) {
      for (auto iter2 = concrete_clusters_.begin(); iter2 != concrete_clusters_.end();) {
        if (iter2->second == cluster) {
          DLOG() << "Place " << iter2->first->name() << " on cpu as path node " << (*found)->name()
                 << " can not place on npu";
          node_placement_[iter2->first] = Placement::CPU;
          (void)concrete_clusters_.erase(iter2++);
        } else {
          ++iter2;
        }
      }
    }
  }

  return tensorflow::Status::OK();
}

std::vector<tensorflow::Node *> NodePlacer::GetNodesPlacedOn(Placement placement) {
  std::vector<tensorflow::Node *> nodes;
  for (auto node : graph_->nodes()) {
    if (IsNodePlacedOn(node, placement)) {
      nodes.emplace_back(node);
    }
  }
  return nodes;
}

bool NodePlacer::SpreadNpuEdge(const tensorflow::Edge &edge, bool forward) {
  DLOG() << "Npu " << (forward ? "forward" : "backward") << " spread " << edge.DebugString();
  auto pending = edge.src();
  auto target = edge.dst();
  if (forward) {
    std::swap(pending, target);
  }
  if (!IsNodeCanPlacedOn(pending, Placement::NPU)) {
    DLOG() << "Stop spread " << edge.DebugString() << " as " << pending->name() << " placed on cpu";
    return false;
  }
  if (!ColocateNpu(pending, target)) {
    DLOG() << "Re-trigger spread from node " << pending->name() << " in cluster " << npu_clusters_[pending]->name;
  }
  return true;
}

tensorflow::Status NodePlacer::SpreadCpuNode() {
  std::vector<tensorflow::Node *> starts = GetNodesPlacedOn(Placement::CPU);
  if (starts.empty()) {
    DLOG() << "Skip spread cpu as no nodes placed on";
    return tensorflow::Status::OK();
  }
  std::stringstream ss;
  const auto enter = [this, &ss](tensorflow::Node *node) {
    auto iter = node_placement_.emplace(node, Placement::CPU);
    if (iter.second) {
      DLOG() << "Spread place " << node->name() << " on cpu";
    } else {
      if (iter.first->second != Placement::CPU) {
        ss << "Failed spread place " << node->name() << " on cpu as has placed on npu" << std::endl;
      }
    }
  };
  const auto filter = [this](const tensorflow::Edge &edge) {
    if (IsSupportedNpuBound(edge)) {
      return false;
    }
    DLOG() << "Spread cpu " << tensorflow::DataTypeString(EdgeDataType(edge)) << " edge " << edge.DebugString();
    return true;
  };
  size_t cpu_nums = starts.size();
  do {
    cpu_nums = starts.size();
    tensorflow::DFSFrom(*graph_, starts, enter, {}, {}, filter);
    tensorflow::ReverseDFSFrom(*graph_, starts, enter, {}, {}, filter);
    starts = GetNodesPlacedOn(Placement::CPU);
  } while (starts.size() != cpu_nums);
  if (!ss.str().empty()) {
    return tensorflow::errors::Unimplemented(ss.str());
  }
  return tensorflow::Status::OK();
}

tensorflow::Status NodePlacer::SpreadNpuNodeFromPlacement(Placement placement) {
  std::vector<tensorflow::Node *> starts = GetNodesPlacedOn(placement);
  if (starts.empty()) {
    DLOG() << "Skip spread npu from placement " << kPlacementString[placement] << " as no nodes placed on";
    return tensorflow::Status::OK();
  }
  for (auto &start : starts) {
    (void)GetOrCreateNpuCluster(start);  // For single node
  }
  DLOG() << "Start spread npu from " << GetNodesPlacedOn(placement).size() << " nodes placed on "
         << kPlacementString[placement] << ", npu node size " << GetNodesPlacedOn(Placement::NPU).size();

  const auto enter = [](tensorflow::Node *node) { (void)node; };
  tensorflow::DFSFrom(*graph_, starts, enter, {}, {},
                      [this](const tensorflow::Edge &edge) { return SpreadNpuEdge(edge, true); });
  tensorflow::ReverseDFSFrom(*graph_, starts, enter, {}, {},
                             [this](const tensorflow::Edge &edge) { return SpreadNpuEdge(edge, false); });
  DLOG() << "Successfully spread npu from placement " << kPlacementString[placement] << ", npu node size "
         << GetNodesPlacedOn(Placement::NPU).size();
  return tensorflow::Status::OK();
}

tensorflow::Status NodePlacer::SpreadNpuNode() {
  if (std::any_of(graph_->op_nodes().begin(), graph_->op_nodes().end(),
                  [](const tensorflow::Node *n) { return n->type_string() == "MutexLock"; })) {
    DLOG() << "Only compile npu-only nodes as MutexLock found. MutexLock usually caused by using tf.CriticalSection(), "
              "which is meaningless in TF2 auto graph mode, Remove it may improve you train performance";
    for (auto node : graph_->op_nodes()) {
      if (IsNodePlacedOn(node, Placement::NPU)) {
        (void)GetOrCreateNpuCluster(node);
      }
    }
    return tensorflow::Status::OK();
  }
  // We spread twice as first iteration spread from determined npu node, and,
  // The second iteration spread from all unplaced node
  NPU_REQUIRES_OK(SpreadNpuNodeFromPlacement(Placement::NPU));
  NPU_REQUIRES_OK(SpreadNpuNodeFromPlacement(Placement::WHEREVER));
  return tensorflow::Status::OK();
}

const std::set<tensorflow::Node *> &NodePlacer::GetConcreteNodes(tensorflow::Node *node) {
  const static std::set<tensorflow::Node *> kEmptyNodes;
  auto iter = concrete_clusters_.find(node);
  if (iter != concrete_clusters_.end()) {
    return iter->second->nodes;
  }
  return kEmptyNodes;
}

void NodeOrCluster::VisitOutNodes(const std::function<void(tensorflow::Node *)> &visitor) const {
  if (is_cluster_) {
    for (auto &n : cluster_->out_nodes) {
      visitor(n);
    }
  } else {
    for (auto n : node_->out_nodes()) {
      visitor(n);
    }
  }
}
void NodeOrCluster::VisitInNodes(const std::function<void(tensorflow::Node *)> &visitor) const {
  if (is_cluster_) {
    for (auto &n : cluster_->in_nodes) {
      visitor(n);
    }
  } else {
    for (auto n : node_->in_nodes()) {
      visitor(n);
    }
  }
}

bool NodeOrCluster::VisitNodes(const std::function<bool(tensorflow::Node *)> &visitor) const {
  if (is_cluster_) {
    for (auto &n : cluster_->nodes) {
      if (!visitor(n)) {
        return false;
      }
    }
    return true;
  } else {
    return visitor(node_);
  }
}

NodeOrCluster NodePlacer::GetNodeOrCluster(tensorflow::Node *node) {
  auto iter = npu_clusters_.find(node);
  if (iter != npu_clusters_.end()) {
    return NodeOrCluster(iter->second.get());
  } else {
    iter = concrete_clusters_.find(node);
    if (iter != concrete_clusters_.end()) {
      return NodeOrCluster(iter->second.get());
    } else {
      return NodeOrCluster(node);
    }
  }
}

bool NodePlacer::VisitPathNodes(tensorflow::Node *start, tensorflow::Node *end,
                                const std::function<bool(tensorflow::Node *)> visitor) {
  if (node_topo_[start] > node_topo_[end]) {  // dst->src
    std::swap(start, end);
  }

  ResetNodeMask();  // Clear all masks

  std::queue<NodeOrCluster> q;
  auto collect_seen_nodes = [this, &q](tensorflow::Node *n) {  // Visitor for collect nodes
    q.push(GetNodeOrCluster(n));
  };

  GetNodeOrCluster(start).VisitOutNodes(collect_seen_nodes);  // Collect start nodes
  while (!q.empty()) {
    auto node = q.front();
    q.pop();
    if (!FetchSetMask(node)) {  // Visit node at first masking
      node.VisitOutNodes(collect_seen_nodes);
    }
  }

  GetNodeOrCluster(end).VisitInNodes(collect_seen_nodes);
  while (!q.empty()) {
    auto node = q.front();
    q.pop();
    if (FetchClearMask(node)) {  // Access masked node only once
      if (!node.VisitNodes(visitor)) {
        return false;
      }
      node.VisitInNodes(collect_seen_nodes);
    }
  }
  return true;
}

std::shared_ptr<Cluster> NodePlacer::GetOrCreateConcreteCluster(tensorflow::Node *node) {
  auto iter = concrete_clusters_.find(node);
  if (iter != concrete_clusters_.end()) {
    return iter->second;
  }
  auto cluster = std::make_shared<Cluster>(this, node, Placement::WHEREVER);
  concrete_clusters_[node] = cluster;
  return cluster;
}

std::shared_ptr<Cluster> NodePlacer::GetOrCreateNpuCluster(tensorflow::Node *node) {
  auto iter = npu_clusters_.find(node);
  if (iter != npu_clusters_.end()) {
    return iter->second;
  }
  auto cluster = std::make_shared<Cluster>(this, node, Placement::NPU);
  npu_clusters_[node] = cluster;
  auto concrete_nodes = GetConcreteNodes(node);
  for (auto &n : concrete_nodes) {
    if (n == node) {
      continue;
    }
    (void)cluster->Merge(n);
    npu_clusters_[n] = cluster;
    (void)concrete_clusters_.erase(n);
  }
  return cluster;
}

bool NodePlacer::ColocateNpu(tensorflow::Node *src, tensorflow::Node *dst) {
  DLOG() << "Try colocate " << src->name() << " with " << dst->name() << " on npu";
  // Allow colocate placed node when place 'wherever' cluster
  if (!IsNodeCanPlacedOn(src, Placement::NPU)) {
    DLOG() << "Skip colocate as " << src->name() << " has placed on cpu";
    return false;
  }

  auto src_cluster = GetOrCreateNpuCluster(src);
  auto target = GetOrCreateNpuCluster(dst);
  if (src_cluster == target) {
    DLOG() << src->name() << " has already colocate with " << dst->name() << " in cluster " << target->name;
    return true;
  }
  if (src_cluster->nodes.size() > target->nodes.size()) {
    target.swap(src_cluster);
  }

  DLOG() << "Start colocate path node from " << src->name() << " to " << dst->name() << " to cluster " << target->name;
  std::unordered_set<tensorflow::Node *> path_nodes;
  auto visitor = [this, &path_nodes](tensorflow::Node *node) {
    (void)path_nodes.insert(node);
    bool placeable = IsNodeCanPlacedOn(node, Placement::NPU);
    DLOG() << "Visited path node " << node->name() << " can " << (placeable ? "" : "not") << " placed on npu";
    return placeable;
  };
  if (VisitPathNodes(src, dst, visitor)) {
    for (auto &node : path_nodes) {
      if (target->Merge(node)) {
        npu_clusters_[node] = target;
        (void)concrete_clusters_.erase(node);
      }
    }
    target->Merge(src_cluster);
    for (auto &node : src_cluster->nodes) {
      npu_clusters_[node] = target;
      (void)concrete_clusters_.erase(node);
    }
    return true;
  }
  return false;
}

// Weather the edge can be npu bound
bool NodePlacer::IsSupportedNpuBound(const tensorflow::Edge &edge) const {
  return edge.IsControlEdge() || device_->SupportedInputAndOutputType(EdgeDataType(edge));
}

// Check weather the node has placed on placement device
bool NodePlacer::IsNodePlacedOn(tensorflow::Node *node, Placement placement) {
  if (placement == Placement::WHEREVER) {
    return !IsNodePlaced(node);
  }
  auto iter = node_placement_.find(node);
  if (iter != node_placement_.end()) {
    return iter->second == placement;
  }
  auto cluster = npu_clusters_.find(node);
  if (cluster != npu_clusters_.end()) {
    return cluster->second->placement == placement;
  }
  return false;
}

// Check weather the node can place on placement device
bool NodePlacer::IsNodeCanPlacedOn(tensorflow::Node *node, Placement placement) {
  return !IsNodePlaced(node) || IsNodePlacedOn(node, placement);
}

// is this node placed in surely device
bool NodePlacer::IsNodePlaced(tensorflow::Node *node) {
  if (node_placement_.find(node) != node_placement_.end()) {
    return true;
  }
  return npu_clusters_.find(node) != npu_clusters_.end();
}
}  // namespace npu
