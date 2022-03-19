/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef NPU_DEVICE_CORE_OPTIMIZERS_RUNTIME_NODE_PLACER_H
#define NPU_DEVICE_CORE_OPTIMIZERS_RUNTIME_NODE_PLACER_H

#include <functional>
#include <string>
#include "npu_device.h"

namespace npu {
enum class Placement { WHEREVER, NPU, CPU };
static std::map<Placement, std::string> kPlacementString = {
  {Placement::WHEREVER, "wherever"}, {Placement::NPU, "npu"}, {Placement::CPU, "cpu"}};

class NodePlacer;
struct Cluster {
  explicit Cluster(const NodePlacer *placer, tensorflow::Node *node, uint64_t topo, Placement place);
  bool Merge(tensorflow::Node *node);
  void Merge(std::shared_ptr<Cluster> other);
  void UpdateTopo(uint64_t topo);
  std::set<tensorflow::Node *> nodes;
  std::unordered_set<tensorflow::Node *> in_nodes;
  std::unordered_set<tensorflow::Node *> out_nodes;
  std::string name;
  Placement placement;
  uint64_t min_topo;
  uint64_t max_topo;

 private:
  const NodePlacer *placer_;  // Not owned
};

struct NodeOrCluster {
  explicit NodeOrCluster(Cluster *cluster) : is_cluster_(true) { cluster_ = cluster; }
  explicit NodeOrCluster(tensorflow::Node *node) { node_ = node; }
  void VisitInNodes(std::function<void(tensorflow::Node *)> visitor);
  void VisitOutNodes(std::function<void(tensorflow::Node *)> visitor);
  bool VisitNodes(std::function<bool(tensorflow::Node *)> visitor);
  size_t Hash() const { return (is_cluster_ ? reinterpret_cast<size_t>(cluster_) : reinterpret_cast<size_t>(node_)); }
  bool operator==(const NodeOrCluster &other) const {
    return (is_cluster_ ? cluster_ == other.cluster_ : node_ == other.node_);
  }
  bool IsCluster() const { return is_cluster_; }
  const Cluster *GetCluster() const { return cluster_; }

 private:
  Cluster *cluster_{nullptr};
  tensorflow::Node *node_{nullptr};
  bool is_cluster_ = false;
};

struct StableNodeCompartor {
  bool operator()(const tensorflow::Node *a, const tensorflow::Node *b) { return a->id() < b->id(); }
};

class NodePlacer {
 public:
  explicit NodePlacer(TFE_Context *context, tensorflow::Graph *graph, NpuDevice *device)
      : context_(context), graph_(graph), device_(device) {}
  tensorflow::Status Apply();
  void InitNodeTopo();
  tensorflow::Status BuildNpuOp();
  tensorflow::Status CopyShareableNode();
  tensorflow::Status MergeCopiedSharedNodes();
  std::vector<tensorflow::Node *> MergeCopiedSharedNodes(std::vector<tensorflow::Node *> all_nodes);
  tensorflow::Status DeterminedSurelyNodes();
  tensorflow::Status BuildConcreteCluster();

  tensorflow::Status SpreadCpuNode();
  bool SpreadNpuEdge(const tensorflow::Edge &edge, bool forward);
  tensorflow::Status SpreadNpuNode();

  const std::set<tensorflow::Node *> &GetConcreteNodes(tensorflow::Node *node);
  bool VisitPathNodes(tensorflow::Node *start, tensorflow::Node *end, std::function<bool(tensorflow::Node *)> visitor);
  void VisitInNodes(tensorflow::Node *node, std::function<void(tensorflow::Node *)> visitor);
  void VisitOutNodes(tensorflow::Node *node, std::function<void(tensorflow::Node *)> visitor);

  NodeOrCluster GetNodeOrCluster(tensorflow::Node *node);
  std::shared_ptr<Cluster> GetOrCreateNpuCluster(tensorflow::Node *node);
  std::shared_ptr<Cluster> GetOrCreateConcreteCluster(tensorflow::Node *node);
  std::set<std::shared_ptr<Cluster>> GetNpuClusters();
  void Concrete(tensorflow::Node *src, tensorflow::Node *dst);
  bool ColocateNpu(tensorflow::Node *src, tensorflow::Node *dst);
  uint64_t Topo(tensorflow::Node *node) const { return node_topo_.at(node); }

 private:
  static bool IsNpuMeaningLessNode(tensorflow::Node *node);
  // Weather the edge can be npu bound
  bool IsSupportedNpuBound(const tensorflow::Edge &edge);
  // is this node placed in surely device
  bool IsNodePlaced(tensorflow::Node *node);
  // Check weather the node has placed on placement device
  bool IsNodePlacedOn(tensorflow::Node *node, Placement placement);
  // Check weather the node can place on placement device
  bool IsNodeCanPlacedOn(tensorflow::Node *node, Placement placement);
  bool IsClusterMustPlaceOnNpu(const Cluster &cluster);
  // Get the node current placement
  Placement GetNodePlacement(tensorflow::Node *node);
  std::vector<tensorflow::Node *> GetNodesPlacedOn(Placement placement);

  TFE_Context *context_;                                                        // not owned
  tensorflow::Graph *graph_;                                                    // not owned
  NpuDevice *device_;                                                           // not owned
  std::map<tensorflow::Node *, uint64_t, StableNodeCompartor> node_shared_id_;  // node to its copied nodes group id
  std::map<tensorflow::Node *, uint64_t> node_topo_;
  std::map<tensorflow::Node *, Placement, StableNodeCompartor>
    node_placement_;  // Just npu or cpu, never store wherever here
  std::map<tensorflow::Node *, std::shared_ptr<Cluster>, StableNodeCompartor>
    npu_clusters_;  // a npu cluster means a npu op
  std::map<tensorflow::Node *, std::shared_ptr<Cluster>, StableNodeCompartor>
    concrete_clusters_;  // node to its concrete nodes
};
}  // namespace npu

namespace std {
template <>
struct hash<npu::NodeOrCluster> {
  size_t operator()(const npu::NodeOrCluster &v) const { return v.Hash(); }
};
}  // namespace std

#endif  // NPU_DEVICE_CORE_OPTIMIZERS_RUNTIME_NODE_PLACER_H
