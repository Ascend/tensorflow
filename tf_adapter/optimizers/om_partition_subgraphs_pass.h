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

#ifndef TENSORFLOW_OM_PARTITION_SUBGRAPHS_PASS_H_
#define TENSORFLOW_OM_PARTITION_SUBGRAPHS_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace OMSplitter {
Status MarkForPartition(const GraphOptimizationPassOptions &options, int &clusterNum, bool mix_compile_mode,
                        int graph_num, const FunctionLibraryDefinition *func_lib,
                        std::map<std::string, std::string> pass_options,
                        std::map<std::string, std::string> &graph_options);

// Transformation that finds subgraphs whose nodes are marked with
// 'groupAttribute', splits those subgraphs into functions, and replaces
// the originals with GEOps.
// 'groupAttribute' must be a string valued-attribute that names the new
// functions to introduce.
Status OMPartitionSubgraphsInFunctions(string groupAttribute, const GraphOptimizationPassOptions &options,
                                       string graph_format);

bool IsNpuSupportingNode(const NodeDef &node_def, bool mix_compile_mode,
                         const FunctionLibraryDefinition *func_lib, bool support_const = false);
bool IsNpuSupportingNode(const Node *node, bool mix_compile_mode, const FunctionLibraryDefinition *func_lib,
                         bool support_const = false);
}  // namespace OMSplitter

class OMPartitionSubgraphsPass : public GraphOptimizationPass {
 public:
  OMPartitionSubgraphsPass() = default;
  ~OMPartitionSubgraphsPass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;

 private:
  Status ProcessGraph(std::unique_ptr<Graph> *graph, FunctionLibraryDefinition *func_lib,
                      const OptimizationPassRegistry::Grouping pass_group_value);
  Status AccumulateNFusion(Graph *graph_in, Node *node) const;
  void GetGraphConfig(const Node *node, bool enable_dp, std::map<std::string, std::string> &graph_options) const;
  void ParseInputShapeRange(const std::string dynamic_inputs_shape_range, bool enable_dp,
                            std::map<std::string, std::string> &graph_options) const;
  Status ProcessGetNext(Node *node, const std::string enable_dp,
                        std::vector<Node*> &remove_nodes, Graph *graph_in) const;
  Status SplitUnaryOpsComposition(Graph *graph, Node *node) const;
  Status CopyVarsBetweenGeOp(Graph *graph) const;
  Status CopyConstBetweenGeOp(Graph *graph) const;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_OM_PARTITION_SUBGRAPHS_PASS_H_
