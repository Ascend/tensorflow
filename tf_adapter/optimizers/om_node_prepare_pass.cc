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

#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/npu_attrs.h"

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
class OmNodePreparePass : public GraphOptimizationPass {
 public:
  OmNodePreparePass() = default;
  ~OmNodePreparePass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;

 private:
  static std::vector<Node *> GetGraphOmNodes(const Graph &graph);
  static std::map<std::string, std::string> GetGraphConfigs(const Graph &graph);
  static Status ProcessGraph(std::unique_ptr<Graph> &graph, FunctionLibraryDefinition &fdef_lib);
};

Status OmNodePreparePass::Run(const GraphOptimizationPassOptions &options) {
  if ((options.graph == nullptr && options.partition_graphs == nullptr) || options.flib_def == nullptr) {
    return Status::OK();
  }

  if (options.graph != nullptr) {
    TF_RETURN_IF_ERROR(ProcessGraph(*options.graph, *options.flib_def));
  } else if (options.partition_graphs != nullptr) {
    for (auto &partition_graph : *options.partition_graphs) {
      TF_RETURN_IF_ERROR(ProcessGraph(partition_graph.second, *options.flib_def));
    }
  }

  return Status::OK();
}

constexpr const char *kOmNodeType = "LoadAndExecuteOm";
std::vector<Node *> OmNodePreparePass::GetGraphOmNodes(const Graph &graph) {
  std::vector<Node *> om_nodes;
  for (auto node : graph.nodes()) {
    if (node->type_string() != kOmNodeType) {
      continue;
    }
    ADP_LOG(INFO) << "Collect om node " << node->name() << " " << node->type_string();
    om_nodes.emplace_back(node);
  }
  return om_nodes;
}

std::map<std::string, std::string> OmNodePreparePass::GetGraphConfigs(const Graph &graph) {
  for (Node *n : graph.nodes()) {
    if ((n != nullptr) && (n->attrs().Find("_NpuOptimizer") != nullptr)) {
      return NpuAttrs::GetAllAttrOptions(n->attrs());
    }
  }
  return {};
}

Status OmNodePreparePass::ProcessGraph(std::unique_ptr<Graph> &graph, FunctionLibraryDefinition &fdef_lib) {
  auto om_nodes = GetGraphOmNodes(*graph);
  if (om_nodes.empty()) {
    ADP_LOG(INFO) << "Skip process graph as no om nodes found";
    return Status::OK();
  }

  static std::atomic_uint64_t graph_index{0U};
  uint64_t index = graph_index.fetch_add(1U);
  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + "TF_BeforeOmPrepare_" + std::to_string(index) + ".pbtxt";
    tensorflow::GraphDef def;
    graph->ToGraphDef(&def);
    (void) WriteTextProto(Env::Default(), pbtxt_path, def);
  }

  ADP_LOG(INFO) << "Prepare for om graph as " << om_nodes.size() << " om nodes found";
  std::string init_fun_name = "empty_for_npu_init_" + std::to_string(Env::Default()->NowNanos());
  tensorflow::AttrValue function_attr;
  function_attr.mutable_func()->set_name(init_fun_name);
  Node *geop_node = nullptr;
  TF_RETURN_IF_ERROR(tensorflow::NodeBuilder("system_init", "GeOp")
                         .Input(std::vector<tensorflow::NodeBuilder::NodeOut>{})
                         .Attr("Tin", tensorflow::DataTypeVector{})
                         .Attr("Tout", tensorflow::DataTypeVector{})
                         .Attr("function", function_attr)
                         .Device(om_nodes.front()->assigned_device_name())
                         .Finalize(graph.get(), &geop_node));
  geop_node->set_assigned_device_name(om_nodes.front()->assigned_device_name());

  geop_node->AddAttr("_NpuOptimizer", "NpuOptimizer");
  for (const auto &option : GetGraphConfigs(*graph)) {
    geop_node->AddAttr(std::string("_") + option.first, option.second);
  }

  tensorflow::FunctionDef fdef;
  fdef.mutable_signature()->set_name(init_fun_name);
  *fdef.mutable_attr() = geop_node->def().attr();
  TF_RETURN_IF_ERROR(fdef_lib.AddFunctionDef(fdef));

  for (auto &om_node : om_nodes) {
    om_node->AddAttr("_NoNeedOptimize", true);  // Skip optimize for graph with om node
    ADP_LOG(INFO) << "Add control edge from system init op " << geop_node->name() << " to om node " << om_node->name();
    REQUIRES_NOT_NULL(graph->AddControlEdge(geop_node, om_node));
  }

  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + "TF_AfterOmPrepare_" + std::to_string(index) + ".pbtxt";
    tensorflow::GraphDef def;
    graph->ToGraphDef(&def);
    (void) WriteTextProto(Env::Default(), pbtxt_path, def);
  }

  return Status::OK();
}
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0, OmNodePreparePass);
}  // namespace tensorflow
