#include "gtest/gtest.h"

#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

#include <memory>

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

namespace {
class OmNodePreparePassTest : public testing::Test {
 public:
  std::unique_ptr<Graph> graph;
  std::unique_ptr<OmNodePreparePass> optimizer;
  std::unique_ptr<FunctionLibraryDefinition> libraries;
  GraphOptimizationPassOptions options;

  Status Run() const {
    return optimizer->Run(options);
  }

 public:
  void SetUp() override {
    graph.reset(new Graph(OpRegistry::Global()));
    optimizer.reset(new OmNodePreparePass());
    FunctionDefLibrary defs;
    libraries.reset(new FunctionLibraryDefinition(OpRegistry::Global(), defs));
    options.graph = &graph;
    options.flib_def = libraries.get();
  }
  void TearDown() override {}
};
/**********************************************
REGISTER_OP("LoadAndExecuteOm")
    .Input("inputs: Tin")
    .Attr("Tin: list(type) >= 0")
    .Output("outputs: output_dtypes")
    .Attr("output_dtypes: list(type) >= 0")
    .Attr("om_path: string")
    .Attr("executor_type: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);
**********************************************/

TEST_F(OmNodePreparePassTest, TestGraphWithoutOmNodeSuccess) {
  ASSERT_EQ(Run(), Status::OK());
}

TEST_F(OmNodePreparePassTest, TestGraphWithOmNodeSuccess) {
  Node *om_node = nullptr;
  NodeBuilder("om_node", "LoadAndExecuteOm")
      .Input(std::vector<NodeBuilder::NodeOut>{})
      .Attr("output_dtypes", tensorflow::DataTypeVector{})
      .Attr("om_path", "")
      .Finalize(graph.get(), &om_node);
  ASSERT_EQ(Run(), Status::OK());

  ASSERT_EQ(graph->num_op_nodes(), 2U);
  Node *system_init = nullptr;
  for (auto edge : om_node->in_edges()) {
    if (edge->IsControlEdge() && edge->src()->IsOp()) {
      system_init = edge->src();
    }
  }
  ASSERT_NE(system_init, nullptr);
  ASSERT_EQ(system_init->type_string(), "GeOp");
}
}  // end namespace
}  // namespace tensorflow
