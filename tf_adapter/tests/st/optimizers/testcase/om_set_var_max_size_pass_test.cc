#include "tf_adapter/optimizers/om_set_var_max_size_pass.h"
#include "gtest/gtest.h"
#include "mmpa/mmpa_api.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {
class SetVarMaxSizePassTest : public testing::Test {
 public:
  SetVarMaxSizePassTest() : graph_(absl::make_unique<Graph>(OpRegistry::Global())) {}
  static void InitGraph(const string &graph_def_path, Graph *graph) {
    GraphDef graph_def;
    ReadTextProto(Env::Default(), graph_def_path, &graph_def);
    GraphConstructorOptions opts;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
  }

  void InitGraph(const string &graph_def_path) {
    char trusted_path[MMPA_MAX_PATH] = { "\0" };
    if (mmRealPath(graph_def_path.c_str(), trusted_path, MMPA_MAX_PATH) != EN_OK) {
      LOG(ERROR) << "Get real path failed.";
      return;
    }
    LOG(INFO) << "input graph def path: " << trusted_path;
    InitGraph(trusted_path, graph_.get());
    original_ = CanonicalGraphString(graph_.get());
  }

  static bool IncludeNode(const Node *n) { return n->IsOp(); }

  static string EdgeId(const Node* n, int index) {
    if (index == 0) {
      return n->type_string();
    } else if (index == Graph::kControlSlot) {
      return strings::StrCat(n->type_string(), ":control");
    } else {
      return strings::StrCat(n->type_string(), ":", index);
    }
  }

  string CanonicalGraphString(Graph* g) {
    for (Node* n : g->nodes()) {
      if (IncludeNode(n)) {
        if (n->assigned_device_name().empty()) {
          n->set_assigned_device_name("/job:localhost/replica:0/task:0/device:CPU:0");
          break;
        }
      }
    }

    std::vector<string> edges;
    for (const Edge* e : g->edges()) {
      if (IncludeNode(e->src()) && IncludeNode(e->dst())) {
        edges.push_back(strings::StrCat(EdgeId(e->src(), e->src_output()), "->",
                                        EdgeId(e->dst(), e->dst_input())));
      }
    }
    // Canonicalize
    return strings::StrCat(absl::StrJoin(edges, ";"));
  }

  string DoRunSetVarMaxSizePassTest() {
    string before = CanonicalGraphString(graph_.get());
    LOG(INFO) << "Before om conversion pass: " << before;

    std::unique_ptr<Graph> *ug = &graph_;
    GraphOptimizationPassOptions options;
    SessionOptions session_options;
    session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
    auto *custom_config = session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
    custom_config->set_name("NpuOptimizer");
    AttrValue job = AttrValue();
    job.set_s("localhost");
    (*custom_config->mutable_parameter_map())["job"] = job;
    AttrValue const_input = AttrValue();
    const_input.set_s("input");
    (*custom_config->mutable_parameter_map())["const_input"] = const_input;
    options.session_options = &session_options;
    options.graph = ug;
    FunctionLibraryDefinition flib_def((*ug)->flib_def());
    options.flib_def = &flib_def;
    SetVarMaxSizePass().Run(options);

    string result = CanonicalGraphString(options.graph->get());
    LOG(INFO) << "After om conversion pass: " << result;
    return result;
  }

  const string &OriginalGraph() const { return original_; }

  std::unique_ptr<Graph> graph_;
  string original_;
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(SetVarMaxSizePassTest, MaxSizeTest) {
  string org_graph_def_path = "tf_adapter/tests/ut//optimizers/pbtxt/input_max_size_test.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "Placeholder->DecodeJpeg";
  EXPECT_EQ(DoRunSetVarMaxSizePassTest(), target_graph);
}
} // end namespace
} // end tensorflow

