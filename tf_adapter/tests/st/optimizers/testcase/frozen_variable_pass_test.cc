#include "tf_adapter/util/npu_attrs.h"
#include "gtest/gtest.h"
#include "mmpa/mmpa_api.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/common_runtime/device_factory.h"

namespace tensorflow {
class FrozenVariablePass : public GraphOptimizationPass {
 public:
  FrozenVariablePass() = default;
  ~FrozenVariablePass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;
 private:
  std::map<std::string, std::string> GetGraphConfigs(const Graph &graph);
  Status DoConstantFolding(const GraphOptimizationPassOptions &options, const uint64_t index);
  bool IsAllOutputsIdentity(const Node * const node);
  bool IsAllOutputsReadOp(const Node * const node);
  bool IsNeedBuildPartitionedCall(const Node * const node);
};

namespace {
class FrozenVariablePassTest : public testing::Test {
public:
  FrozenVariablePassTest() : graph_(absl::make_unique<Graph>(OpRegistry::Global())) {}
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
        if (n->type_string() == "Add" && n->assigned_device_name().empty()) {
          n->set_assigned_device_name("/job:localhost/replica:0/task:0/device:CPU:0");
          break;
        }
        if (n->assigned_device_name().empty()) {
          n->set_assigned_device_name("/job:localhost/replica:0/task:0/device:CPU:0");
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

  string DoRunFrozenVariablePassTest(bool need_frozen) {
    string before = CanonicalGraphString(graph_.get());
    LOG(INFO) << "Before replace variable pass: " << before;

    std::unique_ptr<Graph> *ug = &graph_;
    GraphOptimizationPassOptions options;
    SessionOptions session_options;
    auto *custom_config = session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
    custom_config->set_name("NpuOptimizer");
    AttrValue is_need_frozen = AttrValue();
    is_need_frozen.set_b(need_frozen);
    (*custom_config->mutable_parameter_map())["frozen_variable"] = is_need_frozen;
    options.session_options = &session_options;
    options.graph = ug;
    FunctionLibraryDefinition flib_def((*ug)->flib_def());
    options.flib_def = &flib_def;

    DeviceSet device_set;
    DeviceFactory* cpu_factory = DeviceFactory::GetFactory("CPU");
    std::vector<std::unique_ptr<Device>> devices;
    cpu_factory->CreateDevices(
            session_options, "/job:localhost/replica:0/task:0", &devices);
    device_set.AddDevice(devices.begin()->get());
    options.device_set = &device_set;
    FrozenVariablePass().Run(options);
    string result = CanonicalGraphString(options.graph->get());

    return result;
  }

  const string &OriginalGraph() const { return original_; }

  std::unique_ptr<Graph> graph_;
  string original_;
 protected:
  virtual void SetUp() { *const_cast<bool *>(&kDumpGraph) = true; }
  virtual void TearDown() {}
};

TEST_F(FrozenVariablePassTest, frozen_variable_true) {
  string org_graph_def_path = "tf_adapter/tests/st/optimizers/pbtxt/om_test_build_geop.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "Const->Add;Add->_Retval;PartitionedCall->Add:1";
  EXPECT_EQ(DoRunFrozenVariablePassTest(true), target_graph);
}

TEST_F(FrozenVariablePassTest, frozen_variable_false) {
  string org_graph_def_path = "tf_adapter/tests/st/optimizers/pbtxt/om_test_build_geop.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "VariableV2->Identity;Const->Add;Identity->Add:1;Add->_Retval";
  EXPECT_EQ(DoRunFrozenVariablePassTest(false), target_graph);
}

TEST_F(FrozenVariablePassTest, frozen_varhandleop_true) {
  string org_graph_def_path = "tf_adapter/tests/st/optimizers/pbtxt/varhandleop_test.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "Const->Add;Add->_Retval;PartitionedCall->Add:1";
  EXPECT_EQ(DoRunFrozenVariablePassTest(true), target_graph);
}

TEST_F(FrozenVariablePassTest, frozen_no_variable_true) {
  string org_graph_def_path = "tf_adapter/tests/st/optimizers/pbtxt/om_test_while_loop.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "Const->Enter;Enter->Merge;Merge:control->Const:control;Merge->Less;Const->Less:1;"
                             "Less->LoopCond;Merge->Switch;LoopCond->Switch:1;Switch->Exit;Exit->_Retval;"
                             "Switch:1->Identity;Identity:control->Const:control;Const->Add;Identity->Add:1;"
                             "Add->NextIteration;NextIteration->Merge:1";
  EXPECT_EQ(DoRunFrozenVariablePassTest(true), target_graph);
}
} // end namespace
}