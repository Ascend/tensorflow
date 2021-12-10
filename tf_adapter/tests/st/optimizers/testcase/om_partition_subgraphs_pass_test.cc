#include "tf_adapter/optimizers/om_partition_subgraphs_pass.h"
#include "tf_adapter/util/infershape_util.h"
#include "tf_adapter/util/npu_attrs.h"
#include "gtest/gtest.h"
#include "mmpa/mmpa_api.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {
class OmOptimizationPassTest : public testing::Test {
 public:
  OmOptimizationPassTest() : graph_(absl::make_unique<Graph>(OpRegistry::Global())) {}
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
      return n->name();
    } else if (index == Graph::kControlSlot) {
      return strings::StrCat(n->name(), ":control");
    } else {
      return strings::StrCat(n->name(), ":", index);
    }
  }

  string CanonicalGraphString(Graph* g) {
    std::vector<string> edges;
    for (const Edge* e : g->edges()) {
      if (IncludeNode(e->src()) && IncludeNode(e->dst())) {
        edges.push_back(strings::StrCat(EdgeId(e->src(), e->src_output()), "->",
                                        EdgeId(e->dst(), e->dst_input())));
      }
    }
    return strings::StrCat(absl::StrJoin(edges, ";"));
  }

  void ConvertFuncDefToGraph(const FunctionDef &func_def, Graph *g) {
    Graph graph(OpRegistry::Global());
    FunctionDefLibrary flib;
    FunctionLibraryDefinition flib_def(graph.op_registry(), flib);
    (void)InferShapeUtil::GetSubGraphFromFunctionDef(flib_def, func_def, g);
  }

  string DoRunOmOptimizationPassTest(bool is_convert_func_def = false) {
    string before = CanonicalGraphString(graph_.get());
    LOG(INFO) << "Before om conversion pass: " << before;

    GraphDef org_graph_def;
    std::unique_ptr<Graph> *ug = &graph_;
    GraphOptimizationPassOptions options;
    options.graph = ug;
    FunctionLibraryDefinition flib_def((*ug)->flib_def());
    options.flib_def = &flib_def;
    OMPartitionSubgraphsPass().Run(options);
    string result;
    if (is_convert_func_def) {
      GraphDef graph_def;
      options.graph->get()->ToGraphDef(&graph_def);
      FunctionDefLibrary func_def_lib = graph_def.library();
      for (auto func_def : func_def_lib.function()) {
        Graph ge_graph(OpRegistry::Global());
        ConvertFuncDefToGraph(func_def, &ge_graph);
        result += CanonicalGraphString(&ge_graph) + "|";
      }
      
    } else {
      result = CanonicalGraphString(options.graph->get());
    }
    
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

TEST_F(OmOptimizationPassTest, BuildGeOpTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_build_geop.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "GeOp1_0->retval_Add_0_0";
  EXPECT_EQ(DoRunOmOptimizationPassTest(), target_graph);
}
TEST_F(OmOptimizationPassTest, AccumulateTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_accumulate.pbtxt";
  InitGraph(org_graph_def_path);
  std::string result = DoRunOmOptimizationPassTest(true);
  EXPECT_EQ(result.find("AccumulateNV2") != result.npos, true);
}
TEST_F(OmOptimizationPassTest, MixCompileCopyVarsTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_copy_variables.pbtxt";
  InitGraph(org_graph_def_path);
  std::string result = DoRunOmOptimizationPassTest(true);
  std::vector<std::string> result_graphs;
  Split(result, result_graphs, "|");
  bool has_variables = false;
  for (auto result_graph : result_graphs) {
    if (result_graph.find("Variable_1->") != result_graph.npos) {
      has_variables = true;
    } else {
      has_variables = false;
    }
  }
  EXPECT_EQ(has_variables, true);
}
TEST_F(OmOptimizationPassTest, BuildGetNextGeOpTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_getnext_build_geop.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = DoRunOmOptimizationPassTest(true);
  EXPECT_EQ(target_graph.find("IteratorGetNext") != target_graph.npos, true);
}
TEST_F(OmOptimizationPassTest, DpGraphTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_iterator_build_geop.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "normalize_element/component_0->TensorSliceDataset;TensorSliceDataset->OptimizeDataset;optimizations->OptimizeDataset:1;OptimizeDataset->ModelDataset;ModelDataset->MakeIterator;IteratorV2->MakeIterator:1";
  EXPECT_EQ(DoRunOmOptimizationPassTest(), target_graph);
}
TEST_F(OmOptimizationPassTest, MergeClustersTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_merge_clusters.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = DoRunOmOptimizationPassTest(true);
  bool ret = false;
  if (target_graph.find("v1->") != target_graph.npos && target_graph.find("v2->") != target_graph.npos
    && target_graph.find("v3->") != target_graph.npos) { ret = true; }
  EXPECT_EQ(ret, true);
}
TEST_F(OmOptimizationPassTest, MixCompileTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_mix_compile.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "GeOp7_0->Unique;Unique->GeOp7_1";
  EXPECT_EQ(DoRunOmOptimizationPassTest(), target_graph);
}
TEST_F(OmOptimizationPassTest, UnaryOpsTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_unaryops.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = DoRunOmOptimizationPassTest(true);
  bool ret = false;
  if (target_graph.find("_Floor") != target_graph.npos && target_graph.find("_Abs") != target_graph.npos) {
    ret = true;
  }
  EXPECT_EQ(ret, true);
}
TEST_F(OmOptimizationPassTest, InOutPairFlagFalseTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_in_out_pair_flag_false.pbtxt";
  InitGraph(org_graph_def_path);
  EXPECT_EQ(DoRunOmOptimizationPassTest(), "GeOp9_0->retval_Add_0_0");
}
TEST_F(OmOptimizationPassTest, InOutPairFlagTrueTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_in_out_pair_flag_true.pbtxt";
  InitGraph(org_graph_def_path);
  EXPECT_EQ(DoRunOmOptimizationPassTest(), "Variable->Variable/read;GeOp10_0->Add;Variable/read->Add:1;Add->retval_Add_0_0");
}
TEST_F(OmOptimizationPassTest, DynamicGetNextInputTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_getnext_lazy_recompile.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = DoRunOmOptimizationPassTest(true);
  std::vector<std::string> result_graphs;
  Split(target_graph, result_graphs, "|");
  EXPECT_EQ(result_graphs[1], "IteratorV2->IteratorGetNext;IteratorGetNext->IteratorGetNext_0_retval_RetVal;IteratorGetNext:1->IteratorGetNext_1_retval_RetVal");
}
TEST_F(OmOptimizationPassTest, IncludeNodeFuncTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/ctrl_if_test.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = DoRunOmOptimizationPassTest();
  EXPECT_EQ(target_graph, "arg_Placeholder_0_0->GeOp12_0;GeOp12_0->retval_IF_branch_1_0_0");
}
TEST_F(OmOptimizationPassTest, WhileLoopTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_while_loop.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = DoRunOmOptimizationPassTest();
  EXPECT_EQ(target_graph, "GeOp13_0->retval_Exit_0_0");
}
TEST_F(OmOptimizationPassTest, DynamicGetNextInput1Test) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_getnext_dynamic_execute.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = DoRunOmOptimizationPassTest();
  EXPECT_EQ(target_graph, "arg_arg_Placeholder_0_0->GeOp14_0");
}
TEST_F(OmOptimizationPassTest, StringInputMaxSizeTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/om_test_string_input.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = DoRunOmOptimizationPassTest();
  EXPECT_EQ(target_graph, "arg_input_0_0->DecodeJpeg;DecodeJpeg->retval_DecodeJpeg_0_0");
}
} // end namespace
} // end tensorflow
