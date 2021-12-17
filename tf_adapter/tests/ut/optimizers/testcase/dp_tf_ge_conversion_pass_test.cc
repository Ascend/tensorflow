#include "tf_adapter/optimizers/dp_tf_ge_conversion_pass.h"
#include "tf_adapter/util/npu_attrs.h"
#include "gtest/gtest.h"
#include "mmpa/mmpa_api.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include <stdlib.h>

namespace tensorflow {
namespace {
class DpOptimizationPassTest : public testing::Test {
 public:
  DpOptimizationPassTest() : graph_(absl::make_unique<Graph>(OpRegistry::Global())) {}
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

  string DoRunDpOptimizationPassTest() {
    string before = CanonicalGraphString(graph_.get());
    LOG(INFO) << "Before om conversion pass: " << before;

    std::unique_ptr<Graph> *ug = &graph_;
    GraphOptimizationPassOptions options;
    options.graph = ug;
    FunctionLibraryDefinition flib_def((*ug)->flib_def());
    options.flib_def = &flib_def;
    DpTfToGEConversionPass().Run(options);

    string result = CanonicalGraphString(options.graph->get());
    LOG(INFO) << "After om conversion pass: " << result;
    return result;
  }

  const string &OriginalGraph() const { return original_; }

  std::unique_ptr<Graph> graph_;
  string original_;
 protected:
  virtual void SetUp() { *const_cast<bool *>(&kDumpGraph) = true; }
  virtual void TearDown() {}
};

TEST_F(DpOptimizationPassTest, BuildGeOpTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/dp_test_build_geop.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "Const->TensorSliceDataset;IteratorV2->MakeIterator:1;TensorSliceDataset->HostQueueDataset:1;"\
    "HostQueueDataset->DPGroupDataset;GEOPDataset->HostQueueDataset;DPGroupDataset->MakeIterator";
  EXPECT_EQ(DoRunDpOptimizationPassTest(), target_graph);
}
TEST_F(DpOptimizationPassTest, DatasetNotInDeviceTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/dp_test_no_dataset_in_device.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "Const->TensorSliceDataset;TensorSliceDataset->BatchDatasetV2;Const->BatchDatasetV2:1;"\
    "Const->BatchDatasetV2:2;BatchDatasetV2->RepeatDataset;Const->RepeatDataset:1;RepeatDataset->OptimizeDataset;"\
    "Const->OptimizeDataset:1;OptimizeDataset->ModelDataset;IteratorV2->MakeIterator:1;ModelDataset->HostQueueDataset:1;"\
    "HostQueueDataset->DPGroupDataset;GEOPDataset->HostQueueDataset;DPGroupDataset->MakeIterator";
  EXPECT_EQ(DoRunDpOptimizationPassTest(), target_graph);
}
TEST_F(DpOptimizationPassTest, NewDatasetNotInDeviceTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/dp_test_no_dataset_in_device.pbtxt";
  *const_cast<bool *>(&kIsNewDataTransfer) = true;
  InitGraph(org_graph_def_path);
  std::string target_graph = "Const->TensorSliceDataset;TensorSliceDataset->BatchDatasetV2;Const->BatchDatasetV2:1;"\
    "Const->BatchDatasetV2:2;BatchDatasetV2->RepeatDataset;Const->RepeatDataset:1;RepeatDataset->OptimizeDataset;"\
    "Const->OptimizeDataset:1;OptimizeDataset->ModelDataset;IteratorV2->MakeIterator:1;ModelDataset->HostQueueDataset:1;"\
    "HostQueueDataset->DPGroupDataset;GEOPDataset->HostQueueDataset;DPGroupDataset->MakeIterator";
  EXPECT_EQ(DoRunDpOptimizationPassTest(), target_graph);
  *const_cast<bool *>(&kIsNewDataTransfer) = false;
}
} // end namespace
} // end tensorflow
