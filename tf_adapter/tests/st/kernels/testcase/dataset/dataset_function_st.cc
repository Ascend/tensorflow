#include <stdlib.h>
#include <vector>
#include <fstream>
#include <cstdlib>

#include "securec.h"
#include "gtest/gtest.h"
#include "mmpa/mmpa_api.h"
#include "ascendcl_stub.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#define private public
#include "tf_adapter/kernels/aicpu/dataset_function.h"

namespace tensorflow {
namespace data {
namespace {
class DatasetFunctionTest : public testing::Test {
 public:
  DatasetFunctionTest() : graph_(absl::make_unique<Graph>(OpRegistry::Global())) {}

  static void InitGraph(const string &graph_def_path, Graph *graph) {
    GraphDef graph_def;
    ReadTextProto(Env::Default(), graph_def_path, &graph_def);
    GraphConstructorOptions opts;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
  }

  void InitGraph(const string &graph_def_path, Graph *graph, std::string &result_graph) {
    char trusted_path[MMPA_MAX_PATH] = { "\0" };
    if (mmRealPath(graph_def_path.c_str(), trusted_path, MMPA_MAX_PATH) != EN_OK) {
      ADP_LOG(ERROR) << "Get real path failed.";
      return;
    }
    ADP_LOG(INFO) << "input graph def path: " << trusted_path;
    InitGraph(trusted_path, graph);
    result_graph = CanonicalGraphString(graph_.get());
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

  void InitDatasetFunction() {
    std::map<std::string, std::string> init_options;
    init_options["key"] = "value";
    DataTypeVector input_types;
    DataTypeVector output_types;

    input_types.emplace_back(tensorflow::DataType::DT_STRING);
    input_types.emplace_back(tensorflow::DataType::DT_INT64);
    input_shapes_.emplace_back(PartialTensorShape({}));
    input_shapes_.emplace_back(PartialTensorShape({}));
    output_types.emplace_back(tensorflow::DataType::DT_INT32);
    output_types.emplace_back(tensorflow::DataType::DT_UINT8);
    output_shapes_.emplace_back(PartialTensorShape({}));
    output_shapes_.emplace_back(PartialTensorShape({}));

    dataset_func_ = absl::make_unique<DatasetFunction>(init_options, func_name_,
        input_types, output_types, input_shapes_, output_shapes_);
  }

  const string &OriginalGraph() const { return original_; }

  std::unique_ptr<Graph> graph_;
  std::string original_;
  std::unique_ptr<DatasetFunction> dataset_func_;
  std::vector<PartialTensorShape> input_shapes_;
  std::vector<PartialTensorShape> output_shapes_;
  std::string target_path_ = "./built-in/framework/tensorflow/";
  std::string func_name_ = "dp_sub_graph";
  char* ascend_opp_path_;
 protected:
  void SetUp() {
    InitDatasetFunction();
    ascend_opp_path_ = getenv("ASCEND_OPP_PATH");
    setenv("ASCEND_OPP_PATH", "./", true /*overwrite*/);
    std::string dp_split_graph_file =
      "tf_adapter/tests/depends/support_json/framework/built-in/tensorflow/dp_split_graph_ops.json";
    std::string create_file_cmd("mkdir -p " + target_path_ + "; cp " + dp_split_graph_file + " " + target_path_);
    int ret = system(create_file_cmd.c_str());
    if (ret) {
      ADP_LOG(ERROR) << "DatasetFunctionTest create file failed.";
    }
  }

  void TearDown() {
    // must reset ASCEND_OPP_PATH as before.
    setenv("ASCEND_OPP_PATH", ascend_opp_path_, true /*overwrite*/);
    std::string delete_file_cmd("rm -rf " + target_path_);
    int ret = system(delete_file_cmd.c_str());
    if (ret) {
      ADP_LOG(ERROR) << "DatasetFunctionTest delete file failed.";
    }
  }
};

TEST_F(DatasetFunctionTest, MakeCompatShapeTest) {
  ADP_LOG(INFO) << "====== DatasetFunctionTest MakeCompatShapeTest ======";

  PartialTensorShape shape_a = PartialTensorShape({3, 3});
  PartialTensorShape shape_b = PartialTensorShape({3, 3});
  PartialTensorShape ret = dataset_func_->MakeCompatShape(shape_a, shape_b);
  int expected_dims = shape_a.dims();
  ADP_LOG(INFO) << "Result dims is " << ret.dims();
  EXPECT_EQ(expected_dims, ret.dims());

  PartialTensorShape shape_c = PartialTensorShape({3});
  PartialTensorShape kUnknownRankShape = PartialTensorShape();
  ret = dataset_func_->MakeCompatShape(shape_a, shape_c);
  ADP_LOG(INFO) << "Result dims is " << ret.dims();
  EXPECT_EQ(kUnknownRankShape.dims(), ret.dims());
}

TEST_F(DatasetFunctionTest, ReadJsonFileTest) {
  ADP_LOG(INFO) << "====== DatasetFunctionTest ReadJsonFileTest ======";

  std::string dp_split_graph_file =
    "tf_adapter/tests/depends/support_json/framework/built-in/tensorflow/dp_split_graph_ops.json";
  nlohmann::json json_data;
  TF_CHECK_OK(dataset_func_->ReadJsonFile(dp_split_graph_file, json_data));
}

TEST_F(DatasetFunctionTest, InitAccelateOpListFailedTest) {
  ADP_LOG(INFO) << "====== DatasetFunctionTest InitAccelateOpListFailedTest ======";

  std::string wrong_chip_name = "Ascend910A";
  aclrtSetSocNameStub(wrong_chip_name);
  std::vector<std::string> acc_while_list;
  Status status = dataset_func_->InitAccelateOpList(acc_while_list);
  EXPECT_EQ(status.ok(), false);
  EXPECT_EQ(acc_while_list.size(), 0);
  aclrtSetDefaultSocNameStub();
}


TEST_F(DatasetFunctionTest, InitAccelateOpListSuccessTest) {
  ADP_LOG(INFO) << "====== DatasetFunctionTest InitAccelateOpListSuccessTest ======";

  //1971chips : Ascend910B1 Ascend910B2 Ascend910B3 ...
  std::string correct_chip_name = "Ascend910B1";
  aclrtSetSocNameStub(correct_chip_name);

  std::vector<std::string> acc_while_list;
  Status status = dataset_func_->InitAccelateOpList(acc_while_list);
  EXPECT_EQ(status.ok(), true);
  EXPECT_EQ(acc_while_list.empty(), false);
  aclrtSetDefaultSocNameStub();
}

TEST_F(DatasetFunctionTest, SplitGraphTest) {
  ADP_LOG(INFO) << "====== DatasetFunctionTest SplitGraphTest ======";

  std::string org_graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/dp_sub_graph.pbtxt";
  InitGraph(org_graph_def_path, graph_.get(), original_);

  std::string correct_chip_name = "Ascend910B1";
  aclrtSetSocNameStub(correct_chip_name);
  std::vector<std::string> acc_while_list;
  Status status = dataset_func_->InitAccelateOpList(acc_while_list);
  EXPECT_EQ(status.ok(), true);
  EXPECT_EQ(acc_while_list.empty(), false);

  string orig_graph = CanonicalGraphString(graph_.get());
  ADP_LOG(INFO) << "Original graph : " << orig_graph;

  std::unique_ptr<Graph> *ug = &graph_;
  FunctionLibraryDefinition flib_def((*ug)->flib_def());
  tensorflow::FunctionDefLibrary flib;
  TF_CHECK_OK(tensorflow::GraphToFunctionDef(*(graph_.get()), func_name_, flib.add_function()));
  TF_CHECK_OK(flib_def.AddLibrary(flib));

  // dump graph : DatasetFunc_DpBuild_tf_HostGraph.pbtxt and DatasetFunc_DpBuild_tf_DvppGraph.pbtxt
  std::string graph_def = dataset_func_->SplitSubGraph(flib_def, acc_while_list);
  std::string output_dvpp_graph = "DatasetFunc_DpBuild_tf_DvppGraph.pbtxt";
  std::string dvpp_graph_infos = "";
  std::unique_ptr<tensorflow::Graph> dvpp_graph = absl::make_unique<Graph>(OpRegistry::Global());
  InitGraph(org_graph_def_path, dvpp_graph.get(), dvpp_graph_infos);

  std::string target_graph = "_Arg->ParseSingleExample;Const->ParseSingleExample:1;Const->ParseSingleExample:2;"\
    "Const->ParseSingleExample:3;ParseSingleExample:12->Cast;ParseSingleExample:14->DecodeAndCropJpeg;"\
    "Const->DecodeAndCropJpeg:1;DecodeAndCropJpeg->_Retval;Cast->_Retval";
  EXPECT_EQ(dvpp_graph_infos, target_graph);
  aclrtSetDefaultSocNameStub();
}

TEST_F(DatasetFunctionTest, CreateAclModelDescFailedTest) {
  ADP_LOG(INFO) << "====== DatasetFunctionTest CreateAclModelDescFailedTest ======";
  SetModelDescStub(true);

  uint32_t model_id = 0UL;
  auto  model_desc = dataset_func_->CreateAclModelDesc(model_id);
  EXPECT_EQ(model_desc, nullptr);
  SetModelDescStub(false);
}

TEST_F(DatasetFunctionTest, CreateAclDatasetFailedTest) {
  ADP_LOG(INFO) << "====== DatasetFunctionTest CreateAclDatasetFailedTest ======";
  SetCreateDataset(false);

  uint32_t model_id = 0UL;
  auto acl_dataset = dataset_func_->CreateAclOutputDataset(model_id);
  EXPECT_EQ(acl_dataset, nullptr);
  SetCreateDataset(true);
}


TEST_F(DatasetFunctionTest, CreateTensorDescFailedTest) {
  ADP_LOG(INFO) << "====== DatasetFunctionTest CreateTensorDescFailedTest ======";
  SetCreateTensorDesc(false);

  uint32_t model_id = 0UL;
  auto acl_dataset = dataset_func_->CreateAclOutputDataset(model_id);
  EXPECT_EQ(acl_dataset, nullptr);
  SetCreateTensorDesc(true);
}
}  // namespace
}  // namespace data
}  // namespace tensorflow
