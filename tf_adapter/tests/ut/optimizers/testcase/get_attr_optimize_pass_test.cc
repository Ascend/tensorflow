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

namespace tensorflow {
class GetAttrOptimizePass : public GraphOptimizationPass {
 public:
  GetAttrOptimizePass() = default;
  ~GetAttrOptimizePass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;
};

namespace {
class GetAttrOptimizationPassTest : public testing::Test {
 public:
  GetAttrOptimizationPassTest() : graph_(absl::make_unique<Graph>(OpRegistry::Global())) {}
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

  string DoRunGetAttrOptimizationPassTest(SessionOptions session_options) {
    string before = CanonicalGraphString(graph_.get());
    LOG(INFO) << "Before om conversion pass: " << before;

    std::unique_ptr<Graph> *ug = &graph_;
    GraphOptimizationPassOptions options;
    options.session_options = &session_options;
    options.graph = ug;
    FunctionLibraryDefinition flib_def((*ug)->flib_def());
    options.flib_def = &flib_def;
    GetAttrOptimizePass().Run(options);

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

TEST_F(GetAttrOptimizationPassTest, SetAttrTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/get_attr_job_chief_test.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "VariableV2->Identity;Const->Add;Identity->Add:1;Add->_Retval";
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
    ->mutable_optimizer_options()
    ->set_do_function_inlining(true);
  auto *custom_config = session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  AttrValue job = AttrValue();
  job.set_s("chief");
  (*custom_config->mutable_parameter_map())["job"] = job;
  AttrValue enable_data_pre_proc = AttrValue();
  enable_data_pre_proc.set_b(true);
  (*custom_config->mutable_parameter_map())["enable_data_pre_proc"] = enable_data_pre_proc;
  AttrValue dynamic_input = AttrValue();
  dynamic_input.set_b(true);
  (*custom_config->mutable_parameter_map())["dynamic_input"] = dynamic_input;
  AttrValue dynamic_graph_execute_mode = AttrValue();
  dynamic_graph_execute_mode.set_s("lazy_recompile");
  (*custom_config->mutable_parameter_map())["dynamic_graph_execute_mode"] = dynamic_graph_execute_mode;
  AttrValue local_rank_id = AttrValue();
  local_rank_id.set_i(0);
  (*custom_config->mutable_parameter_map())["local_rank_id"] = local_rank_id;
  AttrValue local_device_list = AttrValue();
  local_device_list.set_s("0,1");
  (*custom_config->mutable_parameter_map())["local_device_list"] = local_device_list;
  AttrValue enable_dump = AttrValue();
  enable_dump.set_b(true);
  (*custom_config->mutable_parameter_map())["enable_dump"] = enable_dump;
  AttrValue dump_path = AttrValue();
  dump_path.set_s("./");
  (*custom_config->mutable_parameter_map())["dump_path"] = dump_path;
  AttrValue dump_step = AttrValue();
  dump_step.set_s("1");
  (*custom_config->mutable_parameter_map())["dump_step"] = dump_step;
  AttrValue dump_mode = AttrValue();
  dump_mode.set_s("all");
  (*custom_config->mutable_parameter_map())["dump_mode"] = dump_mode;
  AttrValue enable_dump_debug = AttrValue();
  enable_dump_debug.set_b(true);
  (*custom_config->mutable_parameter_map())["enable_dump_debug"] = enable_dump_debug;
  AttrValue dump_debug_mode = AttrValue();
  dump_debug_mode.set_s("all");
  (*custom_config->mutable_parameter_map())["dump_debug_mode"] = dump_debug_mode;
  AttrValue profiling_mode = AttrValue();
  profiling_mode.set_b(true);
  (*custom_config->mutable_parameter_map())["profiling_mode"] = profiling_mode;
  AttrValue profiling_options = AttrValue();
  profiling_options.set_s("1");
  (*custom_config->mutable_parameter_map())["profiling_options"] = profiling_options;
  AttrValue graph_run_mode = AttrValue();
  graph_run_mode.set_i(1);
  (*custom_config->mutable_parameter_map())["graph_run_mode"] = graph_run_mode;
  AttrValue mstune_mode = AttrValue();
  mstune_mode.set_s("2");
  (*custom_config->mutable_parameter_map())["mstune_mode"] = mstune_mode;
  AttrValue op_tune_mode = AttrValue();
  op_tune_mode.set_s("GA");
  (*custom_config->mutable_parameter_map())["op_tune_mode"] = op_tune_mode;
  AttrValue work_path = AttrValue();
  work_path.set_s("./");
  (*custom_config->mutable_parameter_map())["work_path"] = work_path;
  AttrValue input_shape = AttrValue();
  input_shape.set_s("data:1,1,40,-1;lable:1,-1;mask:-1,-1");
  (*custom_config->mutable_parameter_map())["input_shape"] = input_shape;
  AttrValue dynamic_dims = AttrValue();
  dynamic_dims.set_s("20,20,1,1;40,40,2,2;80,60,4,4");
  (*custom_config->mutable_parameter_map())["dynamic_dims"] = dynamic_dims;
  AttrValue dynamic_node_type = AttrValue();
  dynamic_node_type.set_s("1");
  (*custom_config->mutable_parameter_map())["dynamic_node_type"] = dynamic_node_type;
  AttrValue buffer_optimize = AttrValue();
  buffer_optimize.set_s("l2_optimize");
  (*custom_config->mutable_parameter_map())["buffer_optimize"] = buffer_optimize;
  AttrValue op_select_implmode = AttrValue();
  op_select_implmode.set_s("high_performance");
  (*custom_config->mutable_parameter_map())["op_select_implmode"] = op_select_implmode;
  AttrValue optypelist_for_implmode = AttrValue();
  optypelist_for_implmode.set_s("Add");
  (*custom_config->mutable_parameter_map())["optypelist_for_implmode"] = optypelist_for_implmode;
  AttrValue op_compiler_cache_mode = AttrValue();
  op_compiler_cache_mode.set_s("Add");
  (*custom_config->mutable_parameter_map())["op_compiler_cache_mode"] = op_compiler_cache_mode;
  AttrValue op_compiler_cache_dir = AttrValue();
  op_compiler_cache_dir.set_s("./");
  (*custom_config->mutable_parameter_map())["op_compiler_cache_dir"] = op_compiler_cache_dir;
  AttrValue debug_dir = AttrValue();
  debug_dir.set_s("./");
  (*custom_config->mutable_parameter_map())["debug_dir"] = debug_dir;
  AttrValue session_device_id = AttrValue();
  session_device_id.set_i(1);
  (*custom_config->mutable_parameter_map())["session_device_id"] = session_device_id;
  AttrValue aoe_mode = AttrValue();
  aoe_mode.set_s("1");
  (*custom_config->mutable_parameter_map())["aoe_mode"] = aoe_mode;
  AttrValue op_wait_timeout = AttrValue();
  op_wait_timeout.set_i(1);
  (*custom_config->mutable_parameter_map())["op_wait_timeout"] = op_wait_timeout;
  AttrValue op_execute_timeout = AttrValue();
  op_execute_timeout.set_i(1);
  (*custom_config->mutable_parameter_map())["op_execute_timeout"] = op_execute_timeout;
  EXPECT_EQ(DoRunGetAttrOptimizationPassTest(session_options), target_graph);
}
TEST_F(GetAttrOptimizationPassTest, NotSetAttrTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/get_attr_job_chief_test.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "VariableV2->Identity;Const->Add;Identity->Add:1;Add->_Retval";
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
    ->mutable_optimizer_options()
    ->set_do_function_inlining(true);
  auto *custom_config = session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  AttrValue job = AttrValue();
  job.set_s("chief");
  (*custom_config->mutable_parameter_map())["job"] = job;
  EXPECT_EQ(DoRunGetAttrOptimizationPassTest(session_options), target_graph);
}
TEST_F(GetAttrOptimizationPassTest, SkipPassTest) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/get_attr_job_chief_test.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "VariableV2->Identity;Const->Add;Identity->Add:1;Add->_Retval";
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
    ->mutable_optimizer_options()
    ->set_do_function_inlining(true);
  auto *custom_config = session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  AttrValue job = AttrValue();
  job.set_s("ps");
  (*custom_config->mutable_parameter_map())["job"] = job;
  EXPECT_EQ(DoRunGetAttrOptimizationPassTest(session_options), target_graph);
}
TEST_F(GetAttrOptimizationPassTest, SkipPass1Test) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/get_attr_no_need_opt.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "VariableV2->Identity;Const->Add;Identity->Add:1;Add->_Retval";
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
    ->mutable_optimizer_options()
    ->set_do_function_inlining(true);
  auto *custom_config = session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  AttrValue job = AttrValue();
  job.set_s("chief");
  (*custom_config->mutable_parameter_map())["job"] = job;
  EXPECT_EQ(DoRunGetAttrOptimizationPassTest(session_options), target_graph);
}
TEST_F(GetAttrOptimizationPassTest, SkipPass2Test) {
  string org_graph_def_path = "tf_adapter/tests/ut/optimizers/pbtxt/get_attr_npu_optimize.pbtxt";
  InitGraph(org_graph_def_path);
  std::string target_graph = "VariableV2->Identity;Const->Add;Identity->Add:1;Add->_Retval";
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
    ->mutable_optimizer_options()
    ->set_do_function_inlining(true);
  auto *custom_config = session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  AttrValue job = AttrValue();
  job.set_s("chief");
  (*custom_config->mutable_parameter_map())["job"] = job;
  EXPECT_EQ(DoRunGetAttrOptimizationPassTest(session_options), target_graph);
}
} // end namespace
} // end tensorflow
