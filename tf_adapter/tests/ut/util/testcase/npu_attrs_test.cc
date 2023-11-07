#include "tf_adapter/util/npu_attrs.h"
#include "gtest/gtest.h"
#include <stdlib.h>
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {
Status CheckOpImplMode(const string &op_select_implmode);
Status CheckVariablePlacement(const std::string &variable_placement);
namespace {
class NpuAttrTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(NpuAttrTest, CheckIsNewDataTransfer) {
  NpuAttrs::SetNewDataTransferFlag(false);
  bool ret = NpuAttrs::GetNewDataTransferFlag();
  EXPECT_EQ(ret, true);
  NpuAttrs::SetNewDataTransferFlag(true);
}

TEST_F(NpuAttrTest, GetEnvDeviceIdDefaultTest) {
  uint32_t device_id = 0;
  (void)GetEnvDeviceID(device_id);
  EXPECT_EQ(device_id, 0);
}
TEST_F(NpuAttrTest, GetEnvAscendDeviceIdEmptyTest) {
  uint32_t device_id = 0;
  setenv("DEVICE_ID", "1", true);
  (void)GetEnvDeviceID(device_id);
  EXPECT_EQ(device_id, 1);
}
TEST_F(NpuAttrTest, GetEnvDeviceIdFailTest) {
  uint32_t device_id = 0;
  setenv("DEVICE_ID", "-1", true);
  Status s = GetEnvDeviceID(device_id);
  EXPECT_EQ(s.ok(), false);
}
TEST_F(NpuAttrTest, GetEnvDeviceIdNotIntFailTest) {
  uint32_t device_id = 0;
  setenv("DEVICE_ID", "1.1", true);
  Status s = GetEnvDeviceID(device_id);
  EXPECT_EQ(s.ok(), false);
}
TEST_F(NpuAttrTest, GetEnvAscendDeviceIdNotIntFailTest) {
  uint32_t device_id = 0;
  setenv("ASCEND_DEVICE_ID", "1.1", true);
  Status s = GetEnvDeviceID(device_id);
  EXPECT_EQ(s.ok(), false);
}
TEST_F(NpuAttrTest, GetEnvDeviceIdEmptyTest) {
  uint32_t device_id = 0;
  setenv("ASCEND_DEVICE_ID", "1", true);
  (void)GetEnvDeviceID(device_id);
  EXPECT_EQ(device_id, 1);
}
TEST_F(NpuAttrTest, GetEnvAscendDeviceIdFailTest) {
  uint32_t device_id = 0;
  setenv("ASCEND_DEVICE_ID", "-aa", true);
  Status s = GetEnvDeviceID(device_id);
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, GetStepFromEnv) {
  uint32_t step = 0;
  Status s = GetStepFromEnv("STEP_NOW", step);
  EXPECT_EQ(s.ok(), false);
  setenv("STEP_NOW", "1000", true);
  s = GetStepFromEnv("STEP_NOW", step);
  EXPECT_EQ(s.ok(), true);
  EXPECT_EQ(step, 1000);
  setenv("STEP_NOW", "1.1", true);
  s = GetStepFromEnv("STEP_NOW", step);
  EXPECT_EQ(s.ok(), false);
  unsetenv("STEP_NOW");
}

TEST_F(NpuAttrTest, GetLossFromEnv) {
  float loss = 0;
  Status s = GetLossFromEnv("LOSS_NOW", loss);
  EXPECT_EQ(s.ok(), false);
  setenv("LOSS_NOW", "1.1", true);
  s = GetLossFromEnv("LOSS_NOW", loss);
  EXPECT_EQ(s.ok(), true);
  EXPECT_FLOAT_EQ(loss, 1.1);
  unsetenv("LOSS_NOW");
}

TEST_F(NpuAttrTest, SplitTest) {
  std::string s = "a,b,c";
  std::vector<std::string> res;
  Split(s, res, ",");
  EXPECT_EQ(res[2], "c");
}
TEST_F(NpuAttrTest, SetNpuOptimizerAttr) {
  Status s = CheckOpImplMode("xxx");
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, CheckAoeMode) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_do_function_inlining(true);
  auto *custom_config =
      session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;

  AttrValue aoe_mode = AttrValue();
  aoe_mode.set_s("3");
  (*custom_config->mutable_parameter_map())["aoe_mode"] = aoe_mode;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, CheckPrecisionMode ) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_do_function_inlining(true);
  auto *custom_config =
      session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;

  AttrValue precision_mode = AttrValue();
  precision_mode.set_s("force_Dp32");
  (*custom_config->mutable_parameter_map())["precision_mode"] = precision_mode;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, reinterpret_cast<Node *>(1));
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, CheckModifyMixList_Failed) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_do_function_inlining(true);
  auto *custom_config =
      session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;

  AttrValue modify_mixlist = AttrValue();
  modify_mixlist.set_s("list");
  (*custom_config->mutable_parameter_map())["modify_mixlist"] = modify_mixlist;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, reinterpret_cast<Node *>(1));
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, CheckModifyMixList_Failed_WithWrongPrecisonModeV2) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_do_function_inlining(true);
  auto *custom_config =
      session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;
  
  AttrValue modify_mixlist = AttrValue();
  modify_mixlist.set_s("list");
  (*custom_config->mutable_parameter_map())["modify_mixlist"] = modify_mixlist;

  AttrValue precision_mode_v2 = AttrValue();
  precision_mode_v2.set_s("fp16");
  (*custom_config->mutable_parameter_map())["precision_mode_v2"] = precision_mode_v2;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, reinterpret_cast<Node *>(1));
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, CheckModifyMixList_Failed_WithWrongPrecisionMode) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_do_function_inlining(true);
  auto *custom_config =
      session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;
  
  AttrValue modify_mixlist = AttrValue();
  modify_mixlist.set_s("list");
  (*custom_config->mutable_parameter_map())["modify_mixlist"] = modify_mixlist;

  AttrValue precision_mode = AttrValue();
  precision_mode.set_s("force_fp16");
  (*custom_config->mutable_parameter_map())["precision_mode"] = precision_mode;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, reinterpret_cast<Node *>(1));
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, CheckPrecisionModeV2) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_do_function_inlining(true);
  auto *custom_config =
      session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;

  AttrValue precision_mode_v2 = AttrValue();
  precision_mode_v2.set_s("invalid");
  (*custom_config->mutable_parameter_map())["precision_mode_v2"] = precision_mode_v2;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, reinterpret_cast<Node *>(1));
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, CheckPrecisionModeV2_Failed_WhenAssignedBoth) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_do_function_inlining(true);
  auto *custom_config =
      session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;

  AttrValue precision_mode_v2 = AttrValue();
  precision_mode_v2.set_s("fp16");
  (*custom_config->mutable_parameter_map())["precision_mode_v2"] = precision_mode_v2;

  AttrValue precision_mode = AttrValue();
  precision_mode.set_s("force_fp32");
  (*custom_config->mutable_parameter_map())["precision_mode"] = precision_mode;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, reinterpret_cast<Node *>(1));
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, CheckJitCompile) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  auto *custom_config =
      session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;
  AttrValue jit_compile = AttrValue();
  AttrValue graph_slice = AttrValue();
  jit_compile.set_b(true);
  (*custom_config->mutable_parameter_map())["jit_compile"] = jit_compile;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, reinterpret_cast<Node *>(1));
  EXPECT_EQ(s.ok(), false);
  jit_compile.clear_b();
  jit_compile.set_s("True");
  (*custom_config->mutable_parameter_map())["jit_compile"] = jit_compile;
  s = NpuAttrs::SetNpuOptimizerAttr(options, reinterpret_cast<Node *>(1));
  EXPECT_EQ(s.ok(), false);
  jit_compile.clear_s();
  jit_compile.set_s("false");
  (*custom_config->mutable_parameter_map())["jit_compile"] = jit_compile;
  graph_slice.set_s("000");
  (*custom_config->mutable_parameter_map())["graph_slice"] = graph_slice;
  s = NpuAttrs::SetNpuOptimizerAttr(options, reinterpret_cast<Node *>(1));
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, CheckVariablePlacement) {
  Status s = CheckVariablePlacement("sss");
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, GetDumpPath) {
setenv("DUMP_GRAPH_PATH", "./", 1);
string path = GetDumpPath();
EXPECT_EQ(path, ".//");
setenv("DUMP_GRAPH_PATH", "./dump_fold", 1);
string new_path = GetDumpPath();
EXPECT_EQ(new_path, "./dump_fold/");
}

TEST_F(NpuAttrTest, GetCollectionPath) {
setenv("NPU_COLLECT_PATH", "./collection", 1);
setenv("DUMP_GRAPH_PATH", "./dump_fold", 1);
string new_path = GetDumpPath();
EXPECT_NE(new_path, "./dump_fold/");
}

TEST_F(NpuAttrTest, SetNpuOptimizerAttrInvalidEnableDump) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  auto *custom_config = session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  AttrValue enable_dump_debug = AttrValue();
  enable_dump_debug.set_b(true);
  (*custom_config->mutable_parameter_map())["enable_dump_debug"] = enable_dump_debug;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  AttrValue dump_path = AttrValue();
  dump_path.set_s("/invalid");
  (*custom_config->mutable_parameter_map())["dump_path"] = dump_path;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  dump_path.set_s("/");
  (*custom_config->mutable_parameter_map())["dump_path"] = dump_path;
  AttrValue dump_step = AttrValue();
  dump_step.set_s("777");
  (*custom_config->mutable_parameter_map())["dump_step"] = dump_step;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  enable_dump_debug.set_b(false);
  (*custom_config->mutable_parameter_map())["enable_dump_debug"] = enable_dump_debug;
  AttrValue local_rank_id = AttrValue();
  local_rank_id.set_i(777);
  (*custom_config->mutable_parameter_map())["local_rank_id"] = local_rank_id;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  local_rank_id.set_i(0);
  (*custom_config->mutable_parameter_map())["local_rank_id"] = local_rank_id;
  AttrValue local_device_list = AttrValue();
  local_device_list.set_s("invalid string");
  (*custom_config->mutable_parameter_map())["local_device_list"] = local_device_list;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  AttrValue dynamic_input = AttrValue();
  dynamic_input.set_b(true);
  (*custom_config->mutable_parameter_map())["dynamic_input"] = dynamic_input;
  AttrValue dynamic_graph_execute_mode = AttrValue();
  dynamic_graph_execute_mode.set_s("execute mode");
  (*custom_config->mutable_parameter_map())["dynamic_graph_execute_mode"] = dynamic_graph_execute_mode;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  AttrValue variable_format_optimize = AttrValue();
  variable_format_optimize.set_b(true);
  (*custom_config->mutable_parameter_map())["variable_format_optimize"] = variable_format_optimize;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  AttrValue op_debug_level = AttrValue();
  op_debug_level.set_i(2);
  (*custom_config->mutable_parameter_map())["op_debug_level"] = op_debug_level;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  AttrValue enable_data_pre_proc = AttrValue();
  enable_data_pre_proc.set_b(false);
  (*custom_config->mutable_parameter_map())["enable_data_pre_proc"] = enable_data_pre_proc;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  AttrValue op_select_implmode = AttrValue();
  op_select_implmode.set_s("high_precision");
  (*custom_config->mutable_parameter_map())["op_select_implmode"] = op_select_implmode;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  AttrValue optypelist_for_implmode = AttrValue();
  optypelist_for_implmode.set_s("Pooling,SoftmaxV2");
  (*custom_config->mutable_parameter_map())["optypelist_for_implmode"] = optypelist_for_implmode;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, SetNpuOptimizerAttrInvalidEnableOnlineInference) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  auto *custom_config = session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);

  AttrValue graph_run_mode = AttrValue();
  graph_run_mode.set_i(0);
  (*custom_config->mutable_parameter_map())["graph_run_mode"] = graph_run_mode;
  s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_EQ(s.ok(), false);
}

TEST_F(NpuAttrTest, CheckGraphCompilerCacheDir) {
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_do_function_inlining(true);
  auto *custom_config =
      session_options.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
  custom_config->set_name("NpuOptimizer");
  options.session_options = &session_options;

  AttrValue graph_compiler_cache_dir = AttrValue();
  graph_compiler_cache_dir.set_s("./cache_dir");
  (*custom_config->mutable_parameter_map())["graph_compiler_cache_dir"] = graph_compiler_cache_dir;
  Status s = NpuAttrs::SetNpuOptimizerAttr(options, nullptr);
  EXPECT_FALSE(s.ok());

  AttrValueMap attr_map;
  AttrValue npu_optimizer = AttrValue();
  npu_optimizer.set_s("NpuOptimizer");
  attr_map["_NpuOptimizer"] = npu_optimizer;
  attr_map["_graph_compiler_cache_dir"] = graph_compiler_cache_dir;
  AttrSlice attrs(&attr_map);
  const auto &all_options = NpuAttrs::GetAllAttrOptions(attrs);
  auto find_ret = all_options.find("graph_compiler_cache_dir");
  ASSERT_TRUE(find_ret != all_options.cend());
  EXPECT_EQ(find_ret->second, "./cache_dir");
}

TEST_F(NpuAttrTest, GetNpuOptimizerAttrCheckDumpStep) {
  AttrValueMap attr_map;

  AttrValue graph_compiler_cache_dir = AttrValue();
  graph_compiler_cache_dir.set_s("./cache_dir");
  attr_map["_graph_compiler_cache_dir"] = graph_compiler_cache_dir;

  AttrValue npu_optimizer = AttrValue();
  npu_optimizer.set_s("NpuOptimizer");
  attr_map["_NpuOptimizer"] = npu_optimizer;

  AttrValue enable_dump = AttrValue();
  enable_dump.set_s("1");
  attr_map["_enable_dump"] = enable_dump;

  AttrValue dump_step = AttrValue();
  dump_step.set_s("yyy");
  attr_map["_dump_step"] = dump_step;

  AttrSlice attrs(&attr_map);
  const auto &all_options = NpuAttrs::GetAllAttrOptions(attrs);
  ASSERT_TRUE(all_options.find("dump_step") != all_options.cend());

  AttrValue dump_step_2 = AttrValue();
  dump_step_2.set_s("0|2-1");
  attr_map["_dump_step"] = dump_step_2;

  AttrSlice attrs2(&attr_map);
  const auto &all_options2 = NpuAttrs::GetAllAttrOptions(attrs2);
  ASSERT_TRUE(all_options2.find("dump_step") != all_options2.cend());
}
}
} // end tensorflow
