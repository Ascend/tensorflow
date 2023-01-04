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
namespace {
class NpuAttrTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};
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
}
} // end tensorflow
