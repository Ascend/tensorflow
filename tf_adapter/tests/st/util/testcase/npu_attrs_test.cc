#include "tf_adapter/util/npu_attrs.h"
#include "gtest/gtest.h"
#include <stdlib.h>

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
}
} // end tensorflow
