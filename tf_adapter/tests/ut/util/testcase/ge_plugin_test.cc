#include "tf_adapter/util/npu_plugin.h"
#include "tf_adapter/util/npu_attrs.h"
#include "gtest/gtest.h"
#include <stdlib.h>

namespace tensorflow {
namespace {
class GePluginTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(GePluginTest, PluginInitTest) {
  std::map<std::string, std::string> init_options;
  setenv("JOB_ID", "1000", true);
  setenv("RANK_SIZE", "1", true);
  setenv("RANK_ID", "0", true);
  setenv("POD_NAME", "0", true);
  setenv("RANK_TABLE_FILE", "rank_table", true);
  setenv("FUSION_TENSOR_SIZE", "524288000", true);
  std::string tf_config = "{'task':{'type':'a'}, 'cluster':{'chief':['1']}}";
  setenv("TF_CONFIG", tf_config.c_str(), true);
  init_options["ge.exec.profilingMode"] = "1";
  init_options["ge.exec.profilingOptions"] = "trace";
  init_options["ge.exec.precision_mode"] = "allow_fp32_to_fp16";
  init_options["ge.autoTuneMode"] = "GA";
  init_options["ge.opDebugLevel"] = "1";
  init_options["ge.jobType"] = "2";
  PluginInit(init_options);
}

TEST_F(GePluginTest, PluginFinalizeTest) {
  PluginFinalize();
}
TEST_F(GePluginTest, InitRdmaPoolOKTest) {
  int32_t ret = InitRdmaPool(1);
  EXPECT_EQ(ret, 0);
}
TEST_F(GePluginTest, InitRdmaPoolFaliedTest) {
  int32_t ret = InitRdmaPool(0);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, RegistRdmaRemoteAddrFailedTest) {
  std::vector<ge::HostVarInfo> var_info;
  int32_t ret = RegistRdmaRemoteAddr(var_info);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, RegistRdmaRemoteAddrOKTest) {
  std::vector<ge::HostVarInfo> var_info;
  ge::HostVarInfo host_var_info;
  host_var_info.base_addr = 0;
  var_info.push_back(host_var_info);
  int32_t ret = RegistRdmaRemoteAddr(var_info);
  EXPECT_EQ(ret, 0);
}
TEST_F(GePluginTest, GetVarAddrAndSizeOKTest) {
  uint64_t base_addr = 0;
  uint64_t var_size = 0;
  int32_t ret = GetVarAddrAndSize("var", base_addr, var_size);
  EXPECT_EQ(ret, 0);
}
TEST_F(GePluginTest, GetVarAddrAndSizeFailedTest) {
  uint64_t base_addr = 0;
  uint64_t var_size = 0;
  int32_t ret = GetVarAddrAndSize("", base_addr, var_size);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, MallocSharedMemFailedTest) {
  ge::TensorInfo tensor_info;
  uint64_t dev_addr = 0;
  uint64_t memory_size = 0;
  int32_t ret = MallocSharedMem(tensor_info, dev_addr, memory_size);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, MallocSharedMemOKTest) {
  ge::TensorInfo tensor_info;
  tensor_info.var_name = "ge";
  uint64_t dev_addr = 0;
  uint64_t memory_size = 0;
  int32_t ret = MallocSharedMem(tensor_info, dev_addr, memory_size);
  EXPECT_EQ(ret, 0);
}
TEST_F(GePluginTest, NpuCloseTest) {
  std::map<std::string, std::string> init_options;
  init_options["ge.jobType"] = "1";
  init_options["ge.tuningPath"] = "./";
  PluginInit(init_options);
  NpuClose();
}
TEST_F(GePluginTest, RdmaInitAndRegisterFail1Test) {
  std::vector<ge::HostVarInfo> var_info;
  ge::HostVarInfo host_var_info;
  host_var_info.base_addr = 0;
  var_info.push_back(host_var_info);
  size_t size = 0;
  int32_t ret = RdmaInitAndRegister(var_info, size);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, RdmaInitAndRegisterFail2Test) {
  std::vector<ge::HostVarInfo> var_info;
  size_t size = 1;
  int32_t ret = RdmaInitAndRegister(var_info, size);
  EXPECT_EQ(ret, -1);
}
TEST_F(GePluginTest, RdmaInitAndRegisterOKTest) {
  std::vector<ge::HostVarInfo> var_info;
  ge::HostVarInfo host_var_info;
  host_var_info.base_addr = 0;
  var_info.push_back(host_var_info);
  size_t size = 1;
  int32_t ret = RdmaInitAndRegister(var_info, size);
  EXPECT_EQ(ret, 0);
}

}
} // end tensorflow