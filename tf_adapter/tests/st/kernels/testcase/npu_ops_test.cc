#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/version.h"
#include <stdlib.h>
#include "gtest/gtest.h"

namespace tensorflow {
namespace {

class NpuOpsTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

class DummyDevice : public DeviceBase {
 public:
  DummyDevice(Env* env, bool save) : DeviceBase(env), save_(save) {}
  bool RequiresRecordingAccessedTensors() const override { return save_; }
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override { return cpu_allocator(); }
 private:
  bool save_;
};

Status NpuOpCompute(std::string graph_def_path, NodeDef &npu_node_def, std::string node_name) {
  Env* env = Env::Default();
  GraphDef graph_def;
  ReadTextProto(env, graph_def_path, &graph_def);
  for (int i = 0; i < graph_def.node_size(); i++) {
    NodeDef *node_def = graph_def.mutable_node(i);
    if (node_def->name() == node_name) {
      npu_node_def = *node_def;
      OpKernelContext::Params params;
      params.record_tensor_accesses = false;
      auto device = absl::make_unique<DummyDevice>(env, params.record_tensor_accesses);
      params.device = device.get();
      Status status;
      std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
                                                  *node_def, TF_GRAPH_DEF_VERSION, &status));
      EXPECT_TRUE(status.ok());
      params.op_kernel = op.get();
      auto ctx = absl::make_unique<OpKernelContext>(&params);
      op->Compute(ctx.get());
    }
  }
  return Status::OK();
}

TEST(NpuOpsTest, TestInitShutdown) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/init_shutdown.pbtxt";
  EXPECT_TRUE(NpuOpCompute(graph_def_path, node_def, "NPUInit").ok());
  EXPECT_TRUE(NpuOpCompute(graph_def_path, node_def, "NPUShutdown").ok());
}

}  // namespace
}  // namespace tensorflow