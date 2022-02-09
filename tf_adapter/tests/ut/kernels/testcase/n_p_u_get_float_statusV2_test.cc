#include <memory>
#include "tf_adapter/kernels/aicore/npu_mixed_precesion_ops.cc"
#include "gtest/gtest.h"

namespace tensorflow {
class NPUGetFloatStatusV2OpTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(NPUGetFloatStatusV2OpTest, TestNPUGetFloatStatusV2) {
    DataTypeSlice input_types({DT_FLOAT});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_FLOAT});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    std::cout << "liyefeng========before";
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    std::cout << "liyefeng========before2";
    NpuGetFloatStatusV2Op npugetfloatstatusv2(context);
    std::cout << "liyefeng========after";
    OpKernelContext *ctx = nullptr;
    std::cout << "liyefeng========before compute";
    npugetfloatstatusv2.Compute(ctx);
    std::cout << "liyefeng========after compute";
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}
}