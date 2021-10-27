#include <memory>
#include "tf_adapter/kernels/non_zero_with_value_ops.cc"
#include "gtest/gtest.h"
namespace tensorflow {
class NonZeroWithValueOpTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(NonZeroWithValueOpTest, TestNonZeroWithValue) {
    DataTypeSlice input_types({DT_INT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_INT32, DT_INT64, DT_INT64});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    NonZeroWithValueOP<int> non_zero_with_value(context);
    OpKernelContext *ctx = nullptr;
    non_zero_with_value.Compute(ctx);
    non_zero_with_value.IsExpensive();
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}
}  // namespace tensorflow