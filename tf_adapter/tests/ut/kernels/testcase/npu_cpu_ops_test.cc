#include "tf_adapter/kernels/npu_cpu_ops.cc"
#include "gtest/gtest.h"

namespace tensorflow {
class NpuCpuOpTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(NpuCpuOpTest, TestCacheAdd) {
    DataTypeSlice input_types({DT_RESOURCE, DT_INT64});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_INT64, DT_INT64, DT_INT64, DT_INT64});
    MemoryTypeSlice output_memory_types;
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    CacheAddOp cache(context);
}
}