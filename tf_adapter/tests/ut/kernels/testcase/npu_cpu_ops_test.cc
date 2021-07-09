#include "tf_adapter/kernels/npu_cpu_ops.cc"
#include "gtest/gtest.h"

namespace tensorflow {
class NpuCpuOpTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(NpuCpuOpTest, TestCacheAdd) {
    OpKernelConstruction *context;
    OpKernelContext *compute_context
    CacheAddOp cache(context);
    cache.Compute(compute_context);
}
}