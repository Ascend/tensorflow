#include "tf_adapter/kernels/npu_cpu_ops.cc"

namespace tensorflow {
TEST(NpuCpuOpTest, TestCacheAdd) {
    OpKernelConstruction *context;
    CacheAddOp cache(context);
    cache.Compute();
}
}