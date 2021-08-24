#include <memory>
#include "tf_adapter/kernels/k_means_centroids.cc"
#include "gtest/gtest.h"

namespace tensorflow {
class KMeansCentroidsOpTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(KMeansCentroidsOpTest, TestKMeansCentroids) {
    DataTypeSlice input_types({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_FLOAT, DT_FLOAT, DT_FLOAT});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    KMeansCentroidsOp k_means_centroids(context);
    OpKernelContext *ctx = nullptr;
    k_means_centroids.Compute(ctx);
    k_means_centroids.IsExpensive();
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}
}