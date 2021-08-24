#include <memory>
#include "tf_adapter/kernels/k_means_centroids_v2.cc"
#include "gtest/gtest.h"

namespace tensorflow {
class KMeansCentroidsV2OpTest : public testing::Test {
  protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(KMeansCentroidsV2OpTest, TestKMeansCentroidsV2) {
    DataTypeSlice input_types({DT_FLOAT, DT_FLOAT, DT_FLOAT});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_FLOAT, DT_FLOAT, DT_FLOAT});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    KMeansCentroidsV2Op k_means_centroids_v2(context);
    OpKernelContext *ctx = nullptr;
    k_means_centroids_v2.Compute(ctx);
    k_means_centroids_v2.IsExpensive();
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}
}