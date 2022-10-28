#include <memory>
#include "tf_adapter/kernels/aicpu/npu_cpu_ops.cc"
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
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    CacheAddOp cache(context);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST_F(NpuCpuOpTest, TestDecodeImageV3) {
    DataTypeSlice input_types({DT_STRING});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_UINT8});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelContext *ctx = nullptr;
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    DecodeImageV3Op decodeImageV3Op(context);
    decodeImageV3Op.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestInitPartitionMap) {
    DataTypeSlice input_types({DT_UINT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_UINT32});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    InitPartitionMapOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestInitEmbeddingHashmap) {
    DataTypeSlice input_types({DT_UINT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_UINT32});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    InitEmbeddingHashmapOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingTableFind) {
    DataTypeSlice input_types({DT_UINT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_FLOAT});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingTableFindOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingTableImport) {
    DataTypeSlice input_types({DT_UINT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_UINT32});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingTableImportOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestUninitPartitionMap) {
    DataTypeSlice input_types({DT_UINT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_UINT32});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    UninitPartitionMapOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestUninitEmbeddingHashmap) {
    DataTypeSlice input_types({DT_UINT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_UINT32});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    UninitEmbeddingHashmapOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}
}