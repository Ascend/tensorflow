#include <memory>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tf_adapter/kernels/aicpu/npu_cpu_ops.cc"
#include "gtest/gtest.h"

namespace tensorflow {
namespace {
PartialTensorShape TShape(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}

FakeInputFunctor FakeInputStub(DataType dt) {
  return [dt](const OpDef &op_def, int in_index, const NodeDef &node_def,
              NodeDefBuilder *builder) {
    char c = 'a' + (in_index % 26);
    string in_node = string(&c, 1);
    builder->Input(in_node, 0, dt);
    return Status::OK();
  };
}

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
    DataTypeSlice input_types({DT_INT32});
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
    DataTypeSlice input_types({DT_INT32});
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

TEST(EmbeddingOpsTest, TestEmbeddingTableFind01) {
    DataTypeSlice input_types({DT_INT32});
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

TEST(EmbeddingOpsTest, TestEmbeddingTableFind02) {
    const OpRegistrationData *reg;
    TF_CHECK_OK(OpRegistry::Global()->LookUp("EmbeddingTableFind", &reg));
    OpDef op_def = reg->op_def;
    NodeDef def;
    TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                    .Attr("embedding_dim", 4)
                    .Input(FakeInputStub(DT_INT32))
                    .Input(FakeInputStub(DT_INT64))
                    .Finalize(&def));

    shape_inference::InferenceContext c(
        0, &def, op_def,
        {TShape({1}), TShape({16})},
        {}, {}, {});
    TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(EmbeddingOpsTest, TestEmbeddingTableImport) {
    DataTypeSlice input_types({DT_INT32});
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
    DataTypeSlice input_types({DT_INT32});
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
    DataTypeSlice input_types({DT_INT32});
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
TEST(EmbeddingOpsTest, TestTableToResource) {
    DataTypeSlice input_types({DT_INT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_RESOURCE});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    TableToResourceOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingTableFindAndInit) {
    DataTypeSlice input_types({DT_INT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_INT32});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingTableFindAndInitOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingComputeVarExport) {
    DataTypeSlice input_types({DT_STRING});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_STRING});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingComputeVarExportOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingComputeVarImport) {
    DataTypeSlice input_types({DT_STRING});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_STRING});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingComputeVarImportOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingTableExport) {
    DataTypeSlice input_types({DT_STRING});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_STRING});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingTableExportOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingApplyAdam) {
    DataTypeSlice input_types({DT_RESOURCE});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_RESOURCE});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingApplyAdamOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingApplyAdamW) {
    DataTypeSlice input_types({DT_RESOURCE});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_RESOURCE});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingApplyAdamWOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingApplyAdaGrad) {
    DataTypeSlice input_types({DT_RESOURCE});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_RESOURCE});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingApplyAdaGradOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingApplySgd) {
    DataTypeSlice input_types({DT_RESOURCE});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_RESOURCE});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingApplySgdOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingApplyRmspropOp) {
    DataTypeSlice input_types({DT_RESOURCE});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_RESOURCE});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingApplyRmspropOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestExponentialDecayLROp) {
    DataTypeSlice input_types({DT_RESOURCE});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_RESOURCE});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    ExponentialDecayLROp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingFeatureMapping) {
    DataTypeSlice input_types({DT_INT64});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_INT32});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    EmbeddingFeatureMappingOp cache(context);
    OpKernelContext *ctx = nullptr;
    cache.Compute(ctx);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(EmbeddingOpsTest, TestEmbeddingFeatureMappingShapeInfer) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("EmbeddingFeatureMapping", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Input(FakeInputStub(DT_INT64))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({2, 2, 3, 4})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}
}
}