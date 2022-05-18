#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tf_adapter/kernels/lru_cache_v2_ops.cc"
#include "gtest/gtest.h"
#include <memory>

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

TEST(LruCacheV2Test, TestLruCacheV2) {
  DataTypeSlice input_types({DT_INT32, DT_FLOAT_REF, DT_FLOAT_REF, DT_INT32_REF, DT_INT32});
  MemoryTypeSlice input_memory_types;
  DataTypeSlice output_types(
      {DT_FLOAT_REF, DT_FLOAT_REF, DT_FLOAT_REF, DT_INT32, DT_INT32, DT_INT32});
  MemoryTypeSlice output_memory_types;
  DeviceBase *device = new DeviceBase(Env::Default());
  NodeDef *node_def = new NodeDef();
  OpDef *op_def = new OpDef();
  OpKernelConstruction *context = new OpKernelConstruction(
      DEVICE_CPU, device, nullptr, node_def, op_def, nullptr, input_types,
      input_memory_types, output_types, output_memory_types, 1, nullptr);
  LRUCacheV2Op lrucachev2op(context);
  OpKernelContext *ctx = nullptr;
  lrucachev2op.Compute(ctx);
  lrucachev2op.IsExpensive();
  
  delete device;
  delete node_def;
  delete op_def;
  delete context;

  const OpRegistrationData *reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("LruCacheV2", &reg));
  OpDef op_def1 = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def1)
                  .Attr("T", DT_INT32)
                  .Attr("dtype", DT_FLOAT)
                  .Attr("pre_route_count", DT_INT32)
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT_REF))
                  .Input(FakeInputStub(DT_FLOAT_REF))
                  .Input(FakeInputStub(DT_INT32_REF))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(
      0, &def, op_def1,
      {TShape({1})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));

}
}
} // namespace tensorflow