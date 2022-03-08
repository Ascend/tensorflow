#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"
#include "tf_adapter/kernels/aicore/prod_env_mat_a.cc"
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

TEST(ProdEnvMatAOpTest, TestProdEnvMatA) {
  DataTypeSlice input_types({DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_FLOAT,DT_FLOAT});
  MemoryTypeSlice input_memory_types;
  DataTypeSlice output_types({DT_FLOAT, DT_FLOAT,DT_FLOAT,DT_INT32});
  MemoryTypeSlice output_memory_types;
  DeviceBase *device = new DeviceBase(Env::Default());
  NodeDef *node_def = new NodeDef();
  OpDef *op_def = new OpDef();
  OpKernelConstruction *context = new OpKernelConstruction(
      DEVICE_CPU, device, nullptr, node_def, op_def, nullptr, input_types,
      input_memory_types, output_types, output_memory_types, 1, nullptr);
  ProdEnvMatAOP<int> prod_env_mat_a(context);
  OpKernelContext *ctx = nullptr;
  prod_env_mat_a.Compute(ctx);
  prod_env_mat_a.IsExpensive();
  delete device;
  delete node_def;
  delete op_def;
  delete context;
}

TEST(ProdEnvMatAOpTest, TestProdEnvMatAShapeInference) {
  const OpRegistrationData *reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("ProdEnvMatA", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_FLOAT)
                  .Attr("rcut_a", 0.0)
                  .Attr("rcut_r", 9.0)
                  .Attr("rcut_r_smth", 2.0)
                  .Attr("sel_a", {600,600,600})
                  .Attr("sel_r", {0,0})
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {TShape({1, 27957}), TShape({1, 9319}),TShape({1, 1,1}),
                                      TShape({1,57000}),TShape({2101249}), TShape({3,7200}), TShape({3,7200})}, {}, {}, {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[1,88473600]", c.DebugString(c.output(0)));
  ASSERT_EQ("[1,265420800]", c.DebugString(c.output(1)));
  ASSERT_EQ("[1,66355200]", c.DebugString(c.output(2)));
  ASSERT_EQ("[1,22118400]", c.DebugString(c.output(3)));
}
} // namespace
} // namespace tensorflow