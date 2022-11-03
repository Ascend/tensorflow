#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"
#include "tf_adapter/kernels/hccl/hccl_ops.cc"

namespace tensorflow {
namespace {

PartialTensorShape S(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}

FakeInputFunctor FakeInputStub(DataType dt) {
  return [dt](const OpDef& op_def, int in_index, const NodeDef& node_def,
              NodeDefBuilder* builder) {
    char c = 'a' + (in_index % 26);
    string in_node =  string(&c, 1);
    builder->Input(in_node, 0, dt);
    return Status::OK();
  };
}

TEST(HcclOpTest, TestHcomAllGatherShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("HcomAllGather", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_FLOAT)
                  .Attr("rank_size", 8)
                  .Attr("group", "hccl_world_group")
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({3, 4})}, {}, {}, {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(c.input("input", &input_shapes));
  ASSERT_EQ("[3,4]", c.DebugString(input_shapes[0]));
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[24,4]", c.DebugString(c.output(0)));
}

TEST(HcclOpTest, TestHcomAllGatherShapeInferenceInvaildRankSize) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("HcomAllGather", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_FLOAT)
                  .Attr("rank_size", 0)
                  .Attr("group", "hccl_world_group")
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({3, 4})}, {}, {}, {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(c.input("input", &input_shapes));
  ASSERT_EQ("[3,4]", c.DebugString(input_shapes[0]));
  Status status = reg->shape_inference_fn(&c);
  EXPECT_TRUE(!status.ok());
}

TEST(HcclOpTest, TestHcomReduceScatterShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("HcomReduceScatter", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_FLOAT)
                  .Attr("rank_size", 8)
                  .Attr("group", "hccl_world_group")
                  .Attr("reduction", "sum")
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({24, 4})}, {}, {}, {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(c.input("input", &input_shapes));
  ASSERT_EQ("[24,4]", c.DebugString(input_shapes[0]));
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[3,4]", c.DebugString(c.output(0)));
}

TEST(HcclOpTest, TestHcomReduceScatterShapeInferenceInvaildRankSize) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("HcomReduceScatter", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_FLOAT)
                  .Attr("rank_size", 0)
                  .Attr("group", "hccl_world_group")
                  .Attr("reduction", "sum")
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({24, 4})}, {}, {}, {});
  std::vector<shape_inference::ShapeHandle> input_shapes;
  TF_CHECK_OK(c.input("input", &input_shapes));
  ASSERT_EQ("[24,4]", c.DebugString(input_shapes[0]));
  Status status = reg->shape_inference_fn(&c);
  EXPECT_TRUE(!status.ok());
}

TEST(HcclOpTest, TestHcomAllToAllVCShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("HcomAllToAllVC", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_INT64)
                  .Attr("rank", 0)
                  .Attr("group", "hccl_world_group")
                  .Input(FakeInputStub(DT_INT64))
                  .Input(FakeInputStub(DT_INT64))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({3, 4}),S({4, 4})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(HcclOpTest, TestHcomAllToAllVCOpKernel) {
  DataTypeSlice input_types({DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,DT_FLOAT,DT_FLOAT});
  MemoryTypeSlice input_memory_types;
  DataTypeSlice output_types({DT_FLOAT, DT_FLOAT});
  MemoryTypeSlice output_memory_types;
  DeviceBase *device = new DeviceBase(Env::Default());
  NodeDef *node_def = new NodeDef();
  OpDef *op_def = new OpDef();
  OpKernelConstruction *context = new OpKernelConstruction(
      DEVICE_CPU, device, nullptr, node_def, op_def, nullptr, input_types,
      input_memory_types, output_types, output_memory_types, 1, nullptr);
  OpKernelContext *context2;

  HcomAllToAllVCOpKernel* hcomAllToAllVCOpKernel = new HcomAllToAllVCOpKernel(context);
  hcomAllToAllVCOpKernel->Compute(context2);
  delete context;
  delete hcomAllToAllVCOpKernel;
}
}  // namespace
}  // namespace tensorflow