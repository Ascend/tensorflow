#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/version.h"
#include <stdlib.h>
#include "gtest/gtest.h"

namespace tensorflow {
namespace {

class NpuOpsTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

class DummyDevice : public DeviceBase {
 public:
  DummyDevice(Env* env, bool save) : DeviceBase(env), save_(save) {}
  bool RequiresRecordingAccessedTensors() const override { return save_; }
  Allocator* GetAllocator(AllocatorAttributes) override { return cpu_allocator(); }

 private:
  bool save_;
};

PartialTensorShape S(std::initializer_list<int64> dims) { return PartialTensorShape(dims); }

FakeInputFunctor FakeInputStub(DataType dt) {
  return [dt](const OpDef& op_def, int in_index, const NodeDef& node_def, NodeDefBuilder* builder) {
    char c = 'a' + (in_index % 26);
    string in_node = string(&c, 1);
    builder->Input(in_node, 0, dt);
    return Status::OK();
  };
}

FakeInputFunctor FakeInputStub(int n, DataType dt) {
  return [n, dt](const OpDef& op_def, int in_index, const NodeDef& node_def, NodeDefBuilder* builder) {
    std::vector<NodeDefBuilder::NodeOut> srcs;
    srcs.reserve(n);
    char c = 'a' + (in_index % 26);
    string in_node = string(&c, 1);
    for (int i = 0; i < n; ++i) {
      srcs.emplace_back(in_node, i, dt);
    }
    builder->Input(gtl::ArraySlice<NodeDefBuilder::NodeOut>(srcs));
    return Status::OK();
  };
}

Status NpuOpCompute(std::string graph_def_path, NodeDef& npu_node_def, std::string node_name) {
  Env* env = Env::Default();
  GraphDef graph_def;
  ReadTextProto(env, graph_def_path, &graph_def);
  for (int i = 0; i < graph_def.node_size(); i++) {
    NodeDef* node_def = graph_def.mutable_node(i);
    if (node_def->name() == node_name) {
      npu_node_def = *node_def;
      OpKernelContext::Params params;
      params.record_tensor_accesses = false;
      auto device = absl::make_unique<DummyDevice>(env, params.record_tensor_accesses);
      params.device = device.get();
      Status status;
      std::unique_ptr<OpKernel> op(
        CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(), *node_def, TF_GRAPH_DEF_VERSION, &status));
      EXPECT_TRUE(status.ok());
      params.op_kernel = op.get();
      auto ctx = absl::make_unique<OpKernelContext>(&params);
      op->Compute(ctx.get());
    }
  }
  return Status::OK();
}

TEST(NpuOpsTest, TestInitShutdown) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/init_shutdown.pbtxt";
  EXPECT_TRUE(NpuOpCompute(graph_def_path, node_def, "NPUInit").ok());
  EXPECT_TRUE(NpuOpCompute(graph_def_path, node_def, "NPUShutdown").ok());
}
TEST(NpuOpsTest, TestLARSShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("LARS", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("LARS", &op_def)
                .Input(FakeInputStub(2, DT_FLOAT))
                .Input(FakeInputStub(2, DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Attr("T", {DT_FLOAT, DT_FLOAT})
                .Attr("hyperpara", 0.001)
                .Attr("epsilon", 0.001)
                .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({1, 2}), S({1, 2}), S({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[1,2]", c.DebugString(c.output(0)));
}
TEST(NpuOpsTest, TestLarsV2ShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("LarsV2", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("LarsV2", &op_def)
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Attr("hyperpara", 0.001)
                .Attr("epsilon", 0.001)
                .Attr("use_clip", false)
                .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({1, 2}), S({1, 2}), S({}), S({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[1,2]", c.DebugString(c.output(0)));
}
TEST(NpuOpsTest, TestDropOutDoMaskShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("DropOutDoMask", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("DropOutDoMask", &op_def)
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_UINT8))
                .Input(FakeInputStub(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({1, 2}), S({}), S({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[1,2]", c.DebugString(c.output(0)));
}
TEST(NpuOpsTest, TestDropOutGenMaskShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("DropOutGenMask", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("DropOutDoMask", &op_def)
                .Input(FakeInputStub(DT_INT64))
                .Attr("T", DT_INT64)
                .Input(FakeInputStub(DT_FLOAT))
                .Attr("S", DT_FLOAT)
                .Attr("seed", 0)
                .Attr("seed2", 0)
                .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
                                      {S({1,}), S({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[?]", c.DebugString(c.output(0)));
  shape_inference::InferenceContext c1(0, &def, op_def,
                                       {S({1,}), S({})}, {}, {S({1,}), S({})}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c1));
  ASSERT_EQ("[16]", c1.DebugString(c1.output(0)));
}
TEST(NpuOpsTest, TestDropOutGenMaskV3ShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("DropOutGenMaskV3", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("DropOutGenMaskV3", &op_def)
                .Input(FakeInputStub(DT_INT64))
                .Attr("T", DT_INT64)
                .Input(FakeInputStub(DT_FLOAT))
                .Attr("S", DT_FLOAT)
                .Attr("seed", 0)
                .Attr("seed2", 0)
                .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({1}), S({})}, {}, {S({1}), S({})}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[16]", c.DebugString(c.output(0)));
  shape_inference::InferenceContext c1(0, &def, op_def, {S({1}), S({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c1));
  ASSERT_EQ("?", c1.DebugString(c1.output(0)));
}
TEST(NpuOpsTest, TestBasicLSTMCellShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("BasicLSTMCell", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("BasicLSTMCell", &op_def)
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Attr("keep_prob", 1.0)
                .Attr("forget_bias", 1.0)
                .Attr("state_is_tuple", true)
                .Attr("activation", "tanh")
                .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({1, 2}), S({1, 2}), S({2, 2}), S({1, 2}), S({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[2,2]", c.DebugString(c.output(0)));
}
TEST(NpuOpsTest, TestBasicLSTMCellCStateGradShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("BasicLSTMCellCStateGrad", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("BasicLSTMCellCStateGrad", &op_def)
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Attr("forget_bias", 1.0)
                .Attr("activation", "tanh")
                .Finalize(&def));
  shape_inference::InferenceContext c(
    0, &def, op_def, {S({1, 2}), S({1, 2}), S({3, 2}), S({1, 2}), S({2, 3}), S({}), S({}), S({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[2,12]", c.DebugString(c.output(0)));
  ASSERT_EQ("[3,2]", c.DebugString(c.output(1)));
}
TEST(NpuOpsTest, TestBasicLSTMCellWeightGradShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("BasicLSTMCellWeightGrad", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("BasicLSTMCellWeightGrad", &op_def)
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({1, 2}), S({1, 2}), S({2, 2})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[4,2]", c.DebugString(c.output(0)));
}
TEST(NpuOpsTest, TestBasicLSTMCellInputGradShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("BasicLSTMCellInputGrad", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("BasicLSTMCellInputGrad", &op_def)
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Attr("keep_prob", 1.0)
                .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {S({1, 4}), S({1, 2})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[1,0]", c.DebugString(c.output(0)));
}
TEST(NpuOpsTest, TestLambApplyOptimizerAssignShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("LambApplyOptimizerAssign", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("LambApplyOptimizerAssign", &op_def)
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
                                      {S({1,}), S({2,}), S({3,}), S({4,}), S({5,}), S({6,}),
                                       S({7,}), S({8,}), S({9,}), S({10,}),S({11,}),S({12,})},
                                      {}, {}, {});

  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[1]", c.DebugString(c.output(0)));
}
TEST(NpuOpsTest, TestLambApplyWeightAssignShapeFn) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("LambApplyWeightAssign", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("LambApplyWeightAssign", &op_def)
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Input(FakeInputStub(DT_FLOAT))
                .Finalize(&def));
  shape_inference::InferenceContext c(
    0, &def, op_def, {S({1,}), S({1,}), S({1,}),S({1,}),S({224,224})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[224,224]", c.DebugString(c.output(0)));
}

TEST(NpuOpsTest, TestDecodeImageV3ShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("DecodeImageV3", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dump", &op_def)
                  .Attr("channels", 3)
                  .Attr("dtype", DT_UINT8)
                  .Attr("expand_animations", true)
                  .Input(FakeInputStub(DT_STRING))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {S({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(NpuOpsTest, TestGetNextShapeInference) {
  std::initializer_list<int64> dims = {};
  TensorShapeProto shape_proto;
  TensorShape(dims).AsProto(&shape_proto);
  std::string channel_name = "channel";
  tensorflow::AttrValue output_shapes;
  tensorflow::AttrValue output_types;
  *(output_shapes.mutable_list()->add_shape()) = shape_proto;
  *(output_shapes.mutable_list()->add_shape()) = shape_proto;
  output_types.mutable_list()->add_type(DT_STRING);
  output_types.mutable_list()->add_type(DT_INT32);
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("GetNext", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("GetNext", &op_def)
                   .Attr("channel_name", channel_name)
                   .Attr("output_types", output_types)
                   .Attr("output_shapes", output_shapes)
                   .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {S({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}
}  // namespace
}  // namespace tensorflow