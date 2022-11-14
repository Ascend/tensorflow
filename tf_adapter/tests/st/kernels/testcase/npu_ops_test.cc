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

PartialTensorShape TShape(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}

class NpuOpsTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

class DummyDevice : public DeviceBase {
 public:
  DummyDevice(Env* env, bool save) : DeviceBase(env), save_(save) {}
  bool RequiresRecordingAccessedTensors() const override { return save_; }
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override { return cpu_allocator(); }
 private:
  bool save_;
};

Status NpuOpCompute(std::string graph_def_path, NodeDef &npu_node_def, std::string node_name) {
  Env* env = Env::Default();
  GraphDef graph_def;
  ReadTextProto(env, graph_def_path, &graph_def);
  for (int i = 0; i < graph_def.node_size(); i++) {
    NodeDef *node_def = graph_def.mutable_node(i);
    if (node_def->name() == node_name) {
      npu_node_def = *node_def;
      OpKernelContext::Params params;
      params.record_tensor_accesses = false;
      auto device = absl::make_unique<DummyDevice>(env, params.record_tensor_accesses);
      params.device = device.get();
      Status status;
      std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
                                                  *node_def, TF_GRAPH_DEF_VERSION, &status));
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
    {TShape({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

}  // namespace
}  // namespace tensorflow