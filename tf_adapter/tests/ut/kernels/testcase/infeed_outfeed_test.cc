#include "tf_adapter/util/npu_attrs.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/version.h"
#include <stdlib.h>
#include "gtest/gtest.h"


namespace tensorflow {
namespace {

#define TF_ASSERT_OK(statement) \
  ASSERT_EQ(::tensorflow::Status::OK(), (statement))

#define TF_EXPECT_OK(statement) \
  EXPECT_EQ(::tensorflow::Status::OK(), (statement))

class DummyDevice : public DeviceBase {
 public:
  DummyDevice(Env* env, bool save) : DeviceBase(env), save_(save) {}
  bool RequiresRecordingAccessedTensors() const override { return save_; }
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override { return cpu_allocator(); }
 private:
  bool save_;
};
}
class InfeedOutfeedTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(InfeedOutfeedTest, LogSummaryTest)  {
  NpuAttrs::SetNewDataTransferFlag(true);
  std::initializer_list<int64> dims = {};
  TensorShapeProto shape_proto;
  TensorShape(dims).AsProto(&shape_proto);

  std::string channel_name = "_npu_log";

  NodeDef outfeed_node;
  tensorflow::AttrValue output_shapes;
  tensorflow::AttrValue output_types;
  *(output_shapes.mutable_list()->add_shape()) = shape_proto;
  *(output_shapes.mutable_list()->add_shape()) = shape_proto;
  output_types.mutable_list()->add_type(DT_STRING);
  output_types.mutable_list()->add_type(DT_INT32);
  TF_ASSERT_OK(NodeDefBuilder("out_feed", "OutfeedDequeueOp")
                   .Attr("channel_name", channel_name)
                   .Attr("output_types", output_types)
                   .Attr("output_shapes", output_shapes)
                   .Finalize(&outfeed_node));

  DeviceType device_type = DEVICE_CPU;
  Env* env = Env::Default();
  auto device = absl::make_unique<DummyDevice>(env, false);

  Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, device.get(),
                                              cpu_allocator(), outfeed_node,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_ASSERT_OK(status);

  OpKernelContext::Params params;
  params.device = device.get();
  params.op_kernel = op.get();
  std::unique_ptr<CancellationManager> cancellation_manager = absl::make_unique<CancellationManager>();
  params.cancellation_manager = cancellation_manager.get();

  OpKernelContext ctx(&params);
  op->Compute(&ctx);
  TF_EXPECT_OK(ctx.status());
  NpuAttrs::SetNewDataTransferFlag(false);
}
} //end tensorflow