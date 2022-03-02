#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"
#include "tf_adapter/kernels/npu_ops.cc"
#include "gtest/gtest.h"
#include <memory>
namespace tensorflow {
namespace {
TEST(FileConstantTest, TestFileConstant) {
  DataTypeSlice input_types({DT_FLOAT});
  MemoryTypeSlice input_memory_types;
  DataTypeSlice output_types({DT_FLOAT});
  MemoryTypeSlice output_memory_types;
  DeviceBase *device = new DeviceBase(Env::Default());
  NodeDef *node_def = new NodeDef();
  OpDef *op_def = new OpDef();
  OpKernelConstruction *context = new OpKernelConstruction(
      DEVICE_CPU, device, nullptr, node_def, op_def, nullptr, input_types,
      input_memory_types, output_types, output_memory_types, 1, nullptr);
  FileConstant file_constant(context);
  OpKernelContext *ctx = nullptr;
  file_constant.Compute(ctx);
  file_constant.IsExpensive();
  delete device;
  delete node_def;
  delete op_def;
  delete context;
}

TEST(FileConstantTest, FileConstantShapeInference) {
  const OpRegistrationData *reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("FileConstant", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("file_id", "test")
                  .Attr("shape", {3,2})
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(&def));
  const std::vector<shape_inference::ShapeHandle> input_shapes = {};
  shape_inference::InferenceContext c(0, &def, op_def, input_shapes,
                                      {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[3,2]", c.DebugString(c.output(0)));
}
} // namespace
} // namespace tensorflow
