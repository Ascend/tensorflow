#include <memory>
#include "tf_adapter/kernels/aicpu/npu_cpu_ops.cc"
#include "gtest/gtest.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

PartialTensorShape TShape(std::initializer_list<int64> dims) {
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


TEST(NonZeroWithValueOpsTest, TestNonZeroWithValueShape) {
    DataTypeSlice input_types({DT_INT32, DT_INT32, DT_INT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_INT32, DT_INT32});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    NonZeroWithValueShapeOp cache(context);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(NonZeroWithValueOpsTest, TestNonZeroWithValueShapeInfer) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("NonZeroWithValueShape", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)                  
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}
}  // namespace
}  // namespace tensorflow