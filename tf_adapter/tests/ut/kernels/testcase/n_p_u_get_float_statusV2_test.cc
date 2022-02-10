#include <memory>
#include "tf_adapter/kernels/aicore/npu_mixed_precesion_ops.cc"
#include "gtest/gtest.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"
#include <iostream>
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

TEST(NPUGetFloatStatusV2OpTest, TestNPUGetFloatStatusV2) {
  LOG(INFO) << "get into Test ---liyefeng---";
  DataTypeSlice input_types({DT_FLOAT});
  MemoryTypeSlice input_memory_types;
  DataTypeSlice output_types({DT_FLOAT});
  MemoryTypeSlice output_memory_types;
  DeviceBase *device = new DeviceBase(Env::Default());
  NodeDef *node_def = new NodeDef();
  OpDef *op_def = new OpDef();
  LOG(INFO) << "before opkernelconstruction ---liyefeng---";
  OpKernelConstruction *context = new OpKernelConstruction(
      DEVICE_CPU, device, nullptr, node_def, op_def, nullptr, input_types,
      input_memory_types, output_types, output_memory_types, 1, nullptr);
  LOG(INFO) << "Before new a npugetfloatstatusv2() ---liyefeng---";
  NpuGetFloatStatusV2Op npugetfloatstatusv2(context);
  OpKernelContext *ctx = nullptr;
  LOG(INFO) << "before compute---liyefeng---";
  npugetfloatstatusv2.Compute(ctx);
  delete device;
  delete node_def;
  delete op_def;
  delete context;
}

TEST(NPUClearFloatStatusV2OpTest, TestNPUClearFloatStatusV2) {
  DataTypeSlice input_types({});
  MemoryTypeSlice input_memory_types;
  DataTypeSlice output_types({});
  MemoryTypeSlice output_memory_types;
  DeviceBase *device = new DeviceBase(Env::Default());
  NodeDef *node_def = new NodeDef();
  OpDef *op_def = new OpDef();
  LOG(INFO) << "before opkernelconstruction ---liyefeng---";
  OpKernelConstruction *context = new OpKernelConstruction(
      DEVICE_CPU, device, nullptr, node_def, op_def, nullptr, input_types,
      input_memory_types, output_types, output_memory_types, 1, nullptr);
  LOG(INFO) << "Before new a npugetfloatstatusv2() ---liyefeng---";
  NpuClearFloatStatusV2Op npuclearfloatstatusv2(context);
  OpKernelContext *ctx = nullptr;
  LOG(INFO) << "before compute---liyefeng---";
  npuclearfloatstatusv2.Compute(ctx);
  delete device;
  delete node_def;
  delete op_def;
  delete context;
}

} // namespace
} // namespace tensorflow
