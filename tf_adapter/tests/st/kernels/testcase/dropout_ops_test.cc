#include <memory>
#include "gtest/gtest.h"
#include "tf_adapter/kernels/aicore/dropout_ops.cc"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/fake_input.h"

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

TEST(DropOutGenOrDoMaskOpTest, TestDropOutDoCompute) {
  DataTypeSlice input_types({DT_INT32, DT_INT32});
  MemoryTypeSlice input_memory_types;
  DataTypeSlice output_types({DT_INT32});
  MemoryTypeSlice output_memory_types;
  DeviceBase *device = new DeviceBase(Env::Default());
  NodeDef *node_def = new NodeDef();
  OpDef *op_def = new OpDef();
  OpKernelConstruction *context =
      new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr, input_types, input_memory_types,
                               output_types, output_memory_types, 1, nullptr);
  DropOutDoMaskOp *dropOutDomaskOp = new DropOutDoMaskOp(context);
  OpKernelContext *ctx = nullptr;
  dropOutDomaskOp->Compute(ctx);
  dropOutDomaskOp->IsExpensive();
  delete device;
  delete node_def;
  delete op_def;
  delete context;
  delete dropOutDomaskOp;
}

TEST(DropOutGenOrDoMaskOpTest, TestDropOutGenCompute) {
  DataTypeSlice input_types({DT_INT32, DT_INT32});
  MemoryTypeSlice input_memory_types;
  DataTypeSlice output_types({DT_INT32});
  MemoryTypeSlice output_memory_types;
  DeviceBase *device = new DeviceBase(Env::Default());
  NodeDef *node_def = new NodeDef();
  OpDef *op_def = new OpDef();
  OpKernelConstruction *context =
      new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr, input_types, input_memory_types,
                               output_types, output_memory_types, 1, nullptr);
  DropOutGenMaskOp *dropOutGenmaskOp = new DropOutGenMaskOp(context);
  OpKernelContext *ctx = nullptr;
  dropOutGenmaskOp->Compute(ctx);
  dropOutGenmaskOp->IsExpensive();
  delete device;
  delete node_def;
  delete op_def;
  delete context;
  delete dropOutGenmaskOp;
}

TEST(DropOutGenOrDoMaskOpTest, TestDropOutGenMaskInference) {
  const OpRegistrationData *reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("DropOutGenMask", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_INT64)
                  .Attr("S", DT_FLOAT)
                  .Input(FakeInputStub(DT_INT64))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(
      0, &def, op_def,
      {TShape({16}), TShape({})},
      {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(DropOutGenOrDoMaskOpTest, TestDropOutGenMaskV3Inference) {
  const OpRegistrationData *reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("DropOutGenMaskV3", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_INT64)
                  .Attr("S", DT_FLOAT)
                  .Input(FakeInputStub(DT_INT64))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(
      0, &def, op_def,
      {TShape({16}), TShape({})},
      {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}
}  // namespace
}  // namespace tensorflow