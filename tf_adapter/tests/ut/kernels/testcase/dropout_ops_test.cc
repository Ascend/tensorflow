#include <memory>
#include "gtest/gtest.h"
#include "tf_adapter/kernels/aicore/dropout_ops.cc"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"

namespace tensorflow {
namespace {

class DropOutGenOrDoMaskOpTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

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

}  // namespace
}  // namespace tensorflow