#if 1
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

TEST(HcclOpTest, TestShapeInference) {
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
  ASSERT_EQ("[3,4]", c.DebugString(c.output(0)));
}

}  // namespace
}  // namespace tensorflow



#else

#include "tf_adapter/kernels/geop_npu.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/version.h"
#include <stdlib.h>
#include "gtest/gtest.h"

namespace tensorflow {
namespace {
using geDataUniquePtr = std::unique_ptr<uint8_t[], std::function<void(uint8_t *)>>;
class NpuGetNextOutputInfo {
public:
  NpuGetNextOutputInfo(ge::Placement placement, std::vector<int64_t> dims,
     size_t output_size, geDataUniquePtr data)
    : placement_(placement), dims_(dims), output_size_(output_size), data_(std::move(data)) {}
  ~NpuGetNextOutputInfo() { LOG(INFO) << "[GEOP] Release NpuGetNextOutputInfo."; }
  ge::Placement placement_;
  std::vector<int64_t> dims_;
  size_t output_size_;
  geDataUniquePtr data_;
};

class NpuHostGetNextAllocator : public tensorflow::Allocator {
 public:
  static tensorflow::Allocator *Create(std::unique_ptr<NpuGetNextOutputInfo> output) {
    return new (std::nothrow) NpuHostGetNextAllocator(std::move(output));
  }
 private:
  explicit NpuHostGetNextAllocator(std::unique_ptr<NpuGetNextOutputInfo> output) : output_(std::move(output)) {
    LOG(INFO) << "[GEOP] getnext data addr:" << reinterpret_cast<uintptr_t>(output_->data_.get());
  }
  ~NpuHostGetNextAllocator() override {
    LOG(INFO) << "[GEOP] Release getnext data addr:" << reinterpret_cast<uintptr_t>(output_->data_.get());
  }
  std::string Name() override { return "NpuHostGetNextAllocator"; }
  void *AllocateRaw(size_t alignment, size_t num_bytes) override { return output_.get(); }
  void DeallocateRaw(void *ptr) override { delete this; }
  std::unique_ptr<NpuGetNextOutputInfo> output_;
};

class HcclOpTest : public testing::Test {
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

Status GeOpRunGraphAsync(std::string example_path, gtl::InlinedVector<TensorValue, 4> inputs,
                         NodeDef &geop_node_def, std::string node_name, bool only_run_once = true) {
  Env* env = Env::Default();
  GraphDef graph_def;
  std::string graph_def_path = example_path;
  ReadTextProto(env, graph_def_path, &graph_def);
  for (int i = 0; i < graph_def.node_size(); i++) {
    NodeDef *node_def = graph_def.mutable_node(i);
    if (node_def->name() == node_name) {
      geop_node_def = *node_def;
      OpKernelContext::Params params;
      params.record_tensor_accesses = false;
      auto device = absl::make_unique<DummyDevice>(env, params.record_tensor_accesses);
      params.device = device.get();
      Status status;
      std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(),
                                   *node_def, TF_GRAPH_DEF_VERSION, &status));
      EXPECT_TRUE(status.ok());
      auto ctx = absl::make_unique<OpKernelContext>(&params);
      op->Compute(ctx.get());
//       AsyncOpKernel* async_op = op->AsAsync();
//       params.op_kernel = async_op;
//       params.session_handle = "session_0";
//       params.inputs = &inputs;

//       //function library
//       FunctionDefLibrary func_def_lib = graph_def.library();
//       std::unique_ptr<FunctionLibraryDefinition> lib_def(
//         new FunctionLibraryDefinition(OpRegistry::Global(), func_def_lib));
//       OptimizerOptions opts;
//       std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr(
//         new ProcessFunctionLibraryRuntime(nullptr, Env::Default(), TF_GRAPH_DEF_VERSION,
//           lib_def.get(), opts, nullptr, nullptr));
//       FunctionLibraryRuntime* flr = proc_flr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
//       params.function_library = flr;
//       auto ctx = absl::make_unique<OpKernelContext>(&params);
//       AsyncOpKernel::DoneCallback done = []() { LOG(INFO) << "DONE DoneCallback"; };
//       async_op->ComputeAsync(ctx.get(), done);
//       if (!only_run_once) {
//         auto ctx1 = absl::make_unique<OpKernelContext>(&params);
//         async_op->ComputeAsync(ctx1.get(), done);
//       }
    }
  }
  return Status::OK();
}

TEST_F(HcclOpTest, GeOpDynamicDimsTest) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/hccl_op.pbtxt";
  Tensor a(DT_INT32, TensorShape({1,}));
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a)};
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "HcclOp1_0").ok());
}


}
} //end tensorflow
#endif
