#include "securec.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/version.h"
#include "register/register_types.h"
#include <stdlib.h>
#include "gtest/gtest.h"
#include "ge_stub.h"

#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/npu_plugin.h"
#include "tf_adapter/util/util.h"
#define private public
#include "tf_adapter/kernels/geop_npu.h"
#undef private

namespace tensorflow {
namespace {
using geDataUniquePtr = std::unique_ptr<uint8_t[], std::function<void(uint8_t*)>>;
class NpuGetNextOutputInfo {
 public:
  NpuGetNextOutputInfo(ge::Placement placement, std::vector<int64_t> dims, size_t output_size, geDataUniquePtr data)
      : placement_(placement), dims_(dims), output_size_(output_size), data_(std::move(data)) {}
  ~NpuGetNextOutputInfo() { LOG(INFO) << "[GEOP] Release NpuGetNextOutputInfo."; }
  ge::Placement placement_;
  std::vector<int64_t> dims_;
  size_t output_size_;
  geDataUniquePtr data_;
};

class NpuHostGetNextAllocator : public tensorflow::Allocator {
 public:
  static tensorflow::Allocator* Create(std::unique_ptr<NpuGetNextOutputInfo> output) {
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
  void* AllocateRaw(size_t alignment, size_t num_bytes) override { return output_.get(); }
  void DeallocateRaw(void* ptr) override { delete this; }
  std::unique_ptr<NpuGetNextOutputInfo> output_;
};

class GeOpTest : public testing::Test {
 protected:
  virtual void SetUp() {
    *const_cast<bool *>(&kDumpGraph) = true;
    NpuAttrs::SetNewDataTransferFlag(true);
  }
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

Status GeOpRunGraphAsync(std::string example_path, gtl::InlinedVector<TensorValue, 4> inputs, NodeDef& geop_node_def,
                         std::string node_name, bool only_run_once = true) {
  Env* env = Env::Default();
  GraphDef graph_def;
  std::string graph_def_path = example_path;
  ReadTextProto(env, graph_def_path, &graph_def);
  for (int i = 0; i < graph_def.node_size(); i++) {
    NodeDef* node_def = graph_def.mutable_node(i);
    if (node_def->name() == node_name) {
      geop_node_def = *node_def;
      OpKernelContext::Params params;
      params.record_tensor_accesses = false;
      auto device = absl::make_unique<DummyDevice>(env, params.record_tensor_accesses);
      params.device = device.get();
      Status status;
      std::unique_ptr<OpKernel> op(
        CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(), *node_def, TF_GRAPH_DEF_VERSION, &status));
      EXPECT_TRUE(status.ok());
      AsyncOpKernel* async_op = op->AsAsync();
      params.op_kernel = async_op;
      params.session_handle = "session_0";
      params.inputs = &inputs;

      // function library
      FunctionDefLibrary func_def_lib = graph_def.library();
      std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), func_def_lib));
      OptimizerOptions opts;
      std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr(new ProcessFunctionLibraryRuntime(
        nullptr, Env::Default(), TF_GRAPH_DEF_VERSION, lib_def.get(), opts, nullptr, nullptr));
      FunctionLibraryRuntime* flr = proc_flr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
      params.function_library = flr;
      auto ctx = absl::make_unique<OpKernelContext>(&params);
      AsyncOpKernel::DoneCallback done = []() { LOG(INFO) << "DONE DoneCallback"; };
      async_op->ComputeAsync(ctx.get(), done);
      if (!only_run_once) {
        auto ctx1 = absl::make_unique<OpKernelContext>(&params);
        async_op->ComputeAsync(ctx1.get(), done);
      }
    }
  }
  return Status::OK();
}

TEST_F(GeOpTest, GeOpFuncTest) {
  NpuClose();
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop.pbtxt";
  gtl::InlinedVector<TensorValue, 4> inputs;
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp1_0").ok());
}
TEST_F(GeOpTest, GeDynamicConfigError) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_dynamic_config.pbtxt";
  gtl::InlinedVector<TensorValue, 4> inputs;
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp61_0").ok());
}
TEST_F(GeOpTest, GeOpOutputError) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_output_error.pbtxt";
  gtl::InlinedVector<TensorValue, 4> inputs;
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp51_0").ok());
}
TEST_F(GeOpTest, GeOpVarInitGraphTest) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_var_init_graph.pbtxt";
  gtl::InlinedVector<TensorValue, 4> inputs;
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp14_0").ok());
}
TEST_F(GeOpTest, GeOpDynamicInputTest) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_dynamic_input_lazy_recompile.pbtxt";
  std::vector<int64_t> ge_output1_dims{2, 2};
  auto getnext_output1_info =
    std::unique_ptr<NpuGetNextOutputInfo>(new NpuGetNextOutputInfo(ge::kPlacementDevice, ge_output1_dims, 8, nullptr));
  Allocator* allocator1 = NpuHostGetNextAllocator::Create(std::move(getnext_output1_info));
  Tensor a(allocator1, DT_INT64, TensorShape({2, 2}));
  std::vector<int64_t> ge_output2_dims{2, 2};
  auto getnext_output2_info =
    std::unique_ptr<NpuGetNextOutputInfo>(new NpuGetNextOutputInfo(ge::kPlacementDevice, ge_output2_dims, 8, nullptr));
  Allocator* allocator2 = NpuHostGetNextAllocator::Create(std::move(getnext_output2_info));
  Tensor b(allocator2, DT_INT64, TensorShape({2, 2}));
  Tensor c(DT_INT32, TensorShape({1,}));
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a), TensorValue(&b), TensorValue(&c)};
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp11_1", false).ok());
  auto attrs = node_def.attr();
  EXPECT_TRUE(attrs.find("_dynamic_input") != attrs.end());
  EXPECT_TRUE(!attrs["_dynamic_input"].s().empty());
}
TEST_F(GeOpTest, GeOpDynamicInputGetNextTest) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_dynamic_input_lazy_recompile.pbtxt";
  gtl::InlinedVector<TensorValue, 4> inputs;
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp11_0").ok());
}
TEST_F(GeOpTest, GeOpDynamicInput1Test) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_dynamic_execute.pbtxt";
  Tensor a(DT_INT32, TensorShape({1,}));
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a)};
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp14_0", false).ok());
  auto attrs = node_def.attr();
  EXPECT_TRUE(attrs.find("_dynamic_input") != attrs.end());
  EXPECT_TRUE(!attrs["_dynamic_input"].s().empty());
  EXPECT_EQ(attrs["_dynamic_graph_execute_mode"].s() == "dynamic_execute", true);
}
TEST_F(GeOpTest, GeOpAoeTuningAndDynamicDimsTest) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_aoe_tuning_and_dynamic_dims.pbtxt";
  Tensor a(DT_INT32, TensorShape({1,}));
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a)};
  setenv("ENABLE_FORCE_V2_CONTROL", "1", true);
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp13_0", false).ok());
}
TEST_F(GeOpTest, GeOpAoeTuningTest) {
  Env* env = Env::Default();
  GraphDef graph_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_aoe_tuning_no_train.pbtxt";
  ReadTextProto(env, graph_def_path, &graph_def);
  for (int i = 0; i < graph_def.node_size(); i++) {
    NodeDef* node_def = graph_def.mutable_node(i);
    if (node_def->name() == "GeOp1_1") {
      auto attrs = node_def->attr();
      EXPECT_TRUE(attrs.find("_aoe_mode") != attrs.end());
      EXPECT_TRUE(!attrs["_aoe_mode"].s().empty());
      EXPECT_TRUE(attrs.find("_work_path") != attrs.end());
      OpKernelContext::Params params;
      params.record_tensor_accesses = false;
      auto device = absl::make_unique<DummyDevice>(env, params.record_tensor_accesses);
      params.device = device.get();
      Status status;
      std::unique_ptr<OpKernel> op(
        CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(), *node_def, TF_GRAPH_DEF_VERSION, &status));
      EXPECT_TRUE(status.ok());
      AsyncOpKernel* async_op = op->AsAsync();
      params.op_kernel = async_op;
      params.session_handle = "session_0";
      gtl::InlinedVector<TensorValue, 4> inputs;
      params.inputs = &inputs;

      // function library
      FunctionDefLibrary func_def_lib = graph_def.library();
      std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), func_def_lib));
      OptimizerOptions opts;
      std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr(new ProcessFunctionLibraryRuntime(
        nullptr, Env::Default(), TF_GRAPH_DEF_VERSION, lib_def.get(), opts, nullptr, nullptr));
      FunctionLibraryRuntime* flr = proc_flr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
      params.function_library = flr;
      int forward_from = 0;
      params.forward_from_array = &forward_from;
      auto ctx = absl::make_unique<OpKernelContext>(&params);
      AsyncOpKernel::DoneCallback done = []() { LOG(INFO) << "DONE DoneCallback"; };
      async_op->ComputeAsync(ctx.get(), done);
      EXPECT_EQ(ctx->status().ok(), true);

      // another graph
      GraphDef train_graph_def;
      std::string train_graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_aoe_tuning.pbtxt";
      ReadTextProto(env, train_graph_def_path, &train_graph_def);
      for (int i = 0; i < train_graph_def.node_size(); i++) {
        NodeDef* node_def = train_graph_def.mutable_node(i);
        if (node_def->name() == "GeOp2_0") {
          auto attrs = node_def->attr();
          EXPECT_TRUE(attrs.find("_aoe_mode") != attrs.end());
          EXPECT_TRUE(!attrs["_aoe_mode"].s().empty());
          EXPECT_TRUE(attrs.find("_work_path") != attrs.end());
          OpKernelContext::Params params;
          params.record_tensor_accesses = false;
          auto device = absl::make_unique<DummyDevice>(env, params.record_tensor_accesses);
          params.device = device.get();
          Status status;
          std::unique_ptr<OpKernel> op(
            CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(), *node_def, TF_GRAPH_DEF_VERSION, &status));
          EXPECT_TRUE(status.ok());
          AsyncOpKernel* async_op = op->AsAsync();
          params.op_kernel = async_op;
          params.session_handle = "session_0";
          gtl::InlinedVector<TensorValue, 4> inputs;
          params.inputs = &inputs;

          // function library
          FunctionDefLibrary func_def_lib = train_graph_def.library();
          std::unique_ptr<FunctionLibraryDefinition> lib_def(
            new FunctionLibraryDefinition(OpRegistry::Global(), func_def_lib));
          OptimizerOptions opts;
          std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr(new ProcessFunctionLibraryRuntime(
            nullptr, Env::Default(), TF_GRAPH_DEF_VERSION, lib_def.get(), opts, nullptr, nullptr));
          FunctionLibraryRuntime* flr = proc_flr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
          params.function_library = flr;
          int forward_from = 0;
          params.forward_from_array = &forward_from;
          auto ctx = absl::make_unique<OpKernelContext>(&params);
          AsyncOpKernel::DoneCallback done = []() { LOG(INFO) << "DONE DoneCallback"; };
          static_cast<GeOp*>(async_op)->session_id_ = 0;
          async_op->ComputeAsync(ctx.get(), done);
          EXPECT_EQ(ctx->status().ok(), true);
        }
      }
    }
  }
}

TEST_F(GeOpTest, GeOpAoeTuningOtherTest) {
  Env* env = Env::Default();
  // another graph
  GraphDef train_graph_def;
  std::string train_graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_aoe_tuning.pbtxt";
  ReadTextProto(env, train_graph_def_path, &train_graph_def);
  for (int i = 0; i < train_graph_def.node_size(); i++) {
    NodeDef* node_def = train_graph_def.mutable_node(i);
    if (node_def->name() == "GeOp2_0") {
      auto attrs = node_def->attr();
      EXPECT_TRUE(attrs.find("_aoe_mode") != attrs.end());
      EXPECT_TRUE(!attrs["_aoe_mode"].s().empty());
      EXPECT_TRUE(attrs.find("_work_path") != attrs.end());
      OpKernelContext::Params params;
      params.record_tensor_accesses = false;
      auto device = absl::make_unique<DummyDevice>(env, params.record_tensor_accesses);
      params.device = device.get();
      Status status;
      std::unique_ptr<OpKernel> op(
        CreateOpKernel(DEVICE_CPU, params.device, cpu_allocator(), *node_def, TF_GRAPH_DEF_VERSION, &status));
      EXPECT_TRUE(status.ok());
      AsyncOpKernel* async_op = op->AsAsync();
      params.op_kernel = async_op;
      params.session_handle = "session_0";
      gtl::InlinedVector<TensorValue, 4> inputs;
      params.inputs = &inputs;

      // function library
      FunctionDefLibrary func_def_lib = train_graph_def.library();
      std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), func_def_lib));
      OptimizerOptions opts;
      std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr(new ProcessFunctionLibraryRuntime(
        nullptr, Env::Default(), TF_GRAPH_DEF_VERSION, lib_def.get(), opts, nullptr, nullptr));
      FunctionLibraryRuntime* flr = proc_flr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
      params.function_library = flr;
      int forward_from = 0;
      params.forward_from_array = &forward_from;
      auto ctx = absl::make_unique<OpKernelContext>(&params);
      AsyncOpKernel::DoneCallback done = []() { LOG(INFO) << "DONE DoneCallback"; };
      static_cast<GeOp*>(async_op)->session_id_ = 9999;
      async_op->ComputeAsync(ctx.get(), done);
      EXPECT_EQ(ctx->status().ok(), false);
    }
  }
}

TEST_F(GeOpTest, GeOpDpOpTest) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_dpop.pbtxt";
  gtl::InlinedVector<TensorValue, 4> inputs;
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp1_0_dp").ok());
}

TEST_F(GeOpTest, GeOpFuncSubGraphTest) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_node_func_subgraph.pbtxt";
  Tensor a(DT_INT32, TensorShape({1,}));
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a)};
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp12_0").ok());
}
TEST_F(GeOpTest, GeOpDynamicDimsTest) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_dynamic_dims.pbtxt";
  Tensor a(DT_INT32, TensorShape({1,}));
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&a)};
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp13_0").ok());
  auto attrs = node_def.attr();
  EXPECT_TRUE(attrs.find("_input_shape") != attrs.end());
  EXPECT_TRUE(!attrs["_input_shape"].s().empty());
}
TEST_F(GeOpTest, GeOpWhileLoopV1Test) {
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_while_loop.pbtxt";
  gtl::InlinedVector<TensorValue, 4> inputs;
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp13_0").ok());
}
TEST_F(GeOpTest, GeOpWhileLoopV2Test) {
  setenv("ENABLE_FORCE_V2_CONTROL", "1", true);
  NodeDef node_def;
  std::string graph_def_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_while_loop.pbtxt";
  gtl::InlinedVector<TensorValue, 4> inputs;
  EXPECT_TRUE(GeOpRunGraphAsync(graph_def_path, inputs, node_def, "GeOp13_0").ok());
}
TEST_F(GeOpTest, GeOpNpuOnnxGraphOpTest) {
  NodeDef node_def;
  std::string grph_pbtxt_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_npu_onnx_graph_op.pbtxt";

  Tensor in(DT_FLOAT, TensorShape({1, 1, 5, 5}));
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&in)};
  EXPECT_TRUE(GeOpRunGraphAsync(grph_pbtxt_path, inputs, node_def, "GeOp91_0", false).ok());
}
TEST_F(GeOpTest, GeOpNpuOnnxGraphOpNoModelTest) {
  NodeDef node_def;
  std::string grph_pbtxt_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_npu_onnx_graph_op_parse.pbtxt";

  Tensor in(DT_FLOAT, TensorShape({1, 1, 5, 5}));
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&in)};
  EXPECT_TRUE(GeOpRunGraphAsync(grph_pbtxt_path, inputs, node_def, "GeOp91_0").ok());
}
TEST_F(GeOpTest, DomiFormatFromStringTest) {
  GeOp* geop_node;
  int32_t domi_format = 0;
  Status ret = geop_node->DomiFormatFromString("NCHW", domi_format);
  EXPECT_EQ(domi_format, domi::domiTensorFormat_t::DOMI_TENSOR_NCHW);
  ret = geop_node->DomiFormatFromString("NHWC", domi_format);
  EXPECT_EQ(domi_format, domi::domiTensorFormat_t::DOMI_TENSOR_NHWC);
  ret = geop_node->DomiFormatFromString("NC1HWC0", domi_format);
  EXPECT_EQ(domi_format, domi::domiTensorFormat_t::DOMI_TENSOR_NC1HWC0);
  ret = geop_node->DomiFormatFromString("NDHWC", domi_format);
  EXPECT_EQ(domi_format, domi::domiTensorFormat_t::DOMI_TENSOR_NDHWC);
  ret = geop_node->DomiFormatFromString("NCDHW", domi_format);
  EXPECT_EQ(domi_format, domi::domiTensorFormat_t::DOMI_TENSOR_NCDHW);
  ret = geop_node->DomiFormatFromString("DHWCN", domi_format);
  EXPECT_EQ(domi_format, domi::domiTensorFormat_t::DOMI_TENSOR_DHWCN);
  ret = geop_node->DomiFormatFromString("DHWNC", domi_format);
  EXPECT_EQ(domi_format, domi::domiTensorFormat_t::DOMI_TENSOR_DHWNC);
  ret = geop_node->DomiFormatFromString("FRACTALZ", domi_format);
  EXPECT_EQ(domi_format, domi::domiTensorFormat_t::DOMI_TENSOR_FRACTAL_Z);
  ret = geop_node->DomiFormatFromString("aa", domi_format);
  EXPECT_TRUE(!ret.ok());

}

TEST_F(GeOpTest, GeOpNpuStringMaxSizeTest) {
  NodeDef node_def;
  std::string grph_pbtxt_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_npu_onnx_graph_op_parse.pbtxt";

  auto buff = reinterpret_cast<char *>(malloc(SECUREC_MEM_MAX_LEN + 1));
  memset_s(buff, SECUREC_MEM_MAX_LEN, '*', SECUREC_MEM_MAX_LEN);
  buff[SECUREC_MEM_MAX_LEN] = '*';
  Tensor in(DT_STRING, TensorShape({1,}));
  in.scalar<tstring>()() = buff;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&in)};
  EXPECT_TRUE(GeOpRunGraphAsync(grph_pbtxt_path, inputs, node_def, "GeOp91_0").ok());
  free(buff);
  buff = nullptr;
}

TEST_F(GeOpTest, UpdateSubgraphMultiDimsAttrTest) {
  GeOp* geop_node;
  Status ret = geop_node->UpdateSubgraphMultiDimsAttr(nullptr, "-1:1", "1:2", "2,2" ,"3,4");
  EXPECT_TRUE(!ret.ok());
}

TEST_F(GeOpTest, BuildSubgraphMuliDimsInput) {
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_shape_map = {{"data0", {4, -1}}, {"data1", {3, 1}}, {"data2", {-1,-1, 5}}};
  std::vector<std::vector<std::string>> dynamic_dims_vec = {{"3", "3", "3"}, {"4", "4", "4"}, {"5", "5", "5"}};
  std::vector<std::string> subgraph_multi_dims_input_shape;
  std::vector<std::string> subgraph_multi_dims_input_dim;
  Status ret = BuildSubgraphMuliDimsInput(user_shape_map, dynamic_dims_vec, subgraph_multi_dims_input_shape, subgraph_multi_dims_input_dim);
  EXPECT_TRUE(ret.ok());
}

TEST_F(GeOpTest, BuildOutputTensorInfo) {
  ge::RegRunGraphAsyncStub(
      [](uint32_t graphId, const std::vector<ge::Tensor> &inputs, ge::RunAsyncCallback callback) -> ge::Status {
      const string str = "abc";
      // extra 16 bytes store head of string
      // extra 1 byte store '\0'
      size_t total_size = str.size() + sizeof(ge::StringHead) + 1U;
      size_t alloc_size = total_size + 63U;
      auto base = std::unique_ptr<uint8_t[], ge::Tensor::DeleteFunc>(new (std::nothrow) uint8_t[alloc_size],
                                                                     [](const uint8_t *ptr) {
                                                                         delete[] ptr;
                                                                         ptr = nullptr;
                                                                     });
      const size_t offset = 63U;
      uint8_t *aligned_addr = ge::PtrToPtr<void, uint8_t>(
              ge::ValueToPtr((ge::PtrToValue(ge::PtrToPtr<uint8_t, void>(base.get())) + offset) & ~offset));

      // front 16 bytes store head of each string
      ge::StringHead *const string_head = ge::PtrToPtr<uint8_t, ge::StringHead>(aligned_addr);
      auto raw_data = ge::PtrAdd<uint8_t>(aligned_addr, total_size + 1U, sizeof(ge::StringHead));
      string_head->addr = static_cast<int64_t>(sizeof(ge::StringHead));
      string_head->len = static_cast<int64_t>(str.size());
      const bool b = memcpy_s(raw_data, str.size() + 1U, str.c_str(), str.size()) == EOK;
      if (!b) {
        LOG(WARNING) << "memcpy failed";
      }

      ge::TensorDesc tensor_desc(ge::Shape({1}), ge::Format::FORMAT_ND, ge::DT_STRING);
      tensor_desc.SetPlacement(ge::kPlacementHost);
      ge::Tensor tensor(tensor_desc);
      const auto base_addr = base.release();
      const auto deleter_func = base.get_deleter();
      tensor.SetData(aligned_addr, total_size, [deleter_func, base_addr](uint8_t *ptr) {
        deleter_func(base_addr);
        ptr = nullptr;
      });
      std::vector<ge::Tensor> outputs;
      outputs.emplace_back(tensor);

      callback(ge::SUCCESS, outputs);
      return ge::SUCCESS;
    });

  NodeDef node_def;
  std::string graph_pbtxt_path = "tf_adapter/tests/ut/kernels/pbtxt/geop_string_op.pbtxt";
  Tensor in(DT_STRING, TensorShape({1}));
  in.scalar<tstring>()() = "ABC";
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&in)};
  EXPECT_TRUE(GeOpRunGraphAsync(graph_pbtxt_path, inputs, node_def, "GeOp1_0").ok());
}
}  // namespace
}  // namespace tensorflow
