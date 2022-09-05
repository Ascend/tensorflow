#include <stdlib.h>
#include <vector>
#include "securec.h"

#include "tensorflow/core/graph/graph_def_builder.h"

#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/kernels/aicpu/map_dataset_op.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "dataset_test_base_extern.h"
#include "tensor_testutil.h"

#include "gtest/gtest.h"
#include "ascendcl_stub.h"
#include "ge_stub.h"
#include "alog_stub.h"
#include "tf_adapter/common/adp_logger.h"

namespace tensorflow {
namespace data {
namespace {
constexpr char kNodeName[] = "npu_map_dataset";

typedef FunctionDefHelper FDH;

class NpuMapDatasetParams : public DatasetParams {
 public:
  NpuMapDatasetParams(
      RangeDatasetParams range_dataset_params, std::vector<Tensor> other_arguments,
      int64 num_parallel_calls, FunctionDefHelper::AttrValueWrapper func,
      std::vector<FunctionDef> func_lib, DataTypeVector type_arguments,
      bool preserve_cardinality, bool deterministic,
      DataTypeVector output_dtypes, std::vector<PartialTensorShape> output_shapes,
      std::string output_device, string node_name)
      : DatasetParams(output_dtypes, output_shapes, node_name),
        base_dataset_params(std::move(range_dataset_params)),
        other_arguments_(std::move(other_arguments)),
        num_parallel_calls_(CreateTensor<int64>(TensorShape({}), {num_parallel_calls})),
        deterministic_(deterministic),
        func_lib_(std::move(func_lib)) {
      FunctionDef *func_def = &func_lib_[0];
      *func_def->mutable_attr() = func.proto.func().attr();
      dataset_node_def = test::function::NDef(
        node_name, name_utils::OpName(NpuMapDatasetOp::kDatasetType),
        {NpuMapDatasetOp::kInputDataset,
         NpuMapDatasetOp::kNumParallelCalls},
        {{NpuMapDatasetOp::kFunc, func},
         {NpuMapDatasetOp::kTarguments, type_arguments},
         {NpuMapDatasetOp::kOutputShapes, output_shapes},
         {NpuMapDatasetOp::kOutputTypes, output_dtypes},
         {NpuMapDatasetOp::kDeterministic, deterministic},
         {NpuMapDatasetOp::kPreserveCardinality, preserve_cardinality},
         {NpuMapDatasetOp::kOutputDevice, output_device}});
    };

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    if (!IsDatasetTensor(input_dataset)) {
      return tensorflow::errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    }
    *inputs = {TensorValue(&input_dataset)};
    for (auto& argument : other_arguments_) {
      inputs->emplace_back(TensorValue(&argument));
    }
    inputs->emplace_back(TensorValue(&num_parallel_calls_));
    return Status::OK();
  }

  std::vector<FunctionDef> func_lib() const { return func_lib_; }

  string dataset_type() const {
    return NpuMapDatasetOp::kDatasetType;
  }

  RangeDatasetParams base_dataset_params;
  Tensor input_dataset;
  NodeDef dataset_node_def;
 private:
  std::vector<Tensor> other_arguments_;
  Tensor num_parallel_calls_;
  bool deterministic_;
  std::vector<FunctionDef> func_lib_;
};
#if 0
class NpuMapDatasetOpTest : public BaseRangeDatasetOpTest<NpuMapDatasetParams> {
 protected:
  Status InitializeForDataset(TestDatasetParamsT* dataset_params) override {
    return InitFunctionLibraryRuntime(map_dataset_params->func_lib(), cpu_num_);
  }
};
#endif

class NpuMapDatasetOpTest : public DatasetOpsTestBaseV2<NpuMapDatasetParams> {
 public:
  Status Initialize(NpuMapDatasetParams* dataset_params) override {
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
    TF_RETURN_IF_ERROR(InitializeForDataset(dataset_params));

    TF_RETURN_IF_ERROR(
        MakeDatasetOpKernel(*dataset_params, &dataset_kernel_));
    TF_RETURN_IF_ERROR(
        MakeBaseDataset(dataset_params->base_dataset_params,
                         &dataset_params->input_dataset));
    gtl::InlinedVector<TensorValue, 4> inputs;
    TF_RETURN_IF_ERROR(dataset_params->MakeInputs(&inputs));
    TF_RETURN_IF_ERROR(
        CreateDatasetContext(dataset_kernel_.get(), &inputs, &dataset_ctx_));
    TF_RETURN_IF_ERROR(
        CreateDataset(dataset_kernel_.get(), dataset_ctx_.get(), &dataset_));
    TF_RETURN_IF_ERROR(
        CreateIteratorContext(dataset_ctx_.get(), &iterator_ctx_));
    TF_RETURN_IF_ERROR(dataset_->MakeIterator(
        iterator_ctx_.get(), dataset_params->iterator_prefix, &iterator_));
    return Status::OK();
  }

 protected:
  // Creates a new MapDataset op kernel.
  Status MakeDatasetOpKernel(const NpuMapDatasetParams& dataset_params,
                             std::unique_ptr<OpKernel>* kernel) override {
    TF_RETURN_IF_ERROR(CreateOpKernel(dataset_params.dataset_node_def, kernel));
    return Status::OK();
  }

  Status InitializeForDataset(NpuMapDatasetParams* dataset_params) {
    return InitFunctionLibraryRuntime(dataset_params->func_lib(), cpu_num_);
  }

  Status MakeBaseDataset(const RangeDatasetParams& params, Tensor* input_dataset) {
    return MakeRangeDataset(params, input_dataset);
  }

  Status CheckIteratorGetNext(const std::vector<Tensor>& expected_outputs, bool orders) {
    ADP_LOG(INFO) << "Call self defined CheckIteratorGetNext. enter";
    bool end_of_sequence = false;
    auto tensor = expected_outputs.cbegin();
    std::vector<Tensor> expected_outputs_ = expected_outputs;
    while (!end_of_sequence) {
      std::vector<Tensor> next;
      TF_RETURN_IF_ERROR(iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
      ADP_LOG(INFO) << "Call self defined CheckIteratorGetNext. after GetNext end_of_sequence="<<end_of_sequence<<", next.size()="<<next.size();
      if (orders) {
        for (auto next_tensor : next) {
          TF_EXPECT_OK(DatasetOpsTestBase::ExpectEqual(next_tensor, *tensor));
          tensor++;
        }
      } else {
        int size = next.size();
        int i = 0;
        int expected_size = expected_outputs_.size();
        Status status;
        bool is_exist = false;
        for (; i < expected_size; i++) {
          int j;
          for (j = 0; j < size; j++) {
            status = DatasetOpsTestBase::ExpectEqual(expected_outputs_[i + j], next[j]);
            ADP_LOG(INFO) << "Result check: compare expected_outputs_["<<i+j<<"] and next["<<j
              <<"]  status.ok="<<status.ok();
            if (!status.ok()) {
              break;
            }
          }
          if (status.ok() && j == size) {
            is_exist = true;
            expected_outputs_.erase(expected_outputs_.begin() + i, expected_outputs_.begin() + i + size);
            break;
          }
        }
        if (!end_of_sequence && !is_exist) {
          TF_EXPECT_OK(Status(tensorflow::errors::Internal("No ternor was found."))) ;
        }
      }
    }
    ADP_LOG(INFO) << "Call self defined CheckIteratorGetNext. out";
    return Status::OK();
  }

  void SetUp() {
    RegAclRunGraphWithStreamAsyncStub([](uint32_t graph_id, const aclmdlDataset *inputs,
        aclmdlDataset *outputs, void *stream) -> aclError {
      ADP_LOG(INFO) << "Map and batch test RunGraphWithStreamAsyncStub, stream = " << stream;
      AclStreamStub *stub = static_cast<AclStreamStub*>(stream);
      stub->hook = std::bind([](const aclmdlDataset *inputs, aclmdlDataset *outputs) -> aclError {
        ADP_LOG(INFO) << "RunGraphWithStreamAsyncStub-graph process:: input= "
            << inputs << ", output= " << outputs;
        const uint8_t *input = static_cast<uint8_t*>(inputs->blobs[0].dataBuf->data);
        uint8_t *output = static_cast<uint8_t*>(outputs->blobs[0].dataBuf->data);
        ADP_LOG(INFO) << "RunGraphWithStreamAsyncStub-graph process:: input.addr = "
            << (void*)input << ", output.addr = " << (void*)output;
        *reinterpret_cast<int64_t*>(output) = (*reinterpret_cast<const int64_t*>(input)) + 1;
        return ACL_SUCCESS;
      }, std::placeholders::_1, std::placeholders::_2);

      return ACL_SUCCESS;
    });

    RegAclRunGraphStub([](uint32_t graph_id, const aclmdlDataset *inputs,
        aclmdlDataset *outputs) -> aclError {
      ADP_LOG(INFO) << "RegRunGraphStub-graph process:: input= "
            << inputs << ", output= " << outputs;
      const uint8_t *input = static_cast<uint8_t*>(inputs->blobs[0].dataBuf->data);
      ADP_LOG(INFO) << "RegRunGraphStub-graph process:: inputs[0].GetData()="
            << input;

      uint8_t *output = nullptr;
      int64_t tensor_mem_size = DataTypeSize(DT_INT64); // size=8
      rtError_t rt = aclrtMalloc(reinterpret_cast<void **>(&output), tensor_mem_size, ACL_MEM_MALLOC_HUGE_FIRST);
      if (rt != RT_ERROR_NONE) {
        ADP_LOG(ERROR) << errors::InvalidArgument("Alloc mem failed: ", tensor_mem_size, "rtError_t: ", rt);
        return ACL_ERROR_BAD_ALLOC;
      }
      aclDataBuffer *dataBuf = new(std::nothrow) aclDataBuffer(output, tensor_mem_size);
      ADP_LOG(INFO) << "RegRunGraphStub-graph process:: input.addr = "
            << (void*)input << ", size = " << inputs
            << ", output.addr = " << (void*)output;
      *reinterpret_cast<int64_t*>(output) = (*reinterpret_cast<const int64_t*>(input)) + 1;
      dataBuf->data = std::move(output);
      acl::AclModelTensor tensor = acl::AclModelTensor(dataBuf, nullptr);
      tensor.tensorDesc = new(std::nothrow) aclTensorDesc(ACL_INT64);
      outputs->blobs.clear();
      outputs->blobs.emplace_back(tensor);

      return ge::SUCCESS;
    });

    SetLogLevelForC(0);
  }

  void TearDown() {
    RegAclRunGraphWithStreamAsyncStub(nullptr);
    ClearLogLevelForC();
  }
};

FunctionDefHelper::AttrValueWrapper MapFunc(const string& func_name,
                                            const DataType& dtype) {
  return FunctionDefHelper::FunctionRef(func_name, {{"T", dtype}});
}

FunctionDef AddOne() {
  const Tensor kOne = test::AsScalar<int64>(1);
  return FDH::Define(
      // Name
      "AddOne",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"one"}, "Const", {}, {{"value", kOne}, {"dtype", DT_INT64}}},
          {{"y"}, "Add", {"x", "one"}, {{"T", "$T"}}},
      });
}

// test case 1: num_parallel_calls = 1, deterministic = false,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = cpu, output_shapes = static shape
NpuMapDatasetParams NpuMapDatasetParams1() {
  return NpuMapDatasetParams(RangeDatasetParams(1, 4, 1),
                                  /*other_arguments=*/{},
                                  /*num_parallel_calls=*/1,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*deterministic=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({})},
                                  /*output_device=*/"cpu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapDatasetOpTest, DatasetParam1) {
  ADP_LOG(INFO) << "====== UT case-1 begin ======";
  auto dataset_params = NpuMapDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({}), {2}),
                                     CreateTensor<int64>(TensorShape({}), {3}),
                                     CreateTensor<int64>(TensorShape({}), {4})}, /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-1 end ======";
}
#endif

// test case 2: num_parallel_calls = 2, deterministic = false,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = cpu, output_shapes = static shape
NpuMapDatasetParams NpuMapDatasetParams2() {
  return NpuMapDatasetParams(RangeDatasetParams(1, 6, 1),
                                  /*other_arguments=*/{},
                                  /*num_parallel_calls=*/2,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*deterministic=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({})},
                                  /*output_device=*/"cpu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapDatasetOpTest, DatasetParam2) {
  ADP_LOG(INFO) << "====== UT case-2 begin ======";
  auto dataset_params = NpuMapDatasetParams2();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({}), {2}),
                                     CreateTensor<int64>(TensorShape({}), {3}),
                                     CreateTensor<int64>(TensorShape({}), {4}),
                                     CreateTensor<int64>(TensorShape({}), {5}),
                                     CreateTensor<int64>(TensorShape({}), {6})}, /*compare_order=*/ false));
  ADP_LOG(INFO) << "====== UT case-2 end ======";
}
#endif

// test case 3: num_parallel_calls = 2, deterministic = false,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = cpu, output_shapes = static shape
NpuMapDatasetParams NpuMapDatasetParams3() {
  return NpuMapDatasetParams(RangeDatasetParams(1, 6, 1),
                                  /*other_arguments=*/{},
                                  /*num_parallel_calls=*/2,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*deterministic=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({})},
                                  /*output_device=*/"npu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapDatasetOpTest, DatasetParam3) {
  ADP_LOG(INFO) << "====== UT case-3 begin ======";
  auto dataset_params = NpuMapDatasetParams3();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({}), {2}),
                                     CreateTensor<int64>(TensorShape({}), {3}),
                                     CreateTensor<int64>(TensorShape({}), {4}),
                                     CreateTensor<int64>(TensorShape({}), {5}),
                                     CreateTensor<int64>(TensorShape({}), {6})}, /*compare_order=*/ false));
  ADP_LOG(INFO) << "====== UT case-3 end ======";
}
#endif

// test case 4: num_parallel_calls = 2, deterministic = true,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = cpu, output_shapes = static shape
NpuMapDatasetParams NpuMapDatasetParams4() {
  return NpuMapDatasetParams(RangeDatasetParams(1, 6, 1),
                                  /*other_arguments=*/{},
                                  /*num_parallel_calls=*/2,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*deterministic=*/true,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({})},
                                  /*output_device=*/"npu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapDatasetOpTest, DatasetParam4) {
  ADP_LOG(INFO) << "====== UT case-4 begin ======";
  auto dataset_params = NpuMapDatasetParams4();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({}), {2}),
                                     CreateTensor<int64>(TensorShape({}), {3}),
                                     CreateTensor<int64>(TensorShape({}), {4}),
                                     CreateTensor<int64>(TensorShape({}), {5}),
                                     CreateTensor<int64>(TensorShape({}), {6})}, /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-4 end ======";
}
#endif

// test case 5: num_parallel_calls = 2, deterministic = true,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = cpu, output_shapes = dynamic shape
NpuMapDatasetParams NpuMapDatasetParams5() {
  return NpuMapDatasetParams(RangeDatasetParams(1, 6, 1),
                                  /*other_arguments=*/{},
                                  /*num_parallel_calls=*/2,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*deterministic=*/true,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({-1})},
                                  /*output_device=*/"cpu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapDatasetOpTest, DatasetParam5) {
  ADP_LOG(INFO) << "====== UT case-5 begin ======";
  auto dataset_params = NpuMapDatasetParams5();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({}), {2}),
                                     CreateTensor<int64>(TensorShape({}), {3}),
                                     CreateTensor<int64>(TensorShape({}), {4}),
                                     CreateTensor<int64>(TensorShape({}), {5}),
                                     CreateTensor<int64>(TensorShape({}), {6})}, /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-5 end ======";
}
#endif

// test case 6: num_parallel_calls = 2, deterministic = true,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = npu, output_shapes = dynamic shape
NpuMapDatasetParams NpuMapDatasetParams6() {
  return NpuMapDatasetParams(RangeDatasetParams(1, 6, 1),
                                  /*other_arguments=*/{},
                                  /*num_parallel_calls=*/2,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*deterministic=*/true,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({-1})},
                                  /*output_device=*/"npu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapDatasetOpTest, DatasetParam6) {
  ADP_LOG(INFO) << "====== UT case-6 begin ======";
  auto dataset_params = NpuMapDatasetParams6();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({}), {2}),
                                     CreateTensor<int64>(TensorShape({}), {3}),
                                     CreateTensor<int64>(TensorShape({}), {4}),
                                     CreateTensor<int64>(TensorShape({}), {5}),
                                     CreateTensor<int64>(TensorShape({}), {6})}, /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-6 end ======";
}
#endif
}  // namespace
}  // namespace data
}  // namespace tensorflow
