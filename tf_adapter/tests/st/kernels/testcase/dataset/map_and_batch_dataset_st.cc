#include <stdlib.h>
#include <vector>
#include "securec.h"

#include "tensorflow/core/graph/graph_def_builder.h"

#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/kernels/aicpu/map_and_batch_dataset_op.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/framework/types.h"
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
constexpr char kNodeName[] = "npu_map_and_batch_dataset";

typedef FunctionDefHelper FDH;

class NpuMapAndBatchDatasetParams : public DatasetParams {
 public:
  NpuMapAndBatchDatasetParams(
      RangeDatasetParams range_dataset_params, std::vector<Tensor> other_arguments,
      int64 batch_size, int64 num_parallel_calls, bool drop_remainder,
      FunctionDefHelper::AttrValueWrapper func,
      std::vector<FunctionDef> func_lib, DataTypeVector type_arguments,
      bool preserve_cardinality, DataTypeVector output_dtypes,
      std::vector<PartialTensorShape> output_shapes,
      std::string output_device,
      string node_name)
      : DatasetParams(output_dtypes, output_shapes, node_name),
        base_dataset_params(std::move(range_dataset_params)),
        other_arguments_(std::move(other_arguments)),
        batch_size_(CreateTensor<int64>(TensorShape({}), {batch_size})),
        num_parallel_calls_(CreateTensor<int64>(TensorShape({}), {num_parallel_calls})),
        drop_remainder_(CreateTensor<bool>(TensorShape({}), {drop_remainder})),
        func_lib_(std::move(func_lib)) {
      FunctionDef *func_def = &func_lib_[0];
      *func_def->mutable_attr() = func.proto.func().attr();
      dataset_node_def = test::function::NDef(
        node_name, name_utils::OpName(NpuMapAndBatchDatasetOp::kDatasetType),
        {NpuMapAndBatchDatasetOp::kInputDataset,
         NpuMapAndBatchDatasetOp::kBatchSize,
         NpuMapAndBatchDatasetOp::kNumParallelCalls,
         NpuMapAndBatchDatasetOp::kDropRemainder},
        {{NpuMapAndBatchDatasetOp::kFunc, func},
         {NpuMapAndBatchDatasetOp::kTarguments, type_arguments},
         {NpuMapAndBatchDatasetOp::kOutputShapes, output_shapes},
         {NpuMapAndBatchDatasetOp::kOutputTypes, output_dtypes},
         {NpuMapAndBatchDatasetOp::kPreserveCardinality, preserve_cardinality},
         {NpuMapAndBatchDatasetOp::kOutputDevice, output_device}});
    }

  NpuMapAndBatchDatasetParams(
      std::vector<Tensor> input_tensors, std::vector<Tensor> other_arguments,
      int64 batch_size, int64 num_parallel_calls, bool drop_remainder,
      FunctionDefHelper::AttrValueWrapper func,
      std::vector<FunctionDef> func_lib, DataTypeVector type_arguments,
      bool preserve_cardinality, DataTypeVector output_dtypes,
      std::vector<PartialTensorShape> output_shapes,
      std::string output_device,
      string node_name)
      : DatasetParams(output_dtypes, output_shapes, node_name),
        base_dataset_params(RangeDatasetParams(1, 2, 1)),
        input_tensors_(std::move(input_tensors)),
        other_arguments_(std::move(other_arguments)),
        batch_size_(CreateTensor<int64>(TensorShape({}), {batch_size})),
        num_parallel_calls_(CreateTensor<int64>(TensorShape({}), {num_parallel_calls})),
        drop_remainder_(CreateTensor<bool>(TensorShape({}), {drop_remainder})),
        func_lib_(std::move(func_lib)) {
      FunctionDef *func_def = &func_lib_[0];
      *func_def->mutable_attr() = func.proto.func().attr();
      dataset_node_def = test::function::NDef(
        node_name, name_utils::OpName(NpuMapAndBatchDatasetOp::kDatasetType),
        {NpuMapAndBatchDatasetOp::kInputDataset,
         NpuMapAndBatchDatasetOp::kBatchSize,
         NpuMapAndBatchDatasetOp::kNumParallelCalls,
         NpuMapAndBatchDatasetOp::kDropRemainder},
        {{NpuMapAndBatchDatasetOp::kFunc, func},
         {NpuMapAndBatchDatasetOp::kTarguments, type_arguments},
         {NpuMapAndBatchDatasetOp::kOutputShapes, output_shapes},
         {NpuMapAndBatchDatasetOp::kOutputTypes, output_dtypes},
         {NpuMapAndBatchDatasetOp::kPreserveCardinality, preserve_cardinality},
         {NpuMapAndBatchDatasetOp::kOutputDevice, output_device}});
    }

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    if (!IsDatasetTensor(input_dataset)) {
      return tensorflow::errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    }
    *inputs = {TensorValue(&input_dataset)};
    for (auto& argument : other_arguments_) {
      inputs->emplace_back(TensorValue(&argument));
    }
    inputs->emplace_back(TensorValue(&batch_size_));
    inputs->emplace_back(TensorValue(&num_parallel_calls_));
    inputs->emplace_back(TensorValue(&drop_remainder_));
    return Status::OK();
  }

  std::vector<FunctionDef> func_lib() const { return func_lib_; }

  string dataset_type() const {
    return NpuMapAndBatchDatasetOp::kDatasetType;
  }

  RangeDatasetParams base_dataset_params;
  std::vector<Tensor> input_tensors_;
  Tensor input_dataset;
  NodeDef dataset_node_def;
 private:
  std::vector<Tensor> other_arguments_;
  Tensor batch_size_;
  Tensor num_parallel_calls_;
  Tensor drop_remainder_;
  std::vector<FunctionDef> func_lib_;
};

class NpuMapAndBatchDatasetOpTestBase : public DatasetOpsTestBaseV2<NpuMapAndBatchDatasetParams> {
 public:
  Status Initialize(NpuMapAndBatchDatasetParams* dataset_params) override {
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
    TF_RETURN_IF_ERROR(InitializeForDataset(dataset_params));

    TF_RETURN_IF_ERROR(
        MakeDatasetOpKernel(*dataset_params, &dataset_kernel_));
    TF_RETURN_IF_ERROR(InitializeInput(dataset_params));
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

  virtual Status InitializeInput(NpuMapAndBatchDatasetParams* dataset_params) = 0;

 protected:
  // Creates a new MapDataset op kernel.
  Status MakeDatasetOpKernel(const NpuMapAndBatchDatasetParams& dataset_params,
                             std::unique_ptr<OpKernel>* kernel) override {
    TF_RETURN_IF_ERROR(CreateOpKernel(dataset_params.dataset_node_def, kernel));
    return Status::OK();
  }

  Status InitializeForDataset(NpuMapAndBatchDatasetParams* dataset_params) {
    return InitFunctionLibraryRuntime(dataset_params->func_lib(), cpu_num_);
  }

  Status CheckIteratorGetNext(const std::vector<Tensor>& expected_outputs, bool orders) {
    ADP_LOG(INFO) << "Call self defined CheckIteratorGetNext. enter";
    bool end_of_sequence = false;
    auto tensor = expected_outputs.cbegin();
    std::vector<Tensor> expected_outputs_ = expected_outputs;
    while (!end_of_sequence) {
      std::vector<Tensor> next;
      ADP_LOG(INFO) << "Call self defined CheckIteratorGetNext. before GetNext end_of_sequence="<<end_of_sequence<<", next.size()="<<next.size();
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

  void TearDown() {
    RegAclRunGraphWithStreamAsyncStub(nullptr);
    ClearLogLevelForC();
  }
};

class NpuMapAndBatchDatasetOpTest : public NpuMapAndBatchDatasetOpTestBase {
 public:
  Status InitializeInput(NpuMapAndBatchDatasetParams* dataset_params) override {
    return MakeBaseDataset(dataset_params->base_dataset_params,
                           &dataset_params->input_dataset);
  }

 protected:
  Status MakeBaseDataset(const RangeDatasetParams& params, Tensor* input_dataset) {
    return MakeRangeDataset(params, input_dataset);
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
      aclDataBuffer *dataBuf = new aclDataBuffer(output, tensor_mem_size);
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

    // Register funciton for ge RunGraph
    #if 0
    ge::RegRunGraphWithStreamAsyncStub([](uint32_t graph_id, void *stream, const std::vector<ge::Tensor> &inputs,
        std::vector<ge::Tensor> &outputs) -> ge::Status {
      ADP_LOG(INFO) << "Map and batch test RunGraphWithStreamAsyncStub, stream = " << stream;
      AclStreamStub *stub = static_cast<AclStreamStub*>(stream);
      stub->hook = std::bind([](const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) -> ge::Status {
        ADP_LOG(INFO) << "RunGraphWithStreamAsyncStub-graph process:: input.num = "
            << inputs.size() << ", output.num = " << outputs.size();
        const uint8_t *input = inputs[0].GetData();
        uint8_t *output = outputs[0].GetData();
        ADP_LOG(INFO) << "RunGraphWithStreamAsyncStub-graph process:: input.addr = "
            << (void*)input << ", size = " << inputs[0].GetSize()
            << ", output.addr = " << (void*)output << ", size = " << outputs[0].GetSize();
        *reinterpret_cast<int64_t*>(output) = (*reinterpret_cast<const int64_t*>(input)) + 1;
        return ge::SUCCESS;
      }, std::placeholders::_1, std::placeholders::_2);

      return ge::SUCCESS;
    });

    ge::RegRunGraphStub([](uint32_t graph_id, const std::vector<ge::Tensor> &inputs,
        std::vector<ge::Tensor> &outputs) -> ge::Status {
      ADP_LOG(INFO) << "Map and batch test RegRunGraphStub";
      ADP_LOG(INFO) << "RegRunGraphStub-graph process:: input.num = "
            << inputs.size() << ", output.num = " << outputs.size();
      const uint8_t *input = inputs[0].GetData();
      ADP_LOG(INFO) << "RegRunGraphStub-graph process:: inputs[0].GetData()="
            << (void*)inputs[0].GetData();

      uint8_t *output = nullptr;
      int64_t tensor_mem_size = DataTypeSize(DT_INT64); // size=8
      rtError_t rt = rtMalloc(reinterpret_cast<void **>(&output), tensor_mem_size, RT_MEMORY_HBM);
      if (rt != RT_ERROR_NONE) {
        ADP_LOG(ERROR) << errors::InvalidArgument("Alloc mem failed: ", tensor_mem_size, "rtError_t: ", rt);
        return ge::FAILED;
      }
      outputs.resize(1);
      // Empty shape for scalar
      std::vector<int64_t> output_shape;
      ge::TensorDesc tensor_desc = outputs[0].GetTensorDesc();
      tensor_desc.Update(ge::Shape(output_shape), ge::Format::FORMAT_ND, ge::DT_INT64);
      outputs[0].SetTensorDesc(tensor_desc);
      outputs[0].SetData(output, tensor_mem_size, [](uint8_t *p){ delete p;});

      ADP_LOG(INFO) << "RegRunGraphStub-graph process:: input.addr = "
            << (void*)input << ", size = " << inputs[0].GetSize()
            << ", output.addr = " << (void*)output << ", size = " << outputs[0].GetSize();
      *reinterpret_cast<int64_t*>(output) = (*reinterpret_cast<const int64_t*>(input)) + 1;

      return ge::SUCCESS;
    });
    #endif

    SetLogLevelForC(0);
  }
};

class NpuMapAndBatchDatasetOpDTStringTest : public NpuMapAndBatchDatasetOpTestBase {
 public:
  Status InitializeInput(NpuMapAndBatchDatasetParams* dataset_params) override {
    dataset_params->input_dataset = Tensor(DT_VARIANT, TensorShape({}));
    return CreateTensorSliceDatasetTensor(&dataset_params->input_tensors_,
                                          &dataset_params->input_dataset);
  }

 protected:
  // Creates `TensorSliceDataset` variant tensor from the input vector of tensors.
  Status CreateTensorSliceDatasetTensor(
      std::vector<Tensor> *const tensor_vector, Tensor *dataset_tensor) {
    DatasetBase *tensor_slice_dataset;
    TF_RETURN_IF_ERROR(CreateTensorSliceDataset(
        "tensor_slice_node", tensor_vector, &tensor_slice_dataset));
    TF_RETURN_IF_ERROR(
        StoreDatasetInVariantTensor(tensor_slice_dataset, dataset_tensor));
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

        uint64_t addr_offset = sizeof(ge::StringHead);
        const uint8_t *input = static_cast<uint8_t*>(inputs->blobs[0].dataBuf->data + addr_offset);
        uint8_t *output = static_cast<uint8_t*>(outputs->blobs[0].dataBuf->data);
        ADP_LOG(INFO) << "RunGraphWithStreamAsyncStub-graph process:: input.addr = "
            << (void*)input << ", output.addr = " << (void*)output;
        *reinterpret_cast<float*>(output) = (*reinterpret_cast<const char*>(input));
        return ACL_SUCCESS;
      }, std::placeholders::_1, std::placeholders::_2);

      return ACL_SUCCESS;
    });

    SetLogLevelForC(0);
  }
};

FunctionDefHelper::AttrValueWrapper MapFunc(const string& func_name,
                                            const DataType& dtype) {
  return FunctionDefHelper::FunctionRef(func_name, {{"T", dtype}});
}

FunctionDefHelper::AttrValueWrapper MapFuncConvertString2Num(const DataType& dtype_input,
                                                             const DataType& dtype_output) {
  return FunctionDefHelper::FunctionRef("ConvertString2Num",{{"T_input", dtype_input},
                                                             {"T_output", dtype_output}});
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

FunctionDef ConvertString2Num() {
  return FDH::Define(
      "ConvertString2Num",
      {"x: T_input"},
      {"y: T_output"},
      {"T_input: {string}", "T_output: {float, double, int32, int64}"},
      {{{"y"}, "StringToNumber", {"x"}, {{"T_output", "$T_output"}}}});
}

// test case 1: num_parallel_calls = 1, drop_remainder = false,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = cpu, output_shapes = static shape
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams1() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 4, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/1,
                                  /*drop_remainder=*/false,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({3})},
                                  /*output_device=*/"cpu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam1) {
  ADP_LOG(INFO) << "====== UT case-1 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext(CreateTensors<int64>(TensorShape({3}), {{2,3,4}}), /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-1 end ======";
}
#endif

// test case 2: num_parallel_calls = 2, drop_remainder = false,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = cpu, output_shapes = static shape
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams2() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 7, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/2,
                                  /*drop_remainder=*/false,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({3})},
                                  /*output_device=*/"cpu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam2) {
  ADP_LOG(INFO) << "====== UT case-2 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams2();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext(CreateTensors<int64>(TensorShape({3}), {{2,3,4},{5,6,7}}), /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-2 end ======";
}
#endif

// test case 3: num_parallel_calls = 2, drop_remainder = true,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = cpu, output_shapes = static shape
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams3() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 6, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/2,
                                  /*drop_remainder=*/true,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({3})},
                                  /*output_device=*/"cpu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam3) {
  ADP_LOG(INFO) << "====== UT case-3 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams3();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext(CreateTensors<int64>(TensorShape({3}), {{2,3,4}}), /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-3 end ======";
}
#endif

// test case 4: num_parallel_calls = 1, drop_remainder = false,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = npu, output_shapes = static shape
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams4() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 4, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/1,
                                  /*drop_remainder=*/false,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({3})},
                                  /*output_device=*/"npu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam4) {
  ADP_LOG(INFO) << "====== UT case-4 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams4();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext(CreateTensors<int64>(TensorShape({3}), {{2,3,4}}), /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-4 end ======";
}
#endif

// test case 5: num_parallel_calls = 2, drop_remainder = false,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = npu, output_shapes = static shape
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams5() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 7, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/2,
                                  /*drop_remainder=*/false,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({3})},
                                  /*output_device=*/"npu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam5) {
  ADP_LOG(INFO) << "====== UT case-5 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams5();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext(CreateTensors<int64>(TensorShape({3}), {{2,3,4},{5,6,7}}), /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-5 end ======";
}
#endif

// test case 6: num_parallel_calls = 2, drop_remainder = true,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = npu, output_shapes = static shape
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams6() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 6, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/2,
                                  /*drop_remainder=*/true,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({3})},
                                  /*output_device=*/"npu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam6) {
  ADP_LOG(INFO) << "====== UT case-6 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams6();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext(CreateTensors<int64>(TensorShape({3}), {{2,3,4}}), /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-6 end ======";
}
#endif

// test case 7: num_parallel_calls = 2, drop_remainder = false,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = cpu, output_shapes = dynamic shape
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams7() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 6, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/2,
                                  /*drop_remainder=*/false,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({-1, -1})},
                                  /*output_device=*/"cpu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam7) {
  ADP_LOG(INFO) << "====== UT case-7 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams7();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({3}), {2,3,4}),
                                     CreateTensor<int64>(TensorShape({2}), {5,6})}, /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-7 end ======";
}
#endif

// test case 8: num_parallel_calls = 2, drop_remainder = false,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = npu, output_shapes = dynamic shape
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams8() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 6, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/2,
                                  /*drop_remainder=*/false,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({-1, -1})},
                                  /*output_device=*/"npu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam8) {
  ADP_LOG(INFO) << "====== UT case-8 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams8();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({3}), {2,3,4}),
                                     CreateTensor<int64>(TensorShape({2}), {5,6})}, /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-8 end ======";
}
#endif

// test case 9: num_parallel_calls = 2, drop_remainder = false,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = npu, output_shapes = dynamic shape
// test for multiple thread and multiple max_batch_results
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams9() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 15, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/4,
                                  /*drop_remainder=*/false,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({-1, -1})},
                                  /*output_device=*/"npu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam9) {
  ADP_LOG(INFO) << "====== UT case-9 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams9();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({3}), {2, 3, 4}),
                                     CreateTensor<int64>(TensorShape({3}), {5, 6, 7}),
                                     CreateTensor<int64>(TensorShape({3}), {8, 9, 10}),
                                     CreateTensor<int64>(TensorShape({3}), {11, 12, 13}),
                                     CreateTensor<int64>(TensorShape({2}), {14, 15})}, /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-9 end ======";
}
#endif

// test case 10: num_parallel_calls = 2, drop_remainder = true,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = npu, output_shapes = dynamic shape
// test for multiple thread and multiple max_batch_results
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams10() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 15, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/4,
                                  /*drop_remainder=*/true,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({-1, -1})},
                                  /*output_device=*/"npu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam10) {
  ADP_LOG(INFO) << "====== UT case-10 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams10();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({3}), {2, 3, 4}),
                                     CreateTensor<int64>(TensorShape({3}), {5, 6, 7}),
                                     CreateTensor<int64>(TensorShape({3}), {8, 9, 10}),
                                     CreateTensor<int64>(TensorShape({3}), {11, 12, 13})}, /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-10 end ======";
}
#endif

// test case 11: num_parallel_calls = 2, drop_remainder = true,
// preserve_cardinality = false, MapFunc = AddOne
// output_device = npu, output_shapes = dynamic shape
// test for multiple threads process small amount of data
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams11() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 5, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/4,
                                  /*drop_remainder=*/true,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/false,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({-1, -1})},
                                  /*output_device=*/"npu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, DatasetParam11) {
  ADP_LOG(INFO) << "====== UT case-11 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams11();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext({CreateTensor<int64>(TensorShape({3}), {2, 3, 4}),
                                     CreateTensor<int64>(TensorShape({1}), {5})}, /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-11 end ======";
}


TEST_F(NpuMapAndBatchDatasetOpTest, DebugString) {
  ADP_LOG(INFO) << "====== UT case-12 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  EXPECT_EQ(dataset_->DebugString(), "NpuMapAndBatchDatasetOp::DataSet");
  ADP_LOG(INFO) << "====== UT case-12 end ======";
}
#endif

NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetParams12() {
  return NpuMapAndBatchDatasetParams(RangeDatasetParams(1, 4, 1),
                                  /*other_arguments=*/{},
                                  /*batch_size=*/3,
                                  /*num_parallel_calls=*/1,
                                  /*drop_remainder=*/false,
                                  /*func=*/MapFunc("AddOne", DT_INT64),
                                  /*func_lib=*/{AddOne()},
                                  /*type_arguments*/ {},
                                  /*preserve_cardinality=*/true,
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({3})},
                                  /*output_device=*/"cpu",
                                  /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpTest, Cardinality) {
  ADP_LOG(INFO) << "====== UT case-13 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetParams12();
  TF_ASSERT_OK(Initialize(&dataset_params));
  int64_t ret_val = 1;
  EXPECT_EQ(dataset_->Cardinality(), ret_val);
  ADP_LOG(INFO) << "====== UT case-13 end ======";
}
#endif

// test case 14: test for DT_String input datatype
NpuMapAndBatchDatasetParams NpuMapAndBatchDatasetDTStringParams1() {
  return NpuMapAndBatchDatasetParams({CreateTensor<tstring>(TensorShape{3}, {"a", "b", "c"})},
                                      /*other_arguments=*/{},
                                      /*batch_size=*/3,
                                      /*num_parallel_calls=*/1,
                                      /*drop_remainder=*/false,
                                      /*func=*/MapFuncConvertString2Num(DT_STRING, DT_FLOAT),
                                      /*func_lib=*/{ConvertString2Num()},
                                      /*type_arguments*/ {},
                                      /*preserve_cardinality=*/false,
                                      /*output_dtypes=*/{DT_FLOAT},
                                      /*output_shapes=*/{PartialTensorShape({3})},
                                      /*output_device=*/"cpu",
                                      /*node_name=*/kNodeName);
}
#if 1
TEST_F(NpuMapAndBatchDatasetOpDTStringTest, DatasetDTStringParam1) {
  ADP_LOG(INFO) << "====== UT case-14 begin ======";
  auto dataset_params = NpuMapAndBatchDatasetDTStringParams1();
  TF_ASSERT_OK(Initialize(&dataset_params));
  TF_ASSERT_OK(CheckIteratorGetNext(CreateTensors<float>(TensorShape({3}), {{97, 98, 99}}), /*compare_order=*/ true));
  ADP_LOG(INFO) << "====== UT case-14 end ======";
}
#endif
}  // namespace
}  // namespace data
}  // namespace tensorflow
