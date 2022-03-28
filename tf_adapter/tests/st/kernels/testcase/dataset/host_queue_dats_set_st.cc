#define protected public
#include <stdlib.h>
#include "gtest/gtest.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tf_adapter/util/npu_attrs.h"
#include "securec.h"

class HostQueueDatasetOp;
namespace tensorflow {
namespace data {
namespace {

static constexpr char kNodeName[] = "host_queue_dataset";
static constexpr const char *const kChannelName = "channel_name";
static constexpr const char *const kOutputTypes = "output_types";
static constexpr const char *const kOutputShapes = "output_shapes";

class HostQueueDatasetOpTest : public DatasetOpsTestBase {
 protected:
  Status CreateTensorSliceDatasetTensorForQueue(std::vector<Tensor> *const vec, Tensor *tensor) {
    DatasetBase *data_base;
    CreateTensorSliceDataset("tensor_slice_node", vec, &data_base);
    StoreDatasetInVariantTensor(data_base, tensor);
    return Status::OK();
  }

  Status CreateHostQueueDatasetKernel(
      const DataTypeVector &output_types,
      const std::vector<PartialTensorShape> &output_shapes,
      std::unique_ptr<OpKernel> *op_kernel, std::string _local_rank_id) {
    name_utils::OpNameParams params;

    NodeDef node_def =
        test::function::NDef(kNodeName, name_utils::OpName("HostQueue", params),
                             {"geop_dataset", "input_dataset"},
                             {{"channel_name", "channel_001"},
                              {"output_types", output_types},
                              {"_local_rank_id", _local_rank_id},
                              {"_local_device_list", "{0,-1}"},
                              {"output_shapes", output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
    return Status::OK();
  }

  // Create a new `HostQueueDataset` op kernel context.
  Status CreateHostQueueDatasetContext(
      OpKernel *op_kernel, gtl::InlinedVector<TensorValue, 4> *const inputs,
      std::unique_ptr<OpKernelContext> *context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }

 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

struct TestCase {
  std::vector<Tensor> input_tensors;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
};

TestCase NormalizeTestCase() {
  return {
      // input_tensors expected_outputs expected_output_dtypes
      // expected_output_shapes
      {CreateTensor<int64>(TensorShape{10, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      {CreateTensor<int64>(TensorShape{1}, {0})},
      {DT_INT64},
      {PartialTensorShape({1})},
  };
}

TestCase NormalizeTestCaseBig() {
  Tensor temp_tensor(DT_INT64, {10, 10240, 1024});
  Tensor expect_tensor(DT_INT64, {10240, 1024});
  std::vector<int64_t> data;
  data.resize(10 * 10240 * 1024, 1);
  std::vector<int64_t> ex_data;
  ex_data.resize(10240 * 1024, 1);
  memcpy_s(const_cast<char *>(expect_tensor.tensor_data().data()), expect_tensor.tensor_data().size(),
           static_cast<void *>(ex_data.data()), ex_data.size());
  memcpy_s(const_cast<char *>(expect_tensor.tensor_data().data()), expect_tensor.tensor_data().size(),
           static_cast<void *>(ex_data.data()), ex_data.size());
  return {
      {temp_tensor},
      {expect_tensor},
      {DT_INT64},
      {PartialTensorShape({10240, 1024})},
  };
}

TEST_F(HostQueueDatasetOpTest, iterator_getnext) {
  *const_cast<bool *>(&kIsNewDataTransfer) = true;
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = NormalizeTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensorForQueue(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs_for_host_queue_dataset(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&tensor_slice_dataset_tensor)});

  std::unique_ptr<OpKernel> host_queue_dataset_kernel;
  TF_ASSERT_OK(CreateHostQueueDatasetKernel(test_case.expected_output_dtypes,
                                            test_case.expected_output_shapes,
                                            &host_queue_dataset_kernel, "-1"));
  std::unique_ptr<OpKernelContext> host_queue_dataset_context;
  TF_ASSERT_OK(CreateHostQueueDatasetContext(host_queue_dataset_kernel.get(),
                                             &inputs_for_host_queue_dataset,
                                             &host_queue_dataset_context));
  DatasetBase *host_queue_dataset;
  TF_ASSERT_OK(CreateDataset(host_queue_dataset_kernel.get(),
                             host_queue_dataset_context.get(),
                             &host_queue_dataset));
  core::ScopedUnref scoped_unref(host_queue_dataset);

  EXPECT_EQ(host_queue_dataset->node_name(), kNodeName);

  host_queue_dataset->output_dtypes();
  host_queue_dataset->output_shapes();
  host_queue_dataset->DebugString();

  SerializationContext context(SerializationContext::Params{});
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node *output;
  host_queue_dataset->AsGraphDefInternal(&context, &db, &output);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(host_queue_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(host_queue_dataset->MakeIterator(iterator_context.get(),
                                                "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  sleep(2);
  TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                 &end_of_sequence));
}

TEST_F(HostQueueDatasetOpTest, iterator_getnext02) {
  *const_cast<bool *>(&kIsNewDataTransfer) = true;
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = NormalizeTestCaseBig();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensorForQueue(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs_for_host_queue_dataset(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&tensor_slice_dataset_tensor)});

  std::unique_ptr<OpKernel> host_queue_dataset_kernel;
  TF_ASSERT_OK(CreateHostQueueDatasetKernel(test_case.expected_output_dtypes,
                                            test_case.expected_output_shapes,
                                            &host_queue_dataset_kernel, "-1"));
  std::unique_ptr<OpKernelContext> host_queue_dataset_context;
  TF_ASSERT_OK(CreateHostQueueDatasetContext(host_queue_dataset_kernel.get(),
                                             &inputs_for_host_queue_dataset,
                                             &host_queue_dataset_context));
  DatasetBase *host_queue_dataset;
  TF_ASSERT_OK(CreateDataset(host_queue_dataset_kernel.get(),
                             host_queue_dataset_context.get(),
                             &host_queue_dataset));
  core::ScopedUnref scoped_unref(host_queue_dataset);

  EXPECT_EQ(host_queue_dataset->node_name(), kNodeName);

  host_queue_dataset->output_dtypes();
  host_queue_dataset->output_shapes();
  host_queue_dataset->DebugString();

  SerializationContext context(SerializationContext::Params{});
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node *output;
  host_queue_dataset->AsGraphDefInternal(&context, &db, &output);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(host_queue_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(host_queue_dataset->MakeIterator(iterator_context.get(),
                                                "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  sleep(2);
  TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                 &end_of_sequence));
}

TEST_F(HostQueueDatasetOpTest, iterator_getnext03) {
  *const_cast<bool *>(&kIsNewDataTransfer) = true;
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = NormalizeTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensorForQueue(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs_for_host_queue_dataset(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&tensor_slice_dataset_tensor)});

  std::unique_ptr<OpKernel> host_queue_dataset_kernel;
  TF_ASSERT_OK(CreateHostQueueDatasetKernel(test_case.expected_output_dtypes,
                                            test_case.expected_output_shapes,
                                            &host_queue_dataset_kernel, "-1"));
  std::unique_ptr<OpKernelContext> host_queue_dataset_context;
  TF_ASSERT_OK(CreateHostQueueDatasetContext(host_queue_dataset_kernel.get(),
                                             &inputs_for_host_queue_dataset,
                                             &host_queue_dataset_context));
  DatasetBase *host_queue_dataset;
  TF_ASSERT_OK(CreateDataset(host_queue_dataset_kernel.get(),
                             host_queue_dataset_context.get(),
                             &host_queue_dataset));
  core::ScopedUnref scoped_unref(host_queue_dataset);

  EXPECT_EQ(host_queue_dataset->node_name(), kNodeName);

  host_queue_dataset->output_dtypes();
  host_queue_dataset->output_shapes();
  host_queue_dataset->DebugString();

  SerializationContext context(SerializationContext::Params{});
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node *output;
  host_queue_dataset->AsGraphDefInternal(&context, &db, &output);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(host_queue_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(host_queue_dataset->MakeIterator(iterator_context.get(),
                                                "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  sleep(2);
  TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                 &end_of_sequence));
}

TEST_F(HostQueueDatasetOpTest, iterator_getnext_tdt) {
  *const_cast<bool *>(&kIsNewDataTransfer) = false;
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = NormalizeTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensorForQueue(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs_for_host_queue_dataset(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&tensor_slice_dataset_tensor)});

  std::unique_ptr<OpKernel> host_queue_dataset_kernel;
  TF_ASSERT_OK(CreateHostQueueDatasetKernel(test_case.expected_output_dtypes,
                                            test_case.expected_output_shapes,
                                            &host_queue_dataset_kernel, "-1"));
  std::unique_ptr<OpKernelContext> host_queue_dataset_context;
  TF_ASSERT_OK(CreateHostQueueDatasetContext(host_queue_dataset_kernel.get(),
                                             &inputs_for_host_queue_dataset,
                                             &host_queue_dataset_context));
  DatasetBase *host_queue_dataset;
  TF_ASSERT_OK(CreateDataset(host_queue_dataset_kernel.get(),
                             host_queue_dataset_context.get(),
                             &host_queue_dataset));
  core::ScopedUnref scoped_unref(host_queue_dataset);

  EXPECT_EQ(host_queue_dataset->node_name(), kNodeName);

  host_queue_dataset->output_dtypes();
  host_queue_dataset->output_shapes();
  host_queue_dataset->DebugString();

  SerializationContext context(SerializationContext::Params{});
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node *output;
  host_queue_dataset->AsGraphDefInternal(&context, &db, &output);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(host_queue_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(host_queue_dataset->MakeIterator(iterator_context.get(),
                                                "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  sleep(2);
  TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                 &end_of_sequence));
}

TEST_F(HostQueueDatasetOpTest, iterator_getnext05_tdt) {
  *const_cast<bool *>(&kIsNewDataTransfer) = false;
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = NormalizeTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensorForQueue(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs_for_host_queue_dataset(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&tensor_slice_dataset_tensor)});

  std::unique_ptr<OpKernel> host_queue_dataset_kernel;
  TF_ASSERT_OK(CreateHostQueueDatasetKernel(test_case.expected_output_dtypes,
                                            test_case.expected_output_shapes,
                                            &host_queue_dataset_kernel, "-1"));
  std::unique_ptr<OpKernelContext> host_queue_dataset_context;
  TF_ASSERT_OK(CreateHostQueueDatasetContext(host_queue_dataset_kernel.get(),
                                             &inputs_for_host_queue_dataset,
                                             &host_queue_dataset_context));
  DatasetBase *host_queue_dataset;
  TF_ASSERT_OK(CreateDataset(host_queue_dataset_kernel.get(),
                             host_queue_dataset_context.get(),
                             &host_queue_dataset));
  core::ScopedUnref scoped_unref(host_queue_dataset);

  EXPECT_EQ(host_queue_dataset->node_name(), kNodeName);

  host_queue_dataset->output_dtypes();
  host_queue_dataset->output_shapes();
  host_queue_dataset->DebugString();

  SerializationContext context(SerializationContext::Params{});
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node *output;
  host_queue_dataset->AsGraphDefInternal(&context, &db, &output);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(host_queue_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(host_queue_dataset->MakeIterator(iterator_context.get(),
                                                "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  sleep(2);
  TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                 &end_of_sequence));
}

TEST_F(HostQueueDatasetOpTest, iterator_getnext_host_queue) {
  *const_cast<bool *>(&kIsHeterogeneous) = true;
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = NormalizeTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensorForQueue(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs_for_host_queue_dataset(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&tensor_slice_dataset_tensor)});

  std::unique_ptr<OpKernel> host_queue_dataset_kernel;
  TF_ASSERT_OK(CreateHostQueueDatasetKernel(test_case.expected_output_dtypes,
                                            test_case.expected_output_shapes,
                                            &host_queue_dataset_kernel, "-1"));
  std::unique_ptr<OpKernelContext> host_queue_dataset_context;
  TF_ASSERT_OK(CreateHostQueueDatasetContext(host_queue_dataset_kernel.get(),
                                             &inputs_for_host_queue_dataset,
                                             &host_queue_dataset_context));
  DatasetBase *host_queue_dataset;
  TF_ASSERT_OK(CreateDataset(host_queue_dataset_kernel.get(),
                             host_queue_dataset_context.get(),
                             &host_queue_dataset));
  core::ScopedUnref scoped_unref(host_queue_dataset);

  EXPECT_EQ(host_queue_dataset->node_name(), kNodeName);

  host_queue_dataset->output_dtypes();
  host_queue_dataset->output_shapes();
  host_queue_dataset->DebugString();

  SerializationContext context(SerializationContext::Params{});
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node *output;
  host_queue_dataset->AsGraphDefInternal(&context, &db, &output);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(host_queue_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(host_queue_dataset->MakeIterator(iterator_context.get(),
                                                "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  sleep(2);
  TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                 &end_of_sequence));
  *const_cast<bool *>(&kIsHeterogeneous) = false;
}

TEST_F(HostQueueDatasetOpTest, iterator_getnext05_host_queue) {
  *const_cast<bool *>(&kIsHeterogeneous) = true;
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  const TestCase &test_case = NormalizeTestCase();
  Tensor tensor_slice_dataset_tensor(DT_VARIANT, TensorShape({}));
  std::vector<Tensor> inputs_for_tensor_slice_dataset = test_case.input_tensors;
  TF_ASSERT_OK(CreateTensorSliceDatasetTensorForQueue(&inputs_for_tensor_slice_dataset,
                                              &tensor_slice_dataset_tensor));

  gtl::InlinedVector<TensorValue, 4> inputs_for_host_queue_dataset(
      {TensorValue(&tensor_slice_dataset_tensor),
       TensorValue(&tensor_slice_dataset_tensor)});

  std::unique_ptr<OpKernel> host_queue_dataset_kernel;
  TF_ASSERT_OK(CreateHostQueueDatasetKernel(test_case.expected_output_dtypes,
                                            test_case.expected_output_shapes,
                                            &host_queue_dataset_kernel, "-1"));
  std::unique_ptr<OpKernelContext> host_queue_dataset_context;
  TF_ASSERT_OK(CreateHostQueueDatasetContext(host_queue_dataset_kernel.get(),
                                             &inputs_for_host_queue_dataset,
                                             &host_queue_dataset_context));
  DatasetBase *host_queue_dataset;
  TF_ASSERT_OK(CreateDataset(host_queue_dataset_kernel.get(),
                             host_queue_dataset_context.get(),
                             &host_queue_dataset));
  core::ScopedUnref scoped_unref(host_queue_dataset);

  EXPECT_EQ(host_queue_dataset->node_name(), kNodeName);

  host_queue_dataset->output_dtypes();
  host_queue_dataset->output_shapes();
  host_queue_dataset->DebugString();

  SerializationContext context(SerializationContext::Params{});
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node *output;
  host_queue_dataset->AsGraphDefInternal(&context, &db, &output);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(host_queue_dataset_context.get(),
                                     &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(host_queue_dataset->MakeIterator(iterator_context.get(),
                                                "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  sleep(2);
  TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                 &end_of_sequence));
  *const_cast<bool *>(&kIsHeterogeneous) = false;
}
}  // namespace
}  // namespace data
}  // namespace tensorflow
