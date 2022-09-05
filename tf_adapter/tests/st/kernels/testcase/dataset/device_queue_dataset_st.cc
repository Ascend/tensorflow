#include <stdlib.h>
#include <vector>
#include "securec.h"

#include "tensorflow/core/graph/graph_def_builder.h"

#include "tf_adapter/util/npu_attrs.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "dataset_test_base_extern.h"
#include "tensor_testutil.h"

#include "gtest/gtest.h"
#include "tf_adapter/common/adp_logger.h"

namespace tensorflow {
namespace data {
namespace {
constexpr char kNodeName[] = "DeviceQueue";

typedef FunctionDefHelper FDH;

class DeviceQueueDatasetParams : public DatasetParams {
 public:
  DeviceQueueDatasetParams(DataTypeVector output_dtypes, std::vector<PartialTensorShape> output_shapes,
      string node_name)
      : DatasetParams(output_dtypes, output_shapes, node_name) {
      dataset_node_def = test::function::NDef(
        node_name, name_utils::OpName(kNodeName),
        {},
        {{"channel_name", "aaa"},
         {"output_types", output_dtypes},
         {"output_shapes", output_shapes}});
    };

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    return Status::OK();
  }

  NodeDef dataset_node_def;
};

class DeviceQueueDatasetOpTest : public DatasetOpsTestBaseV2<DeviceQueueDatasetParams> {
 public:
  Status Initialize(DeviceQueueDatasetParams* dataset_params) override {
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
    TF_RETURN_IF_ERROR(InitializeForDataset(dataset_params));

    TF_RETURN_IF_ERROR(
        MakeDatasetOpKernel(*dataset_params, &dataset_kernel_));
    gtl::InlinedVector<TensorValue, 4> inputs;
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
  Status MakeDatasetOpKernel(const DeviceQueueDatasetParams& dataset_params,
                             std::unique_ptr<OpKernel>* kernel) override {
    TF_RETURN_IF_ERROR(CreateOpKernel(dataset_params.dataset_node_def, kernel));
    return Status::OK();
  }

  Status InitializeForDataset(DeviceQueueDatasetParams* dataset_params) {
    InitFunctionLibraryRuntime({}, cpu_num_);
    return Status::OK();
  }
};
DeviceQueueDatasetParams DeviceQueueDatasetParams1() {
  return DeviceQueueDatasetParams(/*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}
#if 1
TEST_F(DeviceQueueDatasetOpTest, DatasetParam1) {
  ADP_LOG(INFO) << "====== UT case-1 begin ======";
  auto dataset_params = DeviceQueueDatasetParams1();
  Initialize(&dataset_params);
  ADP_LOG(INFO) << "====== UT case-1 end ======";
}
#endif
}  // namespace
}  // namespace data
}  // namespace tensorflow
