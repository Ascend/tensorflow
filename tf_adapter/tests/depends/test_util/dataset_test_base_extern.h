/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_DATA_DATASET_TEST_BASE_EX_H_
#define TENSORFLOW_CORE_DATA_DATASET_TEST_BASE_EX_H_
#include "gtest/gtest.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "iostream"

#if 0
namespace tensorflow {
namespace data {
template <typename BaseDatasetParamsT>
class DataSetParasBaseV3 : public DatasetParams {
 public:
  BaseDatasetParamsT base_dataset_params;
  Tensor input_dataset;
  NodeDef dataset_node_def;
};

template <typename BaseDatasetParamsT, typename TestDatasetParamsT>
class DatasetOpsTestBaseV3 : public DatasetOpsTestBase {
 public:
  Status Initialize(TestDatasetParamsT* dataset_params) override {
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
  Status MakeDatasetOpKernel(const TestDatasetParamsT& dataset_params,
                             std::unique_ptr<OpKernel>* kernel) override {
    TF_RETURN_IF_ERROR(CreateOpKernel(dataset_params.dataset_node_def, kernel));
    return Status::OK();
  }

  virtual Status InitializeForDataset(TestDatasetParamsT* dataset_params)  = 0;
  virtual Status MakeBaseDataset(const BaseDatasetParamsT& params, Tensor* input_dataset)  = 0;
};

template <typename TestDatasetParamsT>
class BaseRangeDatasetOpTest : public DatasetOpsTestBaseV3<RangeDatasetParams, TestDatasetParamsT> {
 protected:
  virtual Status MakeBaseDataset(const RangeDatasetParams& params, Tensor* input_dataset) override {
    return MakeRangeDataset(params, input_dataset));
  }
};
}  // namespace data
}  // namespace tensorflow
#endif
#endif  // TENSORFLOW_CORE_DATA_DATASET_TEST_BASE_EX_H_
