/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/common/compat_tf1_tf2.h"

namespace tensorflow {
namespace data {
namespace {
class DeviceQueueDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  explicit DeviceQueueDatasetOp(OpKernelConstruction *ctx) : DatasetOpKernel(ctx) {
    CHECK_NOT_NULL(ctx);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &outputTypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &outputShapes_));
  }
  ~DeviceQueueDatasetOp() override = default;

 protected:
  void MakeDataset(OpKernelContext *ctx, DatasetBase **output) override {
    CHECK_NOT_NULL(ctx);
    CHECK_NOT_NULL(output);
    *output = new (std::nothrow) Dataset(ctx, outputTypes_, outputShapes_);
    OP_REQUIRES(ctx, *output != nullptr, errors::InvalidArgument("DeviceQueueDatasetOp: new dataset failed"));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext *ctx, const DataTypeVector &outputTypes,
                     const std::vector<PartialTensorShape> &outputShapes)
        : DatasetBase(DatasetContext(ctx)), outputTypes_(outputTypes),
          outputShapes_(outputShapes) {}

    ~Dataset() override = default;

    const DataTypeVector &output_dtypes() const override { return outputTypes_; }

    const std::vector<PartialTensorShape> &output_shapes() const override { return outputShapes_; }

    string DebugString() const override { return "DeviceQueueDatasetOp::Dataset"; }

    STATUS_FUNCTION_ONLY_TF2(CheckExternalState() const override);

   protected:
    std::unique_ptr<IteratorBase> MakeIteratorInternal(const string &prefix) const override {
      return std::unique_ptr<IteratorBase>(new Iterator({this, npu::CatStr(prefix, "::DeviceQueue")}));
    }

    Status AsGraphDefInternal(SerializationContext *ctx, DatasetGraphDefBuilder *b, Node **output) const override {
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params &params) : DatasetIterator<Dataset>(params) {}

      ~Iterator() override = default;

     protected:
      Status GetNextInternal(IteratorContext *ctx, std::vector<Tensor> *out_tensors, bool *end_of_sequence) override {
        *end_of_sequence = false;
        return Status::OK();
      };
      STATUS_FUNCTION_ONLY_TF2(SaveInternal(SerializationContext *ctx, IteratorStateWriter *writer) override);
      STATUS_FUNCTION_ONLY_TF2(RestoreInternal(IteratorContext *ctx, IteratorStateReader *reader) override);
    };
    const DataTypeVector outputTypes_;
    const std::vector<PartialTensorShape> outputShapes_;
  };
  DataTypeVector outputTypes_;
  std::vector<PartialTensorShape> outputShapes_;
};

REGISTER_KERNEL_BUILDER(Name("DeviceQueueDataset").Device(DEVICE_CPU), DeviceQueueDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
