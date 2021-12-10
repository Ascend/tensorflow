/* Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <gtest/gtest.h>
#include <memory>

#include "tensorflow/c/c_api.h"

#include "npu_env.h"
#include "npu_unwrap.h"
#include "npu_device_register.h"

#include "common/test_function_library.h"

const char *kNpuDeviceName = "/job:localhost/replica:0/task:0/device:NPU:0";
const int kNpuDeviceIndex = 0;
const std::map<std::string, std::string> kDeviceOptions;

namespace {
class EagerOpResult {
 public:
  explicit EagerOpResult(std::vector<TFE_TensorHandle *> outputs) : outputs_(outputs) {}
  EagerOpResult() = default;
  ~EagerOpResult() {
    for (auto output : outputs_) {
      TFE_DeleteTensorHandle(output);
    }
  }

  TFE_TensorHandle *Get(size_t i) { return outputs_[i]; }

 private:
  std::vector<TFE_TensorHandle *> outputs_;
};

class EagerOpBuilderHelper {
  friend class ST_NpuDevice;

 public:
  explicit EagerOpBuilderHelper(TFE_Context *context, TF_Status *status)
      : ctx_(context), op_(nullptr), status_(TF_NewStatus()) {}

  EagerOpBuilderHelper &Op(const char *op_name) {
    op_ = TFE_NewOp(ctx_, op_name, status_);
    CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);
    TFE_OpSetDevice(op_, kNpuDeviceName, status_);
    CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);
    return *this;
  }

  EagerOpBuilderHelper &Attr(const char *attr_name, TF_DataType value) {
    TFE_OpSetAttrType(op_, attr_name, value);
    return *this;
  }

  EagerOpBuilderHelper &Attr(const char *attr_name, int value) {
    TFE_OpSetAttrInt(op_, attr_name, value);
    return *this;
  }

  EagerOpBuilderHelper &Attr(const char *attr_name, const std::string &value) {
    TFE_OpSetAttrString(op_, attr_name, value.data(), value.size());
    return *this;
  }

  EagerOpBuilderHelper &Attr(const char *attr_name, std::vector<TF_DataType> types) {
    TFE_OpSetAttrTypeList(op_, attr_name, types.data(), types.size());
    return *this;
  }

  EagerOpBuilderHelper &Attr(const char *attr_name, std::vector<std::vector<int64_t>> shapes) {
    std::vector<const int64_t *> shape_dims_array;
    std::vector<int> shape_rank_array;
    for (auto &shape : shapes) {
      shape_dims_array.push_back(shape.data());
      shape_rank_array.push_back(shape.size());
    }
    TFE_OpSetAttrShapeList(op_, attr_name, shape_dims_array.data(), shape_rank_array.data(), shapes.size(), status_);
    CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);
    return *this;
  }

  EagerOpBuilderHelper &Input(TFE_TensorHandle *value) {
    TFE_OpAddInput(op_, value, status_);
    CHECK_EQ(TF_OK, TF_GetCode(status_)) << TF_Message(status_);
    return *this;
  }

  EagerOpBuilderHelper &NumOutputs(size_t num_outputs) {
    outputs_.resize(num_outputs);
    return *this;
  }

  std::shared_ptr<EagerOpResult> RunExpectStatus(TF_Code expected_code) {
    int num_outputs = outputs_.size();
    TFE_Execute(op_, outputs_.data(), &num_outputs, status_);

    EXPECT_EQ(expected_code, TF_GetCode(status_));
    TFE_DeleteOp(op_);
    TF_DeleteStatus(status_);
    return std::make_shared<EagerOpResult>(outputs_);
  }

  TFE_Context *ctx_;  // not owned
  TFE_Op *op_;
  TF_Status *status_;
  std::vector<TFE_TensorHandle *> outputs_;
};
}  // namespace

class ST_NpuDevice : public ::testing::Test {
 public:
  EagerOpBuilderHelper EagerOpBuilder() { return EagerOpBuilderHelper(context, status); }

 protected:
  void SetUp() override {
    status = TF_NewStatus();
    TFE_ContextOptions *opts = TFE_NewContextOptions();
    context = TFE_NewContext(opts, status);
    TFE_DeleteContextOptions(opts);
    CreateDevice(context, kNpuDeviceName, kNpuDeviceIndex, kDeviceOptions);

    for (const auto &function_def : FunctionStrLibrary::Instance().Get()) {
      TFE_ContextAddFunctionDef(context, function_def.data(), function_def.size(), status);
      CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    }
  }

  void TearDown() override {
    ReleaseDeviceResource();
    TFE_DeleteContext(context);
    TF_DeleteStatus(status);
  }

  template <typename T>
  TFE_TensorHandle *CreateCpuHandle(tensorflow::TensorShape shape, std::vector<T> values) {
    tensorflow::Tensor tensor(tensorflow::DataTypeToEnum<T>::value, shape);
    size_t num_elements = std::min(static_cast<size_t>(tensor.NumElements()), values.size());
    T *tensor_data = static_cast<T *>(tensor.data());
    for (size_t i = 0; i < num_elements; i++) {
      *tensor_data++ = values[i];
    }
    return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
  }

  template <typename T>
  TFE_TensorHandle *CreateCpuHandle(T data) {
    tensorflow::Tensor tensor(data);
    return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
  }

  TFE_TensorHandle *CreateNpuResourceHandle(const std::string &container, const std::string &shared_name) {
    tensorflow::Tensor tensor(tensorflow::DT_RESOURCE, tensorflow::TensorShape{});
    auto &handle = tensor.flat<tensorflow::ResourceHandle>()(0);
    handle.set_container(container);
    handle.set_name(shared_name);
    tensorflow::CustomDevice *custom_device = nullptr;
    if (!npu::UnwrapCtx(context)->FindCustomDeviceFromName(kNpuDeviceName, &custom_device)) {
      return nullptr;
    }
    return tensorflow::wrap(
      tensorflow::TensorHandle::CreateLocalHandle(std::move(tensor), custom_device, npu::UnwrapCtx(context)));
  }

  TFE_Context *context;
  TF_Status *status;
};

TEST_F(ST_NpuDevice, function_add) {
  EagerOpBuilder()
    .Op("function_add")
    .Input(CreateCpuHandle(tensorflow::TensorShape{1}, std::vector<float>{1.0}))
    .NumOutputs(1)
    .RunExpectStatus(TF_OK);
}

TEST_F(ST_NpuDevice, eager_select_v2_op) {
  EagerOpBuilder()
    .Op("SelectV2")
    .Input(CreateCpuHandle(tensorflow::TensorShape{1}, std::vector<bool>{true}))
    .Input(CreateCpuHandle(tensorflow::TensorShape{1}, std::vector<float>{1.0}))
    .Input(CreateCpuHandle(tensorflow::TensorShape{1}, std::vector<float>{1.0}))
    .NumOutputs(1)
    .RunExpectStatus(TF_OK);
}

TEST_F(ST_NpuDevice, eager_add_op) {
  EagerOpBuilder()
    .Op("Add")
    .Input(CreateCpuHandle(tensorflow::TensorShape{1}, std::vector<float>{1.0}))
    .Input(CreateCpuHandle(tensorflow::TensorShape{1}, std::vector<float>{1.0}))
    .NumOutputs(1)
    .RunExpectStatus(TF_OK);
}

TEST_F(ST_NpuDevice, eager_unknown_shape_where_op) {
  EagerOpBuilder()
    .Op("Where")
    .Input(CreateCpuHandle(tensorflow::TensorShape{1}, std::vector<bool>{true}))
    .NumOutputs(1)
    .RunExpectStatus(TF_OK);
}

TEST_F(ST_NpuDevice, eager_iterator_v2_op) {
  std::shared_ptr<EagerOpResult> iterator_result =
    EagerOpBuilder()
      .Op("IteratorV2")
      .Attr("shared_name", "")
      .Attr("container", "")
      .Attr("output_types", std::vector<TF_DataType>{TF_INT64})
      .Attr("output_shapes", std::vector<std::vector<int64_t>>{std::vector<int64_t>{}})
      .NumOutputs(1)
      .RunExpectStatus(TF_OK);

  std::shared_ptr<EagerOpResult> dataset_result =
    EagerOpBuilder()
      .Op("RangeDataset")
      .Input(CreateCpuHandle(tensorflow::TensorShape{}, std::vector<int64_t>{1}))
      .Input(CreateCpuHandle(tensorflow::TensorShape{}, std::vector<int64_t>{10}))
      .Input(CreateCpuHandle(tensorflow::TensorShape{}, std::vector<int64_t>{1}))
      .Attr("output_types", std::vector<TF_DataType>{TF_INT64})
      .Attr("output_shapes", std::vector<std::vector<int64_t>>{std::vector<int64_t>{}})
      .NumOutputs(1)
      .RunExpectStatus(TF_OK);

  EagerOpBuilder()
    .Op("MakeIterator")
    .Input(dataset_result->Get(0))
    .Input(iterator_result->Get(0))
    .RunExpectStatus(TF_OK);

  EagerOpBuilder()
    .Op("function_while_consume_iterator")
    .Input(iterator_result->Get(0))
    .Input(dataset_result->Get(0))
    .RunExpectStatus(TF_OK);
}
