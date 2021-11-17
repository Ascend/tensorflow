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

#include "tensorflow/c/c_api.h"

#include "npu_unwrap.h"
#include "npu_device_register.h"

std::string AddFunction() {
  tensorflow::FunctionDef def;
  CHECK(
    tensorflow::protobuf::TextFormat::ParseFromString("    signature {"
                                                      "      name: 'AddFunction'"
                                                      "      input_arg {"
                                                      "        name: 'a'"
                                                      "        type: DT_FLOAT"
                                                      "      }"
                                                      "      output_arg {"
                                                      "        name: 'o'"
                                                      "        type: DT_FLOAT"
                                                      "      }"
                                                      "    }"
                                                      "    node_def {"
                                                      "      name: 'output'"
                                                      "      op: 'Add'"
                                                      "      input: 'a'"
                                                      "      input: 'a'"
                                                      "      attr {"
                                                      "        key: 'T'"
                                                      "        value {"
                                                      "          type: DT_FLOAT"
                                                      "        }"
                                                      "      }"
                                                      "    }"
                                                      "    ret {"
                                                      "      key: 'o'"
                                                      "      value: 'output:z'"
                                                      "    }",
                                                      &def));
  return def.SerializeAsString();
}

class ST_NpuDevice : public ::testing::Test {
 public:
  static const char *kNpuDeviceName;
  static const int kNpuDeviceIndex;
  static const std::map<std::string, std::string> kDeviceOptions;

 protected:
  void SetUp() override {
    status = TF_NewStatus();
    TFE_ContextOptions *opts = TFE_NewContextOptions();
    context = TFE_NewContext(opts, status);
    TFE_DeleteContextOptions(opts);
    CreateDevice(context, kNpuDeviceName, kNpuDeviceIndex, kDeviceOptions);

    std::string function_def = AddFunction();
    TFE_ContextAddFunctionDef(context, function_def.data(), function_def.size(), status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }
  void TearDown() override {
    ReleaseDeviceResource();
    TFE_DeleteContext(context);
    TF_DeleteStatus(status);
  }

  TFE_TensorHandle *CreateCpuHandle(tensorflow::TensorShape shape, std::vector<int32_t> data) {
    tensorflow::Tensor tensor(tensorflow::DT_INT32, shape);
    return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
  }
  TFE_TensorHandle *CreateCpuHandle(int32_t data) {
    tensorflow::Tensor tensor(data);
    return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
  }
  TFE_TensorHandle *CreateCpuHandle(tensorflow::TensorShape shape, std::vector<float> data) {
    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, shape);
    return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
  }
  TFE_TensorHandle *CreateCpuHandle(float data) {
    tensorflow::Tensor tensor(data);
    return tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensor));
  }
  TFE_TensorHandle *CreateNpuResourceHandle(const tensorflow::TensorShape &shape) {
    tensorflow::Tensor tensor(tensorflow::DT_RESOURCE, shape);
    tensorflow::CustomDevice *custom_device = nullptr;
    if (!npu::UnwrapCtx(context)->FindCustomDeviceFromName(kNpuDeviceName, &custom_device)) {
      return nullptr;
    }
    return tensorflow::wrap(
      tensorflow::TensorHandle::CreateLocalHandle(std::move(tensor), custom_device, npu::UnwrapCtx(context)));
  }

  TFE_Context *context;
  TFE_Op *op;
  std::vector<TFE_TensorHandle *> inputs;
  std::vector<TFE_TensorHandle *> outputs;
  TF_Status *status;
};

const char *ST_NpuDevice::kNpuDeviceName = "/job:localhost/replica:0/task:0/device:NPU:0";
const int ST_NpuDevice::kNpuDeviceIndex = 0;
const std::map<std::string, std::string> ST_NpuDevice::kDeviceOptions;

TEST_F(ST_NpuDevice, register_npu_device) {
  TFE_Op *op = TFE_NewOp(context, "AddFunction", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpSetDevice(op, kNpuDeviceName, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_OpAddInput(op, CreateCpuHandle(1.0f), status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  std::vector<TFE_TensorHandle *> outputs;
  outputs.resize(1);
  int num_outputs = 1;

  TFE_Execute(op, outputs.data(), &num_outputs, status);

  EXPECT_EQ(TF_OK, TF_GetCode(status));
  TFE_DeleteOp(op);
  for (auto output : outputs) {
    TFE_DeleteTensorHandle(output);
  }
}