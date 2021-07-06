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

#ifndef TENSORFLOW_NPU_UTILS_H
#define TENSORFLOW_NPU_UTILS_H

#include "tensorflow/c/eager/c_api.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/c/eager/abstract_tensor_handle.h"

// clang-format off
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/device_filters.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/shape_inference.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"

#include "npu_env.h"
#include "npu_micros.h"
#include "npu_unwrap.h"

#include "acl/acl_base.h"
#include "graph/types.h"

__attribute__((unused)) static bool IsNpuTensorHandle(tensorflow::TensorHandle *handle) {
  tensorflow::Status status;
  tensorflow::DeviceNameUtils::ParsedName parsed_name;
  return tensorflow::DeviceNameUtils::ParseFullName(handle->DeviceName(&status), &parsed_name) &&
         parsed_name.type == "NPU";
}

__attribute__((unused)) static bool IsCpuTensorHandle(tensorflow::TensorHandle *handle) {
  tensorflow::Status status;
  tensorflow::DeviceNameUtils::ParsedName parsed_name;
  return tensorflow::DeviceNameUtils::ParseFullName(handle->DeviceName(&status), &parsed_name) &&
         parsed_name.type == "CPU";
}

class ScopeTensorHandleDeleter {
 public:
  ScopeTensorHandleDeleter() = default;
  ~ScopeTensorHandleDeleter() {
    for (auto handle : handles_) {
      TFE_DeleteTensorHandle(handle);
    }
  }
  void Guard(TFE_TensorHandle *handle) {
    if (handle != nullptr) {
      handles_.insert(handle);
    }
  }

 private:
  std::unordered_set<TFE_TensorHandle *> handles_;
};

__attribute__((unused)) static tensorflow::Status MapGeType2Tf(ge::DataType ge_type, tensorflow::DataType *tf_type) {
  static std::map<ge::DataType, tensorflow::DataType> kGeType2Tf = {
    {ge::DT_FLOAT, tensorflow::DT_FLOAT},           {ge::DT_DOUBLE, tensorflow::DT_DOUBLE},
    {ge::DT_INT32, tensorflow::DT_INT32},           {ge::DT_UINT8, tensorflow::DT_UINT8},
    {ge::DT_INT16, tensorflow::DT_INT16},           {ge::DT_INT8, tensorflow::DT_INT8},
    {ge::DT_STRING, tensorflow::DT_STRING},         {ge::DT_COMPLEX64, tensorflow::DT_COMPLEX64},
    {ge::DT_INT64, tensorflow::DT_INT64},           {ge::DT_BOOL, tensorflow::DT_BOOL},
    {ge::DT_QINT8, tensorflow::DT_QINT8},           {ge::DT_QUINT8, tensorflow::DT_QUINT8},
    {ge::DT_QINT32, tensorflow::DT_QINT32},         {ge::DT_QINT16, tensorflow::DT_QINT16},
    {ge::DT_QUINT16, tensorflow::DT_QUINT16},       {ge::DT_UINT16, tensorflow::DT_UINT16},
    {ge::DT_COMPLEX128, tensorflow::DT_COMPLEX128}, {ge::DT_RESOURCE, tensorflow::DT_RESOURCE},
    {ge::DT_VARIANT, tensorflow::DT_VARIANT},       {ge::DT_UINT32, tensorflow::DT_UINT32},
    {ge::DT_UINT64, tensorflow::DT_UINT64},         {ge::DT_STRING_REF, tensorflow::DT_STRING_REF},
    {ge::DT_FLOAT16, tensorflow::DT_HALF},
  };
  if (kGeType2Tf.find(ge_type) == kGeType2Tf.end()) {
    return tensorflow::errors::InvalidArgument("Unsupport ge data type enmu value ", ge_type, " by tf");
  }
  *tf_type = kGeType2Tf[ge_type];
  return tensorflow::Status::OK();
}

__attribute__((unused)) static tensorflow::Status MapTfType2Ge(tensorflow::DataType tf_type, ge::DataType *ge_type) {
  static std::map<tensorflow::DataType, ge::DataType> kTfType2Ge = {
    {tensorflow::DT_FLOAT, ge::DT_FLOAT},           {tensorflow::DT_DOUBLE, ge::DT_DOUBLE},
    {tensorflow::DT_INT32, ge::DT_INT32},           {tensorflow::DT_UINT8, ge::DT_UINT8},
    {tensorflow::DT_INT16, ge::DT_INT16},           {tensorflow::DT_INT8, ge::DT_INT8},
    {tensorflow::DT_STRING, ge::DT_STRING},         {tensorflow::DT_COMPLEX64, ge::DT_COMPLEX64},
    {tensorflow::DT_INT64, ge::DT_INT64},           {tensorflow::DT_BOOL, ge::DT_BOOL},
    {tensorflow::DT_QINT8, ge::DT_QINT8},           {tensorflow::DT_QUINT8, ge::DT_QUINT8},
    {tensorflow::DT_QINT32, ge::DT_QINT32},         {tensorflow::DT_QINT16, ge::DT_QINT16},
    {tensorflow::DT_QUINT16, ge::DT_QUINT16},       {tensorflow::DT_UINT16, ge::DT_UINT16},
    {tensorflow::DT_COMPLEX128, ge::DT_COMPLEX128}, {tensorflow::DT_RESOURCE, ge::DT_RESOURCE},
    {tensorflow::DT_VARIANT, ge::DT_VARIANT},       {tensorflow::DT_UINT32, ge::DT_UINT32},
    {tensorflow::DT_UINT64, ge::DT_UINT64},         {tensorflow::DT_STRING_REF, ge::DT_STRING_REF},
    {tensorflow::DT_HALF, ge::DT_FLOAT16},
  };
  if (kTfType2Ge.find(tf_type) == kTfType2Ge.end()) {
    return tensorflow::errors::InvalidArgument("Unsupport tf data type enmu value ", ge_type, " by ge");
  }
  *ge_type = kTfType2Ge[tf_type];
  return tensorflow::Status::OK();
}

__attribute__((unused)) static tensorflow::Status MapGeType2Acl(ge::DataType ge_type, aclDataType *acl_type) {
  static std::map<ge::DataType, aclDataType> kGeType2Acl = {
    {ge::DT_FLOAT, ACL_FLOAT},     {ge::DT_DOUBLE, ACL_DOUBLE}, {ge::DT_INT32, ACL_INT32},
    {ge::DT_UINT8, ACL_UINT8},     {ge::DT_INT16, ACL_INT16},   {ge::DT_INT8, ACL_INT8},
    {ge::DT_STRING, ACL_STRING},   {ge::DT_INT64, ACL_INT64},   {ge::DT_BOOL, ACL_BOOL},
    {ge::DT_UINT16, ACL_UINT16},   {ge::DT_UINT32, ACL_UINT32}, {ge::DT_UINT64, ACL_UINT64},
    {ge::DT_FLOAT16, ACL_FLOAT16},
  };
  if (kGeType2Acl.find(ge_type) == kGeType2Acl.end()) {
    return tensorflow::errors::InvalidArgument("Unsupport ge data type enmu value ", ge_type, " by acl");
  }
  *acl_type = kGeType2Acl[ge_type];
  return tensorflow::Status::OK();
}

__attribute__((unused)) static tensorflow::Status MapGeFormat2Acl(ge::Format ge_format, aclFormat *acl_format) {
  static std::map<ge::Format, aclFormat> kGeFormat2Acl = {{ge::Format::FORMAT_NCHW, ACL_FORMAT_NCHW},
                                                          {ge::Format::FORMAT_NHWC, ACL_FORMAT_NHWC},
                                                          {ge::Format::FORMAT_ND, ACL_FORMAT_ND},
                                                          {ge::Format::FORMAT_NC1HWC0, ACL_FORMAT_NC1HWC0},
                                                          {ge::Format::FORMAT_FRACTAL_Z, ACL_FORMAT_FRACTAL_Z},
                                                          {ge::Format::FORMAT_NC1HWC0_C04, ACL_FORMAT_NC1HWC0_C04},
                                                          {ge::Format::FORMAT_NDHWC, ACL_FORMAT_NDHWC},
                                                          {ge::Format::FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ},
                                                          {ge::Format::FORMAT_NCDHW, ACL_FORMAT_NCDHW},
                                                          {ge::Format::FORMAT_NDC1HWC0, ACL_FORMAT_NDC1HWC0},
                                                          {ge::Format::FORMAT_FRACTAL_Z_3D, ACL_FRACTAL_Z_3D}};
  if (kGeFormat2Acl.find(ge_format) == kGeFormat2Acl.end()) {
    return tensorflow::errors::InvalidArgument("Unsupport ge format enmu value ", ge_format, " by acl");
  }
  *acl_format = kGeFormat2Acl[ge_format];
  return tensorflow::Status::OK();
}

// specify the template in utils.cpp if need
template <typename T>
std::string ToString(T v) {
  return std::to_string(v);
}

template <typename T>
std::string VecToString(std::vector<T> vec) {
  if (vec.empty()) {
    return "[]";
  }
  std::string s = "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    s += ToString(vec[i]);
    if (i != vec.size() - 1) {
      s += ",";
    }
  }
  return s + "]";
}

// TODO:在GE处理中，变量名称作为唯一标识，对于shared_name是"_"开头的变量，由于tensorflow禁止变量名以"_"开头，所以无法直接将shared_name
//  作为Node的name，对于GE，则没有这个限制，因而，这个函数需要能够屏蔽这种差异。
__attribute__((unused)) static std::string WrapResourceName(const std::string &name) {
  if (kCustomKernelEnabled) {
    return name;
  }
  return "cpu_" + name;
}

__attribute__((unused)) static tensorflow::Status LoadGraphDefProto(const std::string &file, tensorflow::GraphDef *def) {
  tensorflow::Status status = tensorflow::Env::Default()->FileExists(file);
  if (!status.ok()) {
    return status;
  }
  if (tensorflow::Env::Default()->IsDirectory(file).ok()) {
    return tensorflow::errors::InvalidArgument(file, " is directory");
  }
  if (tensorflow::str_util::EndsWith(file, ".pb")) {
    ReadBinaryProto(tensorflow::Env::Default(), file, def);
  } else if (tensorflow::str_util::EndsWith(file, ".pbtxt")) {
    ReadTextProto(tensorflow::Env::Default(), file, def);
  } else {
    return tensorflow::errors::InvalidArgument(file, " must ends with .pb or .pbtxt");
  }
  return tensorflow::Status::OK();
}

struct ResourceCompare {
  bool operator()(const tensorflow::ResourceHandle &left, const tensorflow::ResourceHandle &right) const {
    return left.name() < right.name() || left.container() < right.container() || left.device() < right.device();
  }
};

#endif  // TENSORFLOW_NPU_UTILS_H
