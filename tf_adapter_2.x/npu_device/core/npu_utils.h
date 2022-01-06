/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef TENSORFLOW_NPU_UTILS_H
#define TENSORFLOW_NPU_UTILS_H

#include "tensorflow/c/eager/c_api.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/c/eager/abstract_tensor_handle.h"

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
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "npu_env.h"
#include "npu_unwrap.h"

#include "acl/acl_base.h"
#include "graph/types.h"

namespace npu {
inline std::string CatStr(const tensorflow::strings::AlphaNum &a) { return StrCat(a); }

inline std::string CatStr(const tensorflow::strings::AlphaNum &a, const tensorflow::strings::AlphaNum &b) {
  return StrCat(a, b);
}

inline std::string CatStr(const tensorflow::strings::AlphaNum &a, const tensorflow::strings::AlphaNum &b,
                          const tensorflow::strings::AlphaNum &c) {
  return StrCat(a, b, c);
}

inline std::string CatStr(const tensorflow::strings::AlphaNum &a, const tensorflow::strings::AlphaNum &b,
                          const tensorflow::strings::AlphaNum &c, const tensorflow::strings::AlphaNum &d) {
  return StrCat(a, b, c, d);
}

template <typename... AV>
inline std::string CatStr(const tensorflow::strings::AlphaNum &a, const tensorflow::strings::AlphaNum &b,
                          const tensorflow::strings::AlphaNum &c, const tensorflow::strings::AlphaNum &d,
                          const tensorflow::strings::AlphaNum &e, const AV &... args) {
  return StrCat(a, b, c, d, e, args...);
}

/**
 * @brief: is npu tensor handle or not
 * @param handle: tensor handle
 */
bool IsNpuTensorHandle(const tensorflow::TensorHandle *handle);

/**
 * @brief: is cpu tensor handle or not
 * @param handle: tensor handle
 */
bool IsCpuTensorHandle(const tensorflow::TensorHandle *handle);

/**
 * @brief: map ge type to tf
 * @param ge_type: ge type
 * @param acl_type: tf type
 */
tensorflow::Status MapGeType2Tf(ge::DataType ge_type, tensorflow::DataType &tf_type);

/**
 * @brief: map tf type to ge
 * @param ge_type: tf type
 * @param acl_type: ge type
 */
tensorflow::Status MapTfType2Ge(tensorflow::DataType tf_type, ge::DataType &ge_type);

/**
 * @brief: map ge type to acl
 * @param ge_type: ge type
 * @param acl_type: acl type
 */
tensorflow::Status MapGeType2Acl(ge::DataType ge_type, aclDataType &acl_type);

/**
 * @brief: map ge format to acl
 * @param ge_format: ge format
 * @param acl_format: acl format
 */
tensorflow::Status MapGeFormat2Acl(ge::Format ge_format, aclFormat &acl_format);

// TODO:在GE处理中，变量名称作为唯一标识，对于shared_name是"_"开头的变量，由于tensorflow禁止变量名以"_"开头，所以无法直接将shared_name
//  作为Node的name，对于GE，则没有这个限制，因而，这个函数需要能够屏蔽这种差异。
std::string WrapResourceName(const std::string &name);

/**
 * @brief: load GraphDef proto
 * @param file: proto file path
 * @param def: point to save GraphDef
 */
tensorflow::Status LoadGraphDefProto(const std::string &file, tensorflow::GraphDef *def);

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

struct ResourceCompare {
  bool operator()(const tensorflow::ResourceHandle &left, const tensorflow::ResourceHandle &right) const {
    return left.name() < right.name() || left.container() < right.container() || left.device() < right.device();
  }
};
}  // namespace npu

#endif  // TENSORFLOW_NPU_UTILS_H
