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

#include "npu_logger.h"
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

#include "npu_micros.h"
#include "npu_managed_buffer.h"
#include "npu_unwrap.h"
#include "npu_logger.h"
#include "npu_device.h"
#include "npu_utils.h"

namespace npu {
class ProfManager {
 public:
  static void RecordOp(const std::string &op, std::string detail, bool is_stateful, bool is_unknown) {
    Instance().RecordOpInner(op, detail, is_stateful, is_unknown);
  }

 private:
  static ProfManager &Instance() {
    static ProfManager prof;
    return prof;
  }
  void RecordOpInner(const std::string &op, std::string detail, bool is_stateful, bool is_unknown) {
    std::lock_guard<std::mutex> lk(mu_);
    op_records_[op]++;
    if (is_unknown) {
      unknown_shape_op_records_[op]++;
    }
    if (is_stateful) {
      stateful_shape_op_records_[op]++;
    }
    op_shape_records_[op].insert(detail);
  }
  ~ProfManager() {
    std::lock_guard<std::mutex> lk(mu_);
    LOG(INFO) << "All nodes executed by acl";
    for (auto iter = op_records_.begin(); iter != op_records_.end(); iter++) {
      LOG(INFO) << iter->first << ":" << iter->second;
    }

    LOG(INFO) << "All stateful nodes executed by acl";
    for (auto iter = stateful_shape_op_records_.begin(); iter != stateful_shape_op_records_.end(); iter++) {
      LOG(INFO) << iter->first << ":" << iter->second;
    }

    LOG(INFO) << "All unknown shape nodes executed by acl";
    for (auto iter = unknown_shape_op_records_.begin(); iter != unknown_shape_op_records_.end(); iter++) {
      LOG(INFO) << iter->first << ":" << iter->second;
    }

    LOG(INFO) << "All nodes' shape and type detail executed by acl";
    for (auto iter = op_shape_records_.begin(); iter != op_shape_records_.end(); iter++) {
      std::stringstream ss;
      ss << std::endl << iter->first << ":";
      for (auto status : iter->second) {
        ss << std::endl << status;
      }
      LOG(INFO) << ss.str();
    }
  }
  ProfManager() = default;
  std::mutex mu_;
  std::map<std::string, size_t> op_records_ GUARDED_BY(mu_);
  std::map<std::string, size_t> unknown_shape_op_records_ GUARDED_BY(mu_);
  std::map<std::string, size_t> stateful_shape_op_records_ GUARDED_BY(mu_);
  std::map<std::string, std::set<std::string>> op_shape_records_ GUARDED_BY(mu_);
};
}  // namespace npu