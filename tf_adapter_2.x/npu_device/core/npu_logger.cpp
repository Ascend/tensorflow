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

namespace npu {
class ProfManager {
 public:
  static void RecordOp(const std::string &op, const std::string &detail, bool is_stateful, bool is_unknown) {
    Instance().RecordOpInner(op, detail, is_stateful, is_unknown);
  }

 private:
  static ProfManager &Instance() {
    static ProfManager prof;
    return prof;
  }
  void RecordOpInner(const std::string &op, const std::string &detail, bool is_stateful, bool is_unknown) {
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
    for (auto iter = op_records_.cbegin(); iter != op_records_.cend(); iter++) {
      LOG(INFO) << iter->first << ":" << iter->second;
    }

    LOG(INFO) << "All stateful nodes executed by acl";
    for (auto iter = stateful_shape_op_records_.cbegin(); iter != stateful_shape_op_records_.cend(); iter++) {
      LOG(INFO) << iter->first << ":" << iter->second;
    }

    LOG(INFO) << "All unknown shape nodes executed by acl";
    for (auto iter = unknown_shape_op_records_.cbegin(); iter != unknown_shape_op_records_.cend(); iter++) {
      LOG(INFO) << iter->first << ":" << iter->second;
    }

    LOG(INFO) << "All nodes' shape and type detail executed by acl";
    for (auto iter = op_shape_records_.cbegin(); iter != op_shape_records_.cend(); iter++) {
      std::stringstream ss;
      ss << std::endl << iter->first << ":";
      for (const auto status : iter->second) {
        ss << std::endl << status;
      }
      LOG(INFO) << ss.str();
    }
  }
  ProfManager() = default;
  std::mutex mu_;
  std::map<std::string, size_t> op_records_ TF_GUARDED_BY(mu_);
  std::map<std::string, size_t> unknown_shape_op_records_ TF_GUARDED_BY(mu_);
  std::map<std::string, size_t> stateful_shape_op_records_ TF_GUARDED_BY(mu_);
  std::map<std::string, std::set<std::string>> op_shape_records_ TF_GUARDED_BY(mu_);
};
}  // namespace npu