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

#include "tf_adapter/util/npu_ops_identifier.h"
#include <fstream>
#include <regex>

#include "nlohmann/json.hpp"
#include "framework/common/string_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/generate_report.h"
#include "tf_adapter/util/npu_attrs.h"
#include "mmpa/mmpa_api.h"

using json = nlohmann::json;

const static std::string kOpsInfoJsonV01 = "/framework/built-in/tensorflow/npu_supported_ops.json";
const static std::string kOpsInfoJsonV02 = "/built-in/framework/tensorflow/npu_supported_ops.json";
const static std::string kCustomOpsInfoJsonV01 = "/framework/custom/tensorflow/npu_supported_ops.json";
const static std::string kCustomOpsInfoJsonV02 = "/vendors/%s/framework/tensorflow/npu_supported_ops.json";
const size_t kVendorConfigPartsCount = 2U;
const static std::string kGray = "isGray";
const static std::string kHeavy = "isHeavy";

NpuOpsIdentifier *NpuOpsIdentifier::GetInstance(bool is_mix) {
  if (is_mix) {
    static json mixJson;
    static NpuOpsIdentifier instance(true, mixJson);
    return &instance;
  } else {
    static json allJson;
    static NpuOpsIdentifier instance(false, allJson);
    return &instance;
  }
}

bool NpuOpsIdentifier::GetOppPluginVendors(const std::string &vendors_config, std::vector<std::string> &vendors) {
  ADP_LOG(INFO) << "Enter get opp plugin config file schedule, config file is: " << vendors_config;
  std::ifstream config(vendors_config);
  if (!config.good()) {
    ADP_LOG(INFO) << "Can not open file: " << vendors_config;
    return false;
  }
  std::string content;
  std::getline(config, content);
  config.close();
  if (content.empty()) {
    ADP_LOG(ERROR) << "Content of file '" << vendors_config << "' is empty!";
    return false;
  }
  std::vector<std::string> v_parts = ge::StringUtils::Split(content, '=');
  if (v_parts.size() != kVendorConfigPartsCount) {
    ADP_LOG(ERROR) << "Format of file content is invalid!";
    return false;
  }
  vendors = ge::StringUtils::Split(v_parts[1], ',');
  if (vendors.empty()) {
    ADP_LOG(ERROR) << "Format of file content is invalid!";
    return false;
  }
  return true;
}

bool NpuOpsIdentifier::IsNewOppPathStruct(const std::string &opp_path) {
  return mmIsDir((opp_path + "/built-in").c_str()) == EN_OK;
}

bool NpuOpsIdentifier::GetCustomOpPath(const std::string &ops_path, std::string &ops_json_path,
                                       std::vector<std::string> &custom_ops_json_path_vec) {
  if (!IsNewOppPathStruct(ops_path)) {
    ops_json_path = ops_path + kOpsInfoJsonV01;
    custom_ops_json_path_vec.push_back(ops_path + kCustomOpsInfoJsonV01);
    return true;
  }
  ops_json_path = ops_path + kOpsInfoJsonV02;
  std::vector<std::string> vendors;
  if (!GetOppPluginVendors(ops_path + "/vendors/config.ini", vendors)) {
    ADP_LOG(INFO) << "Can not get opp plugin vendors!";
    return false;
  }
  for (const auto &vendor : vendors) {
    custom_ops_json_path_vec.push_back(ops_path + std::regex_replace(kCustomOpsInfoJsonV02, std::regex("%s"), vendor));
  }
  return true;
}

// Constructor
NpuOpsIdentifier::NpuOpsIdentifier(bool is_mix, json &ops_info) : is_mix_(is_mix), ops_info_(ops_info) {
  const std::string mode = is_mix ? "MIX" : "ALL";
  const char *path_env = std::getenv("ASCEND_OPP_PATH");
  std::string ops_path;
  if (path_env != nullptr && strlen(path_env) < npu::ADAPTER_ENV_MAX_LENTH) {
    ops_path = path_env;
  } else {
    ops_path = "/usr/local/Ascend/opp";
    ADP_LOG(INFO) << "environment variable ASCEND_OPP_PATH is not set, use default value[" << ops_path << "]";
  }
  std::string ops_json_path;
  std::vector<std::string> custom_ops_json_path_vec;
  if (!GetCustomOpPath(ops_path, ops_json_path, custom_ops_json_path_vec)) {
    ADP_LOG(WARNING) << "Failed to get custom ops path!";
    return;
  }
  int32_t ops_cnt = 0;
  int32_t custom_ops_cnt = 0;
  ADP_LOG(INFO) << "[" << mode << "] Parsing json from " << ops_json_path;
  ops_cnt = NpuOpsIdentifier::ParseOps(ops_json_path, ops_info_);
  for (const auto &custom_ops_json_path : custom_ops_json_path_vec) {
    ADP_LOG(INFO) << "[" << mode << "] Parsing json from " << custom_ops_json_path;
    json custom_ops_info;
    custom_ops_cnt += NpuOpsIdentifier::ParseOps(custom_ops_json_path, custom_ops_info);
    for (const auto elem : custom_ops_info.items()) {
      ops_info_[elem.key()] = elem.value();
    }
  }
  ADP_LOG(INFO) << ops_cnt << " ops parsed";
  ADP_LOG(INFO) << custom_ops_cnt << " custom ops parsed";
  ADP_LOG(INFO) << ops_info_.dump(2);  // 1 is vlog level, 2 is ops info index
}
/**
 * @brief: Parse and store the ops configuration json file, return num of parsed ops
 * @param f: npu supported json file path
 * @param root: json root
 */
int32_t NpuOpsIdentifier::ParseOps(const std::string &f, json &root) const {
  std::ifstream jsonConfigFileStream(f, std::ifstream::in);
  int32_t opsCnt = 0;
  if (jsonConfigFileStream.is_open()) {
    try {
      jsonConfigFileStream >> root;
      for (auto i = root.begin(); i != root.end(); ++i) {
        opsCnt++;
      }
    } catch (json::exception &e) {
      ADP_LOG(INFO) << e.what();
      jsonConfigFileStream.close();
      return 0;
    }
    jsonConfigFileStream.close();
  } else {
    ADP_LOG(INFO) << "Open " << f << ", ret is not true.";
    return 0;
  }
  return opsCnt;
}
// Determine if the node is supported by NPU. Note that it will behave
// differently in mixed mode and full sink mode
bool NpuOpsIdentifier::IsNpuSupported(const char *op_name, const std::string &node_name) {
  return NpuOpsIdentifier::IsNpuSupported(std::string(op_name), node_name);
}

bool NpuOpsIdentifier::IsNpuSupported(const std::string &op_name, const std::string &node_name) {
  bool declared = ops_info_[op_name].is_object();
  if (!declared) {
    tensorflow::GenerateReport::Details infos;
    static const std::string message = "This op is not exsit on npu.";
    infos.code = static_cast<int>(tensorflow::GenerateReport::ReasonCode::TypeNoDefine);
    infos.message = message;
    (void)tensorflow::GenerateReport::GetInstance()->AddUnSupportedInfo(node_name, op_name, infos);
    return false;
  }
  if (is_mix_ && ops_info_[op_name][kGray].is_boolean()) {
    return !ops_info_[op_name][kGray];
  }
  return true;
}
// Determine if the node is performance-sensitive on NPU, this should
// normally be done after calling IsNpuSupported to confirm that the node
// is supported by NPU. To be on the safe side, it internally performs a
// check on whether it is supported by NPU, if not, prints an error log,
// and returns `false`
bool NpuOpsIdentifier::IsPerformanceSensitive(const char *op) {
  return NpuOpsIdentifier::IsPerformanceSensitive(std::string(op));
}
/**
 * @brief: is performance sensitive
 * @param op: op type
 */
bool NpuOpsIdentifier::IsPerformanceSensitive(const std::string &op) {
  if (ops_info_.find(op) != ops_info_.end()) {
    if (ops_info_[op].is_object()) {
      if (ops_info_[op][kHeavy].is_boolean()) {
        return ops_info_[op][kHeavy];
      }
    }
  }
  return false;
}
