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

#ifndef TENSORFLOW_NPU_ATTRS_H_
#define TENSORFLOW_NPU_ATTRS_H_

#include <map>
#include <string>
#include "ge/ge_api_types.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/env_var.h"

// single load all npu mode
namespace tensorflow {
std::string GetDumpPath();
Status GetEnvDeviceID(uint32_t &device_id);
void Split(const std::string &s, std::vector<std::string> &result, const char *delchar = " ");
bool GetNewDataTransferFlag();
extern bool kIsNewDataTransfer;
extern const bool kDumpGraph;
extern const bool kIsHeterogeneous;

class NpuAttrs {
 public:
  // This method returns instance Pointers
  static std::map<std::string, std::string> GetInitOptions();
  static std::map<std::string, std::string> GetInitOptions(const OpKernelConstruction *ctx);
  static std::map<std::string, std::string> GetDefaultInitOptions();
  static std::map<std::string, std::string> GetSessOptions(const OpKernelConstruction *ctx);
  static std::map<std::string, std::string> GetPassOptions(const GraphOptimizationPassOptions &options);
  static std::map<std::string, std::string> GetPassOptions(const OpKernelConstruction *ctx);
  static std::map<std::string, std::string> GetPassOptions(const AttrSlice &attrs);
  static std::map<std::string, std::string> GetAllAttrOptions(const AttrSlice &attrs);
  static std::map<std::string, std::string> GetDefaultPassOptions();
  static Status SetNpuOptimizerAttr(const GraphOptimizationPassOptions &options, Node *node);
  static void LogOptions(const std::map<std::string, std::string> &options);
  static void SetUseTdtStatus(int32_t device_id, bool is_turn_on_tdt);
  static bool GetUseTdtStatus(int32_t device_id);
  static bool GetUseAdpStatus(const std::string &iterator_name);
  static void SetUseAdpStatus(const std::string &iterator_name, bool is_use_adp);
  static bool IsDatasetExecuteInDevice(const std::string &iterator_name);
  static void SetDatasetExecuteInDeviceStatus(const std::string &iterator_name, bool is_dataset_execute_device);

 private:
  static std::map<int32_t, bool> turn_on_tdt_info_;
  static std::map<std::string, bool> use_adp_info_;
  static std::map<std::string, bool> dataset_execute_info_;
  static std::map<std::string, std::string> init_options_;
};
}  // namespace tensorflow

#endif
