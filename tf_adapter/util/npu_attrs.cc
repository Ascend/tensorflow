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

#include "tf_adapter/util/npu_attrs.h"
#include <mutex>
#include <regex>
#include <iostream>
#include <sstream>
#include "securec.h"
#include "acl/acl_tdt.h"
#include "acl/acl.h"
#include "tdt/index_transform.h"
#include "runtime/config.h"
#include "tf_adapter/common/adp_logger.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/env_var.h"
#include "mmpa/mmpa_api.h"
#include "tf_adapter/util/ge_plugin.h"
namespace tensorflow {
std::map<int32_t, bool> NpuAttrs::turn_on_tdt_info_;
std::map<std::string, bool> NpuAttrs::use_adp_info_;
std::map<std::string, bool> NpuAttrs::dataset_execute_info_;
std::map<std::string, std::string> NpuAttrs::init_options_;
const static int32_t kRuntimeTypeHeterogeneous = 1;
bool kIsNewDataTransfer = true;
bool GetNewDataTransferFlag() {
  uint32_t device_id = 0U;
  (void) GetEnvDeviceID(device_id);
  auto acl_status = aclrtSetDevice(device_id);
  acltdtChannelHandle *check_queue_handle = acltdtCreateChannelWithCapacity(device_id, "check_is_queue", 3UL);
  if (check_queue_handle != nullptr) {
    acltdtDestroyChannel(check_queue_handle);
    return true;
  }
  check_queue_handle = acltdtCreateChannel(device_id, "check_is_queue");
  if (check_queue_handle !=nullptr) {
    acltdtDestroyChannel(check_queue_handle);
    return false;
  } else {
    ADP_LOG(ERROR) << "Create channel failed by acltdtCreateChannelWithCapacity and acltdtCreateChannel";
  }
  return true;
};

extern const bool kDumpGraph = []() -> bool {
  bool print_model = false;
  tensorflow::ReadBoolFromEnvVar("PRINT_MODEL", false, &print_model);
  return print_model;
}();

extern const bool kIsHeterogeneous = []() -> bool {
  int32_t is_heterogeneous = 0;
  (void) rtGetIsHeterogenous(&is_heterogeneous);
  return is_heterogeneous == kRuntimeTypeHeterogeneous;
}();

std::string GetDumpPath() {
  std::string npu_collect_path;
  (void) ReadStringFromEnvVar("NPU_COLLECT_PATH", "", &npu_collect_path);
  if (!npu_collect_path.empty()) {
    std::string collect_path_str(npu_collect_path);
    collect_path_str.erase(0, collect_path_str.find_first_not_of(" "));
    collect_path_str.erase(collect_path_str.find_last_not_of(" ") + 1);
    std::string base_path_str = collect_path_str.empty() ? "./" : collect_path_str + "/";

    uint32_t device_id = 0;
    GetEnvDeviceID(device_id);
    base_path_str += "/extra-info/graph/" + std::to_string(mmGetPid()) + "_" + std::to_string(device_id) + "/";
    if (mmAccess2(base_path_str.c_str(), M_F_OK) != EN_OK) {
      int32_t ret = mmMkdir(base_path_str.c_str(), M_IRUSR | M_IWUSR | M_IXUSR);
      if (ret != 0) {
        ADP_LOG(WARNING) << "create dump graph dir failed, path:" << base_path_str;
        return "./";
      }
    }
    return base_path_str;
  }

  std::string dump_graph_path;
  (void) ReadStringFromEnvVar("DUMP_GRAPH_PATH", "./", &dump_graph_path);
  dump_graph_path.erase(0, dump_graph_path.find_first_not_of(" "));
  dump_graph_path.erase(dump_graph_path.find_last_not_of(" ") + 1);

  std::string base_path = dump_graph_path.empty() ? "./" : dump_graph_path + "/";
  if (mmAccess2(base_path.c_str(), M_F_OK) != EN_OK) {
    if (mmMkdir(base_path.c_str(), M_IRUSR | M_IWUSR | M_IXUSR) != 0) {
      ADP_LOG(WARNING) << "create dump graph dir failed, path:" << base_path;
      return "./";
    }
  }
  return base_path;
}

Status GetEnvDeviceID(uint32_t &device_id) {
  int64 phy_device_id = -1;
  int64 logic_device_id = -1;
  std::string env_ascend_device_id;
  (void) ReadStringFromEnvVar("ASCEND_DEVICE_ID", "", &env_ascend_device_id);
  std::string env_device_id;
  (void) ReadStringFromEnvVar("DEVICE_ID", "", &env_device_id);
  if (env_ascend_device_id.empty() && env_device_id.empty()) {
    ADP_LOG(WARNING) << "[GePlugin] DEVICE_ID and ASCEND_DEVICE_ID is none, use default device id : 0, if set "
                        "session_device_id, session_device_id has a higher priority";
    LOG(WARNING) << "[GePlugin] DEVICE_ID and ASCEND_DEVICE_ID is none, use default device id : 0, if set "
                    "session_device_id, session_device_id has a higher priority";
  } else if (!env_ascend_device_id.empty()) {
    if (!strings::safe_strto64(env_ascend_device_id, &logic_device_id)) {
      return errors::InvalidArgument("ASCEND_DEVICE_ID is valid, not digit.");
    }
    if (logic_device_id < 0) {
      return errors::InvalidArgument("ASCEND_DEVICE_ID should be >= 0.");
    }
    device_id = static_cast<uint32_t>(logic_device_id);
  } else {
    if (!strings::safe_strto64(env_device_id, &phy_device_id)) {
      return errors::InvalidArgument("DEVICE_ID is valid, not digit.");
    }
    if (phy_device_id < 0) {
      return errors::InvalidArgument("DEVICE_ID should be >= 0.");
    }
    if (IndexTransform(static_cast<uint32_t>(phy_device_id), device_id) != 0) {
      return errors::InvalidArgument("get logic device id by DEVICE_ID failed.");
    }
  }
  return Status::OK();
}
void Split(const std::string &s, std::vector<std::string> &result, const char *delchar) {
  if (s.empty()) {
    return;
  }
  result.clear();

  std::stringstream ss(s);
  std::string item;
  while (getline(ss, item, *delchar)) {
    result.push_back(item);
  }
  return;
}

inline Status checkDumpStep(const std::string &dump_step) {
  std::string tmp_dump_step = dump_step + "|";
  std::smatch result;
  std::vector<string> match_vecs;
  std::regex pattern(R"((\d{1,}-\d{1,}\||\d{1,}\|)+)");
  if (regex_match(tmp_dump_step, result, pattern)) {
    Split(result.str(), match_vecs, "|");
    // 100 is the max sets of dump steps.
    if (match_vecs.size() > 100) {
      return errors::InvalidArgument("dump_step only support dump <= 100 sets of data");
    }
    for (const auto &match_vec : match_vecs) {
      std::vector<string> tmp_vecs;
      Split(match_vec, tmp_vecs, "-");
      if (tmp_vecs.size() > 1) {
        if (std::atoi(tmp_vecs[0].c_str()) >= std::atoi(tmp_vecs[1].c_str())) {
          return errors::InvalidArgument("in range steps, the first step is >= "
                                         "second step, correct example:'0|5|10-20'");
        }
      }
    }
  } else {
    return errors::InvalidArgument("dump_step string style is error,"
                                   " correct example:'0|5|10|50-100'");
  }
  return Status::OK();
}

inline Status checkDumpMode(const std::string &dump_mode) {
  std::set<string> dump_mode_list = {"input", "output", "all"};
  if (dump_mode_list.find(dump_mode) != dump_mode_list.cend()) {
    return Status::OK();
  } else {
    return errors::InvalidArgument("dump mode should be one of the list:[input, output, all]");
  }
}

inline Status checkDumpDebugMode(const std::string &dump_debug_mode) {
  std::set<string> dump_debug_mode_list = {"aicore_overflow", "atomic_overflow", "all"};
  if (dump_debug_mode_list.find(dump_debug_mode) != dump_debug_mode_list.cend()) {
    return Status::OK();
  } else {
    return errors::InvalidArgument("dump debug mode should be one of the list:[aicore_overflow, atomic_overflow, all]");
  }
}

inline Status CheckPath(const std::string &input, std::string &output) {
  if (mmIsDir(input.c_str()) != EN_OK) {
    return errors::InvalidArgument("the path ", input.c_str(), " is not directory.");
  }
  char trusted_path[MMPA_MAX_PATH] = {"\0"};
  if (mmRealPath(input.c_str(), trusted_path, MMPA_MAX_PATH) != EN_OK) {
    return errors::InvalidArgument("the path ", input.c_str(), " is invalid.");
  }
  if (mmAccess2(trusted_path, R_OK | W_OK) != EN_OK) {
    return errors::InvalidArgument("the path ", input.c_str(), " does't have read, write permissions.");
  }
  output = trusted_path;
  return Status::OK();
}

inline Status CheckOpImplMode(const std::string &op_select_implmode) {
  std::set<string> op_impl_mode_list = {"high_precision", "high_performance", "high_precision_for_all",
                                        "high_performance_for_all"};

  if (op_impl_mode_list.find(op_select_implmode) != op_impl_mode_list.end()) {
    return Status::OK();
  } else {
    return errors::InvalidArgument("op select impl mode should be one of the list:[high_precision, "
                                   "high_performance, high_precision_for_all, high_performance_for_all]");
  }
}

inline Status CheckAoeMode(const std::string &aoe_mode) {
  std::set<string> aoe_mode_list = {"1", "2", "3", "4"};

  if (aoe_mode_list.find(aoe_mode) != aoe_mode_list.end()) {
    return Status::OK();
  } else {
    return errors::InvalidArgument("aoe mode should be one of the list:['1', '2', '3', '4']");
  }
}

inline Status CheckInputShape(const std::string &input_shape) {
  std::vector<std::string> inputs;
  Split(input_shape, inputs, ";");
  if (inputs.empty()) {
    return errors::InvalidArgument("input_shape is empty.");
  }
  for (auto input : inputs) {
    std::string input_tmp = input + ",";
    std::regex pattern(R"(\w{1,}:((\d{1,}|-\d{1,}),)+)");
    if (!regex_match(input_tmp, pattern)) {
      return errors::InvalidArgument("input_shape string style is invalid");
    }
  }
  return Status::OK();
}

inline Status CheckDynamicDims(const std::string &dynamic_dims) {
  std::vector<std::string> inputs;
  Split(dynamic_dims, inputs, ";");
  if (inputs.empty()) {
    return errors::InvalidArgument("dynamic_dims is empty.");
  }
  for (auto input : inputs) {
    std::string input_tmp = input + ",";
    std::regex pattern(R"((\d{1,},)+)");
    if (!regex_match(input_tmp, pattern)) {
      return errors::InvalidArgument("dynamic_dims string style is invalid");
    }
  }
  return Status::OK();
}

inline Status CheckLocalRankId(int local_rank_id) {
  int kMaxDeviceId = 7;
  if (local_rank_id < 0 || local_rank_id > kMaxDeviceId) {
    return errors::InvalidArgument("local rank id should be in [0,7]");
  }
  return Status::OK();
}

inline Status CheckDeviceList(const std::string &local_device_list) {
  std::string tmp_device_list = local_device_list + ",";
  std::regex pattern("(\\d{1,},)+");
  if (!regex_match(tmp_device_list, pattern)) {
    return errors::InvalidArgument("local_device_list style is invalid, example:'1,2,3'");
  }
  return Status::OK();
}

bool NpuAttrs::GetUseTdtStatus(int32_t device_id) {
  if (turn_on_tdt_info_.count(device_id) > 0) {
    ADP_LOG(INFO) << "get device: " << device_id << " turn_on_tdt_info_: " << turn_on_tdt_info_[device_id];
    return turn_on_tdt_info_[device_id];
  } else {
    return false;
  }
}

void NpuAttrs::SetUseTdtStatus(int32_t device_id, bool is_turn_on_tdt) {
  turn_on_tdt_info_[device_id] = is_turn_on_tdt;
  ADP_LOG(INFO) << "set device: " << device_id << " turn_on_tdt_info_: " << turn_on_tdt_info_[device_id];
}

bool NpuAttrs::GetUseAdpStatus(std::string iterator_name) {
  if (use_adp_info_.count(iterator_name) > 0) {
    ADP_LOG(INFO) << "get iterator: " << iterator_name << " use_adp_info_: " << use_adp_info_[iterator_name];
    return use_adp_info_[iterator_name];
  } else {
    return false;
  }
}

void NpuAttrs::SetUseAdpStatus(std::string iterator_name, bool is_use_adp) {
  use_adp_info_[iterator_name] = is_use_adp;
  ADP_LOG(INFO) << "set iterator: " << iterator_name << " use_adp_info_: " << use_adp_info_[iterator_name];
}

bool NpuAttrs::IsDatasetExecuteInDevice(std::string iterator_name) {
  if (dataset_execute_info_.count(iterator_name) > 0) {
    ADP_LOG(INFO) << "get data pre-process graph: " << iterator_name
                  << " dataset_execute_info_: " << dataset_execute_info_[iterator_name];
    return dataset_execute_info_[iterator_name];
  } else {
    return false;
  }
}

void NpuAttrs::SetDatasetExecuteInDeviceStatus(std::string iterator_name, bool is_dataset_execute_device) {
  dataset_execute_info_[iterator_name] = is_dataset_execute_device;
  ADP_LOG(INFO) << "data pre-process graph: " << iterator_name
                << " dataset_execute_info_: " << dataset_execute_info_[iterator_name];
}

std::map<std::string, std::string> NpuAttrs::GetSessOptions(OpKernelConstruction *ctx) {
  std::map<std::string, std::string> sess_options;
  std::string variable_format_optimize = std::to_string(true);
  std::string hcom_parallel = std::to_string(false);
  std::string graph_memory_max_size;
  std::string variable_memory_max_size;
  std::string enable_dump = std::to_string(false);
  std::string enable_dump_debug = std::to_string(false);
  std::string dump_path;
  std::string dump_step;
  std::string dump_mode = "output";
  std::string dump_debug_mode = "all";
  std::string stream_max_parallel_num;
  std::string npuOptimizer;
  std::string is_tailing_optimization = std::to_string(false);
  std::string op_select_implmode;
  std::string optypelist_for_implmode;
  std::string buffer_optimize = "l2_optimize";
  std::string enable_small_channel = "0";
  std::string fusion_switch_file;
  std::string enable_compress_weight = std::to_string(false);
  std::string compress_weight_conf;
  std::string input_shape;
  std::string dynamic_dims;
  std::string dynamic_node_type;
  std::string session_device_id;
  std::string modify_mixlist;
  std::string op_precision_mode;
  std::string graph_run_mode = "1";
  std::string hccl_timeout;

  if (ctx != nullptr && ctx->GetAttr("_NpuOptimizer", &npuOptimizer) == Status::OK()) {
    ctx->GetAttr("_variable_format_optimize", &variable_format_optimize);
    ctx->GetAttr("_hcom_parallel", &hcom_parallel);
    ctx->GetAttr("_graph_memory_max_size", &graph_memory_max_size);
    ctx->GetAttr("_variable_memory_max_size", &variable_memory_max_size);
    ctx->GetAttr("_enable_dump", &enable_dump);
    ctx->GetAttr("_enable_dump_debug", &enable_dump_debug);
    if (enable_dump != std::to_string(false) || enable_dump_debug != std::to_string(false)) {
      ctx->GetAttr("_dump_path", &dump_path);
    }
    if (enable_dump != std::to_string(false)) {
      if (ctx->GetAttr("_dump_step", &dump_step) == Status::OK() && !dump_step.empty()) {
        Status s = checkDumpStep(dump_step);
        if (!s.ok()) {
          ADP_LOG(FATAL) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
      }
      if (ctx->GetAttr("_dump_mode", &dump_mode) == Status::OK()) {
        Status s = checkDumpMode(dump_mode);
        if (!s.ok()) {
          ADP_LOG(FATAL) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
      }
    }
    if (enable_dump_debug != std::to_string(false)) {
      if (ctx->GetAttr("_dump_debug_mode", &dump_debug_mode) == Status::OK()) {
        Status s = checkDumpDebugMode(dump_debug_mode);
        if (!s.ok()) {
          ADP_LOG(FATAL) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
      }
    }
    ctx->GetAttr("_stream_max_parallel_num", &stream_max_parallel_num);
    ctx->GetAttr("_is_tailing_optimization", &is_tailing_optimization);
    ctx->GetAttr("_op_select_implmode", &op_select_implmode);
    ctx->GetAttr("_optypelist_for_implmode", &optypelist_for_implmode);
    ctx->GetAttr("_input_shape", &input_shape);
    ctx->GetAttr("_dynamic_dims", &dynamic_dims);
    ctx->GetAttr("_buffer_optimize", &buffer_optimize);
    ctx->GetAttr("_enable_small_channel", &enable_small_channel);
    ctx->GetAttr("_fusion_switch_file", &fusion_switch_file);
    ctx->GetAttr("_enable_compress_weight", &enable_compress_weight);
    ctx->GetAttr("_compress_weight_conf", &compress_weight_conf);
    ctx->GetAttr("_dynamic_node_type", &dynamic_node_type);
    ctx->GetAttr("_session_device_id", &session_device_id);
    ctx->GetAttr("_modify_mixlist", &modify_mixlist);
    ctx->GetAttr("_op_precision_mode", &op_precision_mode);
    ctx->GetAttr("_graph_run_mode", &graph_run_mode);
    ctx->GetAttr("_hccl_timeout", &hccl_timeout);
  }

  // session options
  sess_options["ge.exec.variable_acc"] = variable_format_optimize;
  sess_options[ge::HCOM_PARALLEL] = hcom_parallel;
  sess_options[ge::STREAM_MAX_PARALLEL_NUM] = stream_max_parallel_num;
  if (!graph_memory_max_size.empty()) {
    sess_options[ge::GRAPH_MEMORY_MAX_SIZE] = graph_memory_max_size;
  }
  if (!variable_memory_max_size.empty()) {
    sess_options[ge::VARIABLE_MEMORY_MAX_SIZE] = variable_memory_max_size;
  }
  sess_options[ge::OPTION_EXEC_ENABLE_DUMP] = enable_dump;
  sess_options[ge::OPTION_EXEC_DUMP_PATH] = dump_path;
  sess_options[ge::OPTION_EXEC_DUMP_STEP] = dump_step;
  sess_options[ge::OPTION_EXEC_DUMP_MODE] = dump_mode;
  sess_options[ge::OPTION_EXEC_ENABLE_DUMP_DEBUG] = enable_dump_debug;
  sess_options[ge::OPTION_EXEC_DUMP_DEBUG_MODE] = dump_debug_mode;
  sess_options["ge.exec.isTailingOptimization"] = is_tailing_optimization;
  sess_options[ge::OP_SELECT_IMPL_MODE] = op_select_implmode;
  sess_options[ge::OPTYPELIST_FOR_IMPLMODE] = optypelist_for_implmode;
  sess_options["ge.inputShape"] = input_shape;
  sess_options["ge.dynamicDims"] = dynamic_dims;
  sess_options["ge.bufferOptimize"] = buffer_optimize;
  sess_options["ge.enableSmallChannel"] = enable_small_channel;
  sess_options["ge.fusionSwitchFile"] = fusion_switch_file;
  sess_options["ge.enableCompressWeight"] = enable_compress_weight;
  sess_options["compress_weight_conf"] = compress_weight_conf;
  sess_options["ge.dynamicNodeType"] = dynamic_node_type;
  if (std::atoi(session_device_id.c_str()) >= 0) {
    sess_options["ge.session_device_id"] = session_device_id;
  }
  sess_options[ge::MODIFY_MIXLIST] = modify_mixlist;
  sess_options["ge.exec.op_precision_mode"] = op_precision_mode;
  sess_options[ge::OPTION_GRAPH_RUN_MODE] = graph_run_mode;
  sess_options["ge.exec.hcclExecuteTimeOut"] = hccl_timeout;

  return sess_options;
}

std::map<std::string, std::string> NpuAttrs::GetDefaultInitOptions() {
  std::map<std::string, std::string> init_options;
  init_options["ge.exec.precision_mode"] = "allow_fp32_to_fp16";
  init_options[ge::OPTION_EXEC_PROFILING_MODE] = std::to_string(false);
  init_options[ge::OPTION_EXEC_PROFILING_OPTIONS] = "";
  init_options[ge::AUTO_TUNE_MODE] = "";
  init_options[ge::OPTION_GRAPH_RUN_MODE] = "1";
  init_options[ge::OP_DEBUG_LEVEL] = "0";
  init_options[ge::OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES] = "";
  init_options[ge::OPTION_EXEC_PROFILING_FPPONIT_OPTIONS] = "";
  init_options[ge::OPTION_EXEC_PROFILING_BPPONIT_OPTIONS] = "";
  init_options["ge.jobType"] = "";
  init_options["ge.tuningPath"] = "";
  init_options["ge.deviceType"] = "default_device_type";
  return init_options;
}

std::map<std::string, std::string> NpuAttrs::GetInitOptions(OpKernelConstruction *ctx) {
  std::string precision_mode = "allow_fp32_to_fp16";
  std::string profiling_mode = std::to_string(false);
  std::string profiling_options;
  std::string auto_tune_mode;
  std::string graph_run_mode = "1";
  std::string op_debug_level = "0";
  std::string enable_scope_fusion_passes;
  std::string enable_exception_dump;
  std::string op_compiler_cache_mode;
  std::string op_compiler_cache_dir;
  std::string debug_dir;
  std::string hcom_multi_mode;
  std::string npuOptimizer;
  std::string aoe_mode;
  std::string work_path;
  std::string distribute_config;
  std::string modify_mixlist;
  std::string fusion_switch_file;
  std::string op_precision_mode;
  std::string op_select_implmode;
  std::string optypelist_for_implmode;
  std::string device_type = "default_device_type";
  std::string soc_config;
  std::string hccl_timeout;
  std::string op_wait_timeout;
  std::string op_execute_timeout;

  if (ctx != nullptr && ctx->GetAttr("_NpuOptimizer", &npuOptimizer) == Status::OK()) {
    ctx->GetAttr("_precision_mode", &precision_mode);
    ctx->GetAttr("_auto_tune_mode", &auto_tune_mode);
    ctx->GetAttr("_graph_run_mode", &graph_run_mode);
    ctx->GetAttr("_op_debug_level", &op_debug_level);
    ctx->GetAttr("_enable_scope_fusion_passes", &enable_scope_fusion_passes);
    ctx->GetAttr("_enable_exception_dump", &enable_exception_dump);
    ctx->GetAttr("_aoe_mode", &aoe_mode);
    ctx->GetAttr("_work_path", &work_path);
    ctx->GetAttr("_op_compiler_cache_mode", &op_compiler_cache_mode);
    ctx->GetAttr("_op_compiler_cache_dir", &op_compiler_cache_dir);
    ctx->GetAttr("_debug_dir", &debug_dir);
    ctx->GetAttr("_hcom_multi_mode", &hcom_multi_mode);
    ctx->GetAttr("_distribute_config", &distribute_config);
    ctx->GetAttr("_modify_mixlist", &modify_mixlist);
    ctx->GetAttr("_fusion_switch_file", &fusion_switch_file);
    ctx->GetAttr("_op_precision_mode", &op_precision_mode);
    ctx->GetAttr("_op_select_implmode", &op_select_implmode);
    ctx->GetAttr("_optypelist_for_implmode", &optypelist_for_implmode);
    ctx->GetAttr("_device_type", &device_type);
    ctx->GetAttr("_soc_config", &soc_config);
    ctx->GetAttr("_hccl_timeout", &hccl_timeout);
    ctx->GetAttr("_op_wait_timeout", &op_wait_timeout);
    ctx->GetAttr("_op_execute_timeout", &op_execute_timeout);
  }

  if (precision_mode.empty()) {
    init_options_[ge::PRECISION_MODE] = "allow_fp32_to_fp16";
  } else {
    init_options_[ge::PRECISION_MODE] = precision_mode;
  }

  init_options_[ge::AUTO_TUNE_MODE] = auto_tune_mode;
  init_options_[ge::OPTION_GRAPH_RUN_MODE] = graph_run_mode;
  init_options_[ge::OP_DEBUG_LEVEL] = op_debug_level;
  init_options_[ge::OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES] = enable_scope_fusion_passes;
  init_options_["ge.exec.enable_exception_dump"] = enable_exception_dump;
  init_options_["ge.jobType"] = aoe_mode;
  init_options_["ge.tuningPath"] = work_path;
  init_options_["distribute_config"] = distribute_config;
  init_options_["ge.op_compiler_cache_mode"] = op_compiler_cache_mode;
  init_options_["ge.op_compiler_cache_dir"] = op_compiler_cache_dir;
  init_options_["ge.debugDir"] = debug_dir;
  init_options_["ge.hcomMultiMode"] = hcom_multi_mode;
  init_options_[ge::MODIFY_MIXLIST] = modify_mixlist;
  init_options_["ge.fusionSwitchFile"] = fusion_switch_file;
  init_options_[ge::OP_PRECISION_MODE] = op_precision_mode;
  init_options_[ge::OP_SELECT_IMPL_MODE] = op_select_implmode;
  init_options_[ge::OPTYPELIST_FOR_IMPLMODE] = optypelist_for_implmode;
  init_options_["ge.deviceType"] = device_type;
  init_options_["ge.exec.hcclExecuteTimeOut"] = hccl_timeout;
  init_options_["ge.exec.opWaitTimeout"] = op_wait_timeout;
  init_options_["ge.exec.opExecuteTimeout"] = op_execute_timeout;
  if (!soc_config.empty()) {
    init_options_["ge.socVersion"] = soc_config;
  }

  return init_options_;
}

std::map<std::string, std::string> NpuAttrs::GetInitOptions() {
  return init_options_;
}

std::map<std::string, std::string> NpuAttrs::GetPassOptions(const GraphOptimizationPassOptions &options) {
  std::map<std::string, std::string> pass_options;
  const RewriterConfig &rewrite_options = options.session_options->config.graph_options().rewrite_options();
  bool do_npu_optimizer = false;
  bool enable_dp = false;
  bool use_off_line = true;
  bool mix_compile_mode = false;
  int iterations_per_loop = 1;
  bool lower_functional_ops = false;
  std::string job = "default";
  int task_index = 0;
  bool dynamic_input = false;
  std::string dynamic_graph_execute_mode = "dynamic_execute";
  std::string dynamic_inputs_shape_range;
  int local_rank_id = -1;
  std::string local_device_list;
  bool in_out_pair_flag = true;
  std::string in_out_pair;

  for (const auto &custom_optimizer : rewrite_options.custom_optimizers()) {
    if (custom_optimizer.name() == "NpuOptimizer") {
      do_npu_optimizer = true;
      const auto &params = custom_optimizer.parameter_map();
      if (params.count("enable_data_pre_proc")) {
        enable_dp = params.at("enable_data_pre_proc").b();
      }
      if (params.count("use_off_line")) {
        use_off_line = params.at("use_off_line").b();
      }
      if (params.count("mix_compile_mode")) {
        mix_compile_mode = params.at("mix_compile_mode").b();
      }
      if (params.count("iterations_per_loop")) {
        iterations_per_loop = params.at("iterations_per_loop").i();
      }
      if (params.count("lower_functional_ops")) {
        lower_functional_ops = params.at("lower_functional_ops").b();
      }
      if (params.count("job")) {
        job = params.at("job").s();
      } else {
        job = "localhost";
      }
      if (params.count("task_index")) {
        task_index = params.at("task_index").i();
      }
      if (params.count("dynamic_input")) {
        dynamic_input = params.at("dynamic_input").b();
        if (dynamic_input) {
          if (params.count("dynamic_graph_execute_mode")) {
            dynamic_graph_execute_mode = params.at("dynamic_graph_execute_mode").s();
            if (dynamic_graph_execute_mode != "lazy_recompile" && dynamic_graph_execute_mode != "dynamic_execute") {
              ADP_LOG(ERROR) << "dynamic_graph_execute_mode should be lazy_recompile or dynamic_execute.";
              LOG(FATAL) << "dynamic_graph_execute_mode should be lazy_recompile or dynamic_execute.";
            }
          }
          if (params.count("dynamic_inputs_shape_range")) {
            dynamic_inputs_shape_range = params.at("dynamic_inputs_shape_range").s();
          }
        }
      }
      if (params.count("local_rank_id")) {
        local_rank_id = params.at("local_rank_id").i();
        Status s = CheckLocalRankId(local_rank_id);
        if (!s.ok()) {
          ADP_LOG(ERROR) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
      }
      if (params.count("local_device_list")) {
        local_device_list = params.at("local_device_list").s();
        Status s = CheckDeviceList(local_device_list);
        if (!s.ok()) {
          ADP_LOG(ERROR) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
      }
      if (params.count("in_out_pair_flag")) {
        in_out_pair_flag = params.at("in_out_pair_flag").b();
      }
      if (params.count("in_out_pair")) {
        in_out_pair = params.at("in_out_pair").s();
      }
    }
  }
  if (!do_npu_optimizer) {
    if ((const_cast<SessionOptions *>(options.session_options))->config.mutable_graph_options() != nullptr &&
        (const_cast<SessionOptions *>(options.session_options))
                ->config.mutable_graph_options()
                ->mutable_rewrite_options() != nullptr) {
      (const_cast<SessionOptions *>(options.session_options))
          ->config.mutable_graph_options()
          ->mutable_rewrite_options()
          ->set_remapping(RewriterConfig::OFF);
    }
  }
  // pass options
  pass_options["do_npu_optimizer"] = std::to_string(do_npu_optimizer);
  pass_options["enable_dp"] = std::to_string(enable_dp);
  pass_options["use_off_line"] = std::to_string(use_off_line);
  pass_options["mix_compile_mode"] = std::to_string(mix_compile_mode);
  pass_options["iterations_per_loop"] = std::to_string(iterations_per_loop);
  pass_options["lower_functional_ops"] = std::to_string(lower_functional_ops);
  pass_options["job"] = job;
  pass_options["task_index"] = std::to_string(task_index);
  pass_options["dynamic_input"] = std::to_string(dynamic_input);
  pass_options["dynamic_graph_execute_mode"] = dynamic_graph_execute_mode;
  pass_options["dynamic_inputs_shape_range"] = dynamic_inputs_shape_range;
  pass_options["local_rank_id"] = std::to_string(local_rank_id);
  pass_options["local_device_list"] = local_device_list;
  pass_options["in_out_pair_flag"] = std::to_string(in_out_pair_flag);
  pass_options["in_out_pair"] = in_out_pair;

  return pass_options;
}

std::map<std::string, std::string> NpuAttrs::GetPassOptions(OpKernelConstruction *ctx) {
  std::map<std::string, std::string> pass_options;
  std::string do_npu_optimizer = std::to_string(false);
  std::string enable_dp = std::to_string(false);
  std::string use_off_line = std::to_string(true);
  std::string mix_compile_mode = std::to_string(false);
  std::string iterations_per_loop = "1";
  std::string lower_functional_ops = std::to_string(false);
  std::string job = "default";
  std::string task_index = "0";
  std::string dynamic_input = std::to_string(false);
  std::string dynamic_graph_execute_mode = "dynamic_execute";
  std::string dynamic_inputs_shape_range;
  std::string local_rank_id = "-1";
  std::string local_device_list;
  std::string in_out_pair_flag = std::to_string(true);
  std::string in_out_pair;
  Status s = Status::OK();
  std::string npuOptimizer;

  if (ctx != nullptr && ctx->GetAttr("_NpuOptimizer", &npuOptimizer) == Status::OK()) {
    do_npu_optimizer = "1";
    ctx->GetAttr("_enable_data_pre_proc", &enable_dp);
    if (ctx->GetAttr("_use_off_line", &use_off_line) == Status::OK()) {
      ctx->GetAttr("_mix_compile_mode", &mix_compile_mode);
      ctx->GetAttr("_iterations_per_loop", &iterations_per_loop);
      ctx->GetAttr("_lower_functional_ops", &lower_functional_ops);
      if (ctx->GetAttr("_job", &job) != Status::OK()) {
        job = "localhost";
      }
      ctx->GetAttr("_task_index", &task_index);
      ctx->GetAttr("_dynamic_input", &dynamic_input);
      ctx->GetAttr("_dynamic_graph_execute_mode", &dynamic_graph_execute_mode);
      ctx->GetAttr("_dynamic_inputs_shape_range", &dynamic_inputs_shape_range);
      ctx->GetAttr("_local_rank_id", &local_rank_id);
      ctx->GetAttr("_local_device_list", &local_device_list);
    }
    ctx->GetAttr("_in_out_pair_flag", &in_out_pair_flag);
    ctx->GetAttr("_in_out_pair", &in_out_pair);
  }
  // pass options
  pass_options["do_npu_optimizer"] = do_npu_optimizer;
  pass_options["enable_dp"] = enable_dp;
  pass_options["use_off_line"] = use_off_line;
  pass_options["mix_compile_mode"] = mix_compile_mode;
  pass_options["iterations_per_loop"] = iterations_per_loop;
  pass_options["lower_functional_ops"] = lower_functional_ops;
  pass_options["job"] = job;
  pass_options["task_index"] = task_index;
  pass_options["dynamic_input"] = dynamic_input;
  pass_options["dynamic_graph_execute_mode"] = dynamic_graph_execute_mode;
  pass_options["dynamic_inputs_shape_range"] = dynamic_inputs_shape_range;
  pass_options["local_rank_id"] = local_rank_id;
  pass_options["local_device_list"] = local_device_list;
  pass_options["in_out_pair_flag"] = in_out_pair_flag;
  pass_options["in_out_pair"] = in_out_pair;

  return pass_options;
}

std::map<std::string, std::string> NpuAttrs::GetPassOptions(AttrSlice attrs) {
  std::map<std::string, std::string> pass_options;
  std::string do_npu_optimizer = std::to_string(false);
  std::string enable_dp = std::to_string(false);
  std::string use_off_line = std::to_string(true);
  std::string mix_compile_mode = std::to_string(false);
  std::string iterations_per_loop = "1";
  std::string lower_functional_ops = std::to_string(false);
  std::string job = "default";
  std::string task_index = "0";
  std::string dynamic_input = std::to_string(false);
  std::string dynamic_graph_execute_mode = "dynamic_execute";
  std::string dynamic_inputs_shape_range;
  std::string local_rank_id = "-1";
  std::string local_device_list;
  std::string in_out_pair_flag = std::to_string(true);
  std::string in_out_pair;
  Status s = Status::OK();

  auto NpuOptimizer_value = attrs.Find("_NpuOptimizer");
  auto enable_data_pre_proc_value = attrs.Find("_enable_data_pre_proc");
  auto use_off_line_value = attrs.Find("_use_off_line");
  auto mix_compile_mode_value = attrs.Find("_mix_compile_mode");
  auto iterations_per_loop_value = attrs.Find("_iterations_per_loop");
  auto lower_functional_ops_value = attrs.Find("_lower_functional_ops");
  auto job_value = attrs.Find("_job");
  auto task_index_value = attrs.Find("_task_index");
  auto dynamic_input_value = attrs.Find("_dynamic_input");
  auto dynamic_graph_execute_mode_value = attrs.Find("_dynamic_graph_execute_mode");
  auto dynamic_inputs_shape_range_value = attrs.Find("_dynamic_inputs_shape_range");
  auto local_rank_id_value = attrs.Find("_local_rank_id");
  auto local_device_list_value = attrs.Find("_local_device_list");
  auto in_out_pair_flag_value = attrs.Find("_in_out_pair_flag");
  auto in_out_pair_value = attrs.Find("_in_out_pair");

  if (NpuOptimizer_value != nullptr) {
    do_npu_optimizer = std::to_string(true);
    if (enable_data_pre_proc_value != nullptr) {
      enable_dp = enable_data_pre_proc_value->s();
    }
    if (use_off_line_value != nullptr) {
      use_off_line = use_off_line_value->s();
    }
    if (mix_compile_mode_value != nullptr) {
      mix_compile_mode = mix_compile_mode_value->s();
    }
    if (iterations_per_loop_value != nullptr) {
      iterations_per_loop = iterations_per_loop_value->s();
    }
    if (lower_functional_ops_value != nullptr) {
      lower_functional_ops = lower_functional_ops_value->s();
    }
    if (job_value != nullptr) {
      job = job_value->s();
    } else {
      job = "localhost";
    }
    if (task_index_value != nullptr) {
      task_index = task_index_value->s();
    }
    if (dynamic_input_value != nullptr) {
      dynamic_input = dynamic_input_value->s();
    }
    if (dynamic_graph_execute_mode_value != nullptr) {
      dynamic_graph_execute_mode = dynamic_graph_execute_mode_value->s();
    }
    if (dynamic_inputs_shape_range_value != nullptr) {
      dynamic_inputs_shape_range = dynamic_inputs_shape_range_value->s();
    }
    if (local_rank_id_value != nullptr) {
      local_rank_id = local_rank_id_value->s();
    }
    if (local_device_list_value != nullptr) {
      local_device_list = local_device_list_value->s();
    }
    if (in_out_pair_flag_value != nullptr) {
      in_out_pair_flag = in_out_pair_flag_value->s();
    }
    if (in_out_pair_value != nullptr) {
      in_out_pair = in_out_pair_value->s();
    }
  }
  // pass options
  pass_options["do_npu_optimizer"] = do_npu_optimizer;
  pass_options["enable_dp"] = enable_dp;
  pass_options["use_off_line"] = use_off_line;
  pass_options["mix_compile_mode"] = mix_compile_mode;
  pass_options["iterations_per_loop"] = iterations_per_loop;
  pass_options["lower_functional_ops"] = lower_functional_ops;
  pass_options["job"] = job;
  pass_options["task_index"] = task_index;
  pass_options["dynamic_input"] = dynamic_input;
  pass_options["dynamic_graph_execute_mode"] = dynamic_graph_execute_mode;
  pass_options["dynamic_inputs_shape_range"] = dynamic_inputs_shape_range;
  pass_options["local_rank_id"] = local_rank_id;
  pass_options["local_device_list"] = local_device_list;
  pass_options["in_out_pair_flag"] = in_out_pair_flag;
  pass_options["in_out_pair"] = in_out_pair;

  return pass_options;
}

std::map<std::string, std::string> NpuAttrs::GetAllAttrOptions(AttrSlice attrs) {
  std::map<std::string, std::string> all_options;
  std::string do_npu_optimizer = std::to_string(false);
  std::string enable_dp = std::to_string(false);
  std::string use_off_line = std::to_string(true);
  std::string mix_compile_mode = std::to_string(false);
  std::string iterations_per_loop = "1";
  std::string lower_functional_ops = std::to_string(false);
  std::string job = "default";
  std::string task_index = "0";
  std::string local_rank_id = "-1";
  std::string local_device_list;
  std::string in_out_pair_flag = std::to_string(true);
  std::string in_out_pair;

  std::string variable_format_optimize = std::to_string(true);
  std::string hcom_parallel = std::to_string(false);
  std::string graph_memory_max_size;
  std::string variable_memory_max_size;
  std::string enable_dump = std::to_string(false);
  std::string enable_dump_debug = std::to_string(false);
  std::string dump_path;
  std::string dump_step;
  std::string dump_mode = "output";
  std::string dump_debug_mode = "all";
  std::string stream_max_parallel_num;
  std::string soc_config;

  std::string is_tailing_optimization = std::to_string(false);
  std::string precision_mode;
  std::string profiling_mode = std::to_string(false);
  std::string profiling_options;
  std::string auto_tune_mode;
  std::string graph_run_mode = "1";
  std::string op_debug_level = "0";
  std::string enable_scope_fusion_passes;
  std::string enable_exception_dump;
  std::string npuOptimizer;
  std::string op_select_implmode;
  std::string optypelist_for_implmode;
  std::string input_shape;
  std::string dynamic_dims;
  std::string dynamic_node_type;
  std::string aoe_mode;
  std::string work_path;
  std::string distribute_config;
  std::string buffer_optimize = "l2_optimize";
  std::string enable_small_channel = "0";
  std::string fusion_switch_file;
  std::string enable_compress_weight = std::to_string(false);
  std::string compress_weight_conf;
  std::string op_compiler_cache_mode;
  std::string op_compiler_cache_dir;
  std::string debug_dir;
  std::string hcom_multi_mode;
  std::string session_device_id;
  std::string modify_mixlist;
  std::string op_precision_mode;
  std::string device_type = "default_device_type";
  std::string hccl_timeout;
  std::string op_wait_timeout;
  std::string op_execute_timeout;

  auto NpuOptimizer_value = attrs.Find("_NpuOptimizer");
  auto enable_data_pre_proc_value = attrs.Find("_enable_data_pre_proc");
  auto use_off_line_value = attrs.Find("_use_off_line");
  auto mix_compile_mode_value = attrs.Find("_mix_compile_mode");
  auto iterations_per_loop_value = attrs.Find("_iterations_per_loop");
  auto lower_functional_ops_value = attrs.Find("_lower_functional_ops");
  auto job_value = attrs.Find("_job");
  auto task_index_value = attrs.Find("_task_index");
  auto local_rank_id_value = attrs.Find("_local_rank_id");
  auto local_device_list_value = attrs.Find("_local_device_list");
  auto in_out_pair_flag_value = attrs.Find("_in_out_pair_flag");
  auto in_out_pair_value = attrs.Find("_in_out_pair");

  auto variable_format_optimize_value = attrs.Find("_variable_format_optimize");
  auto hcom_parallel_value = attrs.Find("_hcom_parallel");
  auto graph_memory_max_size_value = attrs.Find("_graph_memory_max_size");
  auto variable_memory_max_size_value = attrs.Find("_variable_memory_max_size");
  auto enable_dump_value = attrs.Find("_enable_dump");
  auto enable_dump_debug_value = attrs.Find("_enable_dump_debug");
  auto dump_path_value = attrs.Find("_dump_path");
  auto dump_step_value = attrs.Find("_dump_step");
  auto dump_mode_value = attrs.Find("_dump_mode");
  auto dump_debug_mode_value = attrs.Find("_dump_debug_mode");
  auto stream_max_parallel_num_value = attrs.Find("_stream_max_parallel_num");
  auto soc_config_value = attrs.Find("_soc_config");

  auto is_tailing_optimization_value = attrs.Find("_is_tailing_optimization");
  auto precision_mode_value = attrs.Find("_precision_mode");
  auto profiling_mode_value = attrs.Find("_profiling_mode");
  auto profiling_options_value = attrs.Find("_profiling_options");
  auto auto_tune_mode_value = attrs.Find("_auto_tune_mode");
  auto graph_run_mode_value = attrs.Find("_graph_run_mode");
  auto op_debug_level_value = attrs.Find("_op_debug_level");
  auto enable_scope_fusion_passes_value = attrs.Find("_enable_scope_fusion_passes");
  auto enable_exception_dump_value = attrs.Find("_enable_exception_dump");
  auto op_select_implmode_value = attrs.Find("_op_select_implmode");
  auto optypelist_for_implmode_value = attrs.Find("_optypelist_for_implmode");
  auto input_shape_value = attrs.Find("_input_shape");
  auto dynamic_dims_value = attrs.Find("_dynamic_dims");
  auto dynamic_node_type_value = attrs.Find("_dynamic_node_type");
  auto aoe_mode_value = attrs.Find("_aoe_mode");
  auto work_path_value = attrs.Find("_work_path");
  auto distribute_config_value = attrs.Find("_distribute_config");
  auto buffer_optimize_value = attrs.Find("_buffer_optimize");
  auto enable_small_channel_value = attrs.Find("_enable_small_channel");
  auto fusion_switch_file_value = attrs.Find("_fusion_switch_file");
  auto enable_compress_weight_value = attrs.Find("_enable_compress_weight");
  auto compress_weight_conf_value = attrs.Find("_compress_weight_conf");
  auto op_compiler_cache_mode_value = attrs.Find("_op_compiler_cache_mode");
  auto op_compiler_cache_dir_value = attrs.Find("_op_compiler_cache_dir");
  auto debug_dir_value = attrs.Find("_debug_dir");
  auto hcom_multi_mode_value = attrs.Find("_hcom_multi_mode");
  auto session_device_id_value = attrs.Find("_session_device_id");
  auto modify_mixlist_value = attrs.Find("_modify_mixlist");
  auto op_precision_mode_value = attrs.Find("_op_precision_mode");
  auto device_type_value = attrs.Find("_device_type");
  auto hccl_timeout_value = attrs.Find("_hccl_timeout");
  auto op_wait_timeout_value = attrs.Find("_op_wait_timeout");
  auto op_execute_timeout_value = attrs.Find("_op_execute_timeout");

  if (NpuOptimizer_value != nullptr) {
    do_npu_optimizer = std::to_string(true);
    if (enable_data_pre_proc_value != nullptr) {
      enable_dp = enable_data_pre_proc_value->s();
    }
    if (use_off_line_value != nullptr) {
      use_off_line = use_off_line_value->s();
    }
    if (mix_compile_mode_value != nullptr) {
      mix_compile_mode = mix_compile_mode_value->s();
    }
    if (iterations_per_loop_value != nullptr) {
      iterations_per_loop = iterations_per_loop_value->s();
    }
    if (lower_functional_ops_value != nullptr) {
      lower_functional_ops = lower_functional_ops_value->s();
    }
    if (job_value != nullptr) {
      job = job_value->s();
    } else {
      job = "localhost";
    }
    if (task_index_value != nullptr) {
      task_index = task_index_value->s();
    }
    if (local_rank_id_value != nullptr) {
      local_rank_id = local_rank_id_value->s();
    }
    if (local_device_list_value != nullptr) {
      local_device_list = local_device_list_value->s();
    }
    if (in_out_pair_flag_value != nullptr) {
      in_out_pair_flag = in_out_pair_flag_value->s();
    }
    if (in_out_pair_value != nullptr) {
      in_out_pair = in_out_pair_value->s();
    }

    if (variable_format_optimize_value != nullptr) {
      variable_format_optimize = variable_format_optimize_value->s();
    }
    if (hcom_parallel_value != nullptr) {
      hcom_parallel = hcom_parallel_value->s();
    }
    if (graph_memory_max_size_value != nullptr) {
      graph_memory_max_size = graph_memory_max_size_value->s();
    }
    if (variable_memory_max_size_value != nullptr) {
      variable_memory_max_size = variable_memory_max_size_value->s();
    }
    if (enable_dump_value != nullptr) {
      enable_dump = enable_dump_value->s();
    }
    if (enable_dump_debug_value != nullptr) {
      enable_dump_debug = enable_dump_debug_value->s();
    }
    if (enable_dump != std::to_string(false) || enable_dump_debug != std::to_string(false)) {
      if (dump_path_value != nullptr) {
        dump_path = dump_path_value->s();
      }
    }
    if (enable_dump != std::to_string(false)) {
      if (dump_step_value != nullptr) {
        dump_step = dump_step_value->s();
        if (!dump_step.empty()) {
          Status s = checkDumpStep(dump_step);
          if (!s.ok()) {
            ADP_LOG(FATAL) << s.error_message();
            LOG(FATAL) << s.error_message();
          }
        }
      }
      if (dump_mode_value != nullptr) {
        dump_mode = dump_mode_value->s();
        Status s = checkDumpMode(dump_mode);
        if (!s.ok()) {
          ADP_LOG(FATAL) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
      }
    }
    if (enable_dump_debug != std::to_string(false)) {
      if (dump_debug_mode_value != nullptr) {
        dump_debug_mode = dump_debug_mode_value->s();
        Status s = checkDumpDebugMode(dump_debug_mode);
        if (!s.ok()) {
          ADP_LOG(FATAL) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
      }
    }
    if (stream_max_parallel_num_value != nullptr) {
      stream_max_parallel_num = stream_max_parallel_num_value->s();
    }

    if (is_tailing_optimization_value != nullptr) {
      is_tailing_optimization = is_tailing_optimization_value->s();
    }
    if (precision_mode_value != nullptr) {
      precision_mode = precision_mode_value->s();
    }
    if (profiling_mode_value != nullptr) {
      profiling_mode = profiling_mode_value->s();
    }
    if (profiling_options_value != nullptr) {
      profiling_options = profiling_options_value->s();
    }
    if (auto_tune_mode_value != nullptr) {
      auto_tune_mode = auto_tune_mode_value->s();
    }
    if (graph_run_mode_value != nullptr) {
      graph_run_mode = graph_run_mode_value->s();
    }
    if (op_debug_level_value != nullptr) {
      op_debug_level = op_debug_level_value->s();
    }
    if (enable_scope_fusion_passes_value != nullptr) {
      enable_scope_fusion_passes = enable_scope_fusion_passes_value->s();
    }
    if (enable_exception_dump_value != nullptr) {
      enable_exception_dump = enable_exception_dump_value->s();
    }
    if (op_select_implmode_value != nullptr) {
      op_select_implmode = op_select_implmode_value->s();
    }
    if (optypelist_for_implmode_value != nullptr) {
      optypelist_for_implmode = optypelist_for_implmode_value->s();
    }
    if (input_shape_value != nullptr) {
      input_shape = input_shape_value->s();
    }
    if (dynamic_dims_value != nullptr) {
      dynamic_dims = dynamic_dims_value->s();
    }
    if (dynamic_node_type_value != nullptr) {
      dynamic_node_type = dynamic_node_type_value->s();
    }
    if (aoe_mode_value != nullptr) {
      aoe_mode = aoe_mode_value->s();
    }
    if (work_path_value != nullptr) {
      work_path = work_path_value->s();
    }
    if (distribute_config_value != nullptr) {
      distribute_config = distribute_config_value->s();
    }
    if (buffer_optimize_value != nullptr) {
      buffer_optimize = buffer_optimize_value->s();
    }
    if (enable_small_channel_value != nullptr) {
      enable_small_channel = enable_small_channel_value->s();
    }
    if (fusion_switch_file_value != nullptr) {
      fusion_switch_file = fusion_switch_file_value->s();
    }
    if (enable_compress_weight_value != nullptr) {
      enable_compress_weight = enable_compress_weight_value->s();
    }
    if (compress_weight_conf_value != nullptr) {
      compress_weight_conf = compress_weight_conf_value->s();
    }
    if (op_compiler_cache_mode_value != nullptr) {
      op_compiler_cache_mode = op_compiler_cache_mode_value->s();
    }
    if (op_compiler_cache_dir_value != nullptr) {
      op_compiler_cache_dir = op_compiler_cache_dir_value->s();
    }
    if (debug_dir_value != nullptr) {
      debug_dir = debug_dir_value->s();
    }
    if (hcom_multi_mode_value != nullptr) {
      hcom_multi_mode = hcom_multi_mode_value->s();
    }
    if (session_device_id_value != nullptr) {
      session_device_id = session_device_id_value->s();
    }
    if (modify_mixlist_value != nullptr) {
      modify_mixlist = modify_mixlist_value->s();
    }
    if (hccl_timeout_value != nullptr) {
      hccl_timeout = hccl_timeout_value->s();
    }
    if (op_precision_mode_value != nullptr) {
      op_precision_mode = op_precision_mode_value->s();
    }
    if (device_type_value != nullptr) {
      device_type = device_type_value->s();
    }
    if (soc_config_value != nullptr) {
      soc_config = soc_config_value->s();
    }
    if (op_wait_timeout_value != nullptr) {
      op_wait_timeout = op_wait_timeout_value->s();
    }
    if (op_execute_timeout_value != nullptr) {
      op_execute_timeout = op_execute_timeout_value->s();
    }
  }

  all_options["variable_format_optimize"] = variable_format_optimize;
  all_options["hcom_parallel"] = hcom_parallel;
  all_options["stream_max_parallel_num"] = stream_max_parallel_num;
  if (!graph_memory_max_size.empty()) {
    all_options["graph_memory_max_size"] = graph_memory_max_size;
  }
  if (!variable_memory_max_size.empty()) {
    all_options["variable_memory_max_size"] = variable_memory_max_size;
  }

  all_options["enable_dump"] = enable_dump;
  all_options["dump_path"] = dump_path;
  all_options["dump_step"] = dump_step;
  all_options["dump_mode"] = dump_mode;
  all_options["enable_dump_debug"] = enable_dump_debug;
  all_options["dump_debug_mode"] = dump_debug_mode;
  all_options["soc_config"] = soc_config;

  all_options["is_tailing_optimization"] = is_tailing_optimization;
  all_options["precision_mode"] = precision_mode;
  all_options["profiling_mode"] = profiling_mode;
  all_options["profiling_options"] = profiling_options;
  all_options["auto_tune_mode"] = auto_tune_mode;
  all_options["graph_run_mode"] = graph_run_mode;
  all_options["op_debug_level"] = op_debug_level;
  all_options["enable_scope_fusion_passes"] = enable_scope_fusion_passes;
  all_options["enable_exception_dump"] = enable_exception_dump;

  all_options["do_npu_optimizer"] = do_npu_optimizer;
  all_options["enable_data_pre_proc"] = enable_dp;
  all_options["use_off_line"] = use_off_line;
  all_options["mix_compile_mode"] = mix_compile_mode;
  all_options["iterations_per_loop"] = iterations_per_loop;
  all_options["lower_functional_ops"] = lower_functional_ops;
  all_options["job"] = job;
  all_options["task_index"] = task_index;
  all_options["local_rank_id"] = local_rank_id;
  all_options["local_device_list"] = local_device_list;
  all_options["in_out_pair_flag"] = in_out_pair_flag;
  all_options["in_out_pair"] = in_out_pair;
  all_options["op_select_implmode"] = op_select_implmode;
  all_options["optypelist_for_implmode"] = optypelist_for_implmode;
  all_options["input_shape"] = input_shape;
  all_options["dynamic_dims"] = dynamic_dims;
  all_options["dynamic_node_type"] = dynamic_node_type;
  all_options["aoe_mode"] = aoe_mode;
  all_options["work_path"] = work_path;
  all_options["distribute_config"] = distribute_config;
  all_options["buffer_optimize"] = buffer_optimize;
  all_options["enable_small_channel"] = enable_small_channel;
  all_options["fusion_switch_file"] = fusion_switch_file;
  all_options["enable_compress_weight"] = enable_compress_weight;
  all_options["compress_weight_conf"] = compress_weight_conf;
  all_options["op_compiler_cache_mode"] = op_compiler_cache_mode;
  all_options["op_compiler_cache_dir"] = op_compiler_cache_dir;
  all_options["debug_dir"] = debug_dir;
  all_options["hcom_multi_mode"] = hcom_multi_mode;
  all_options["session_device_id"] = session_device_id;
  all_options["modify_mixlist"] = modify_mixlist;
  all_options["op_precision_mode"] = op_precision_mode;
  all_options["device_type"] = device_type;
  all_options["hccl_timeout"] = hccl_timeout;
  all_options["op_wait_timeout"] = op_wait_timeout;
  all_options["op_execute_timeout"] = op_execute_timeout;

  return all_options;
}

std::map<std::string, std::string> NpuAttrs::GetDefaultPassOptions() {
  std::map<std::string, std::string> pass_options;
  pass_options["do_npu_optimizer"] = std::to_string(false);
  pass_options["enable_dp"] = std::to_string(false);
  pass_options["use_off_line"] = std::to_string(true);
  pass_options["mix_compile_mode"] = std::to_string(false);
  pass_options["iterations_per_loop"] = std::to_string(1);
  pass_options["lower_functional_ops"] = std::to_string(false);
  pass_options["job"] = "default";
  pass_options["task_index"] = std::to_string(0);
  return pass_options;
}

Status NpuAttrs::SetNpuOptimizerAttr(const GraphOptimizationPassOptions &options, Node *node) {
  std::map<std::string, std::string> sess_options;
  bool variable_format_optimize = true;
  bool hcom_parallel = false;
  std::string graph_memory_max_size;
  std::string variable_memory_max_size;
  bool enable_dump = false;
  bool enable_dump_debug = false;
  std::string dump_path;
  std::string dump_step;
  std::string dump_mode = "output";
  std::string dump_debug_mode = "all";
  std::string stream_max_parallel_num;
  std::string soc_config;
  std::string hccl_timeout;

  bool is_tailing_optimization = false;
  std::string precision_mode;
  bool profiling_mode = false;
  std::string profiling_options;
  std::string auto_tune_mode;
  int graph_run_mode = 1;
  int op_debug_level = 0;
  std::string enable_scope_fusion_passes;

  std::map<std::string, std::string> pass_options;
  bool do_npu_optimizer = false;
  bool enable_dp = false;
  bool use_off_line = true;
  bool mix_compile_mode = false;
  int iterations_per_loop = 1;
  bool lower_functional_ops = false;
  std::string job = "localhost";
  int task_index = 0;
  bool dynamic_input = false;
  std::string dynamic_graph_execute_mode = "dynamic_execute";
  std::string dynamic_inputs_shape_range;
  int local_rank_id = -1;
  std::string local_device_list;
  bool in_out_pair_flag = true;
  std::string in_out_pair;
  int enable_exception_dump = 0;
  std::string op_select_implmode;
  std::string optypelist_for_implmode;
  std::string input_shape;
  std::string dynamic_dims;
  int dynamic_node_type = -1;
  std::string aoe_mode;
  std::string work_path = "./";
  std::string distribute_config;
  std::string buffer_optimize = "l2_optimize";
  int enable_small_channel = 0;
  std::string fusion_switch_file;
  bool enable_compress_weight = false;
  std::string compress_weight_conf;
  std::string op_compiler_cache_mode;
  std::string op_compiler_cache_dir;
  std::string debug_dir;
  bool hcom_multi_mode = false;
  int session_device_id = -1;
  std::string modify_mixlist;
  std::string op_precision_mode;
  std::string device_type = "default_device_type";
  std::string op_wait_timeout;
  std::string op_execute_timeout;

  const RewriterConfig &rewrite_options = options.session_options->config.graph_options().rewrite_options();
  for (const auto &custom_optimizer : rewrite_options.custom_optimizers()) {
    if (custom_optimizer.name() == "NpuOptimizer") {
      const auto &params = custom_optimizer.parameter_map();
      if (params.count("variable_format_optimize")) {
        variable_format_optimize = params.at("variable_format_optimize").b();
      }
      if (params.count("hcom_parallel")) {
        hcom_parallel = params.at("hcom_parallel").b();
      }
      if (params.count("graph_memory_max_size")) {
        graph_memory_max_size = params.at("graph_memory_max_size").s();
      }
      if (params.count("variable_memory_max_size")) {
        variable_memory_max_size = params.at("variable_memory_max_size").s();
      }
      if (params.count("enable_dump")) {
        enable_dump = params.at("enable_dump").b();
      }
      if (params.count("enable_dump_debug")) {
        enable_dump_debug = params.at("enable_dump_debug").b();
      }
      if (enable_dump || enable_dump_debug) {
        if (params.count("dump_path")) {
          std::string tmp_path = params.at("dump_path").s();
          Status s = CheckPath(tmp_path, dump_path);
          if (!s.ok()) {
            ADP_LOG(ERROR) << s.error_message();
            LOG(ERROR) << s.error_message();
            return errors::Internal(s.error_message());
          }
        } else {
          ADP_LOG(ERROR) << "if use dump function, dump_path must be set.";
          LOG(ERROR) << "if use dump function, dump_path must be set.";
          return errors::Internal("if use dump function, dump_path must be set.");
        }
      }
      if (enable_dump) {
        if (params.count("dump_step")) {
          dump_step = params.at("dump_step").s();
          Status s = checkDumpStep(dump_step);
          if (!s.ok()) {
            ADP_LOG(FATAL) << s.error_message();
            LOG(FATAL) << s.error_message();
          }
        }
        if (params.count("dump_mode")) {
          dump_mode = params.at("dump_mode").s();
          Status s = checkDumpMode(dump_mode);
          if (!s.ok()) {
            ADP_LOG(FATAL) << s.error_message();
            LOG(FATAL) << s.error_message();
          }
        }
      }
      if (enable_dump_debug) {
        if (params.count("dump_debug_mode")) {
          dump_debug_mode = params.at("dump_debug_mode").s();
          Status s = checkDumpDebugMode(dump_debug_mode);
          if (!s.ok()) {
            ADP_LOG(FATAL) << s.error_message();
            LOG(FATAL) << s.error_message();
          }
        }
      }
      if (params.count("stream_max_parallel_num")) {
        stream_max_parallel_num = params.at("stream_max_parallel_num").s();
      }

      if (params.count("is_tailing_optimization")) {
        is_tailing_optimization = params.at("is_tailing_optimization").b();
      }
      if (params.count("profiling_mode")) {
        profiling_mode = params.at("profiling_mode").b();
      }
      if (profiling_mode) {
        if (params.count("profiling_options")) {
          profiling_options = params.at("profiling_options").s();
        } else {
          ADP_LOG(FATAL) << "profiling_options must be set when use profiling";
          LOG(FATAL) << "profiling_options must be set when use profiling";
        }
      }
      if (params.count("auto_tune_mode")) {
        auto_tune_mode = params.at("auto_tune_mode").s();
      }
      if (params.count("graph_run_mode")) {
        graph_run_mode = params.at("graph_run_mode").i();
        if (graph_run_mode > 1) {
          ADP_LOG(FATAL) << "graph_run_mode value must be 0 or 1";
          LOG(FATAL) << "graph_run_mode value must be 0 or 1";
        }
      }
      if (params.count("op_debug_level")) {
        op_debug_level = params.at("op_debug_level").i();
      }
      if (params.count("enable_scope_fusion_passes")) {
        enable_scope_fusion_passes = params.at("enable_scope_fusion_passes").s();
      }

      if (params.count("aoe_mode")) {
        aoe_mode = params.at("aoe_mode").s();
        if (aoe_mode.empty()) {
          ADP_LOG(ERROR) << "aoe_mode should be one of the list:['1','2','3','4']";
        }
      } else {
        TF_RETURN_IF_ERROR(ReadStringFromEnvVar("AOE_MODE", "", &aoe_mode));
      }
      if (!aoe_mode.empty()) {
        Status s = CheckAoeMode(aoe_mode);
        if (!s.ok()) {
          ADP_LOG(FATAL) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
        if (params.count("work_path")) {
          std::string tmp_path = params.at("work_path").s();
          s = CheckPath(tmp_path, work_path);
          if (!s.ok()) {
            ADP_LOG(FATAL) << s.error_message();
            LOG(FATAL) << s.error_message();
          }
        } else {
          std::string tmp_path = work_path;
          s = CheckPath(tmp_path, work_path);
          if (!s.ok()) {
            ADP_LOG(FATAL) << s.error_message();
            LOG(FATAL) << s.error_message();
          }
        }
        if (params.count("distribute_config")) {
          distribute_config = params.at("distribute_config").s();
        }
      }
      if (params.count("precision_mode")) {
        precision_mode = params.at("precision_mode").s();
      } else {
        if (graph_run_mode) {
          precision_mode = "allow_fp32_to_fp16";
        } else {
          precision_mode = "force_fp16";
        }
      }
      if (params.count("soc_config")) {
        soc_config = params.at("soc_config").s();
      }

      do_npu_optimizer = true;
      if (params.count("enable_data_pre_proc")) {
        enable_dp = params.at("enable_data_pre_proc").b();
      }
      if (params.count("use_off_line")) {
        use_off_line = params.at("use_off_line").b();
      }
      if (params.count("mix_compile_mode")) {
        mix_compile_mode = params.at("mix_compile_mode").b();
      }
      if (params.count("iterations_per_loop")) {
        iterations_per_loop = params.at("iterations_per_loop").i();
      }
      if (params.count("lower_functional_ops")) {
        lower_functional_ops = params.at("lower_functional_ops").b();
      }
      if (params.count("job")) {
        job = params.at("job").s();
      } else {
        job = "localhost";
      }
      if (params.count("task_index")) {
        task_index = params.at("task_index").i();
      }
      if (params.count("dynamic_input")) {
        dynamic_input = params.at("dynamic_input").b();
        if (dynamic_input) {
          if (params.count("dynamic_graph_execute_mode")) {
            dynamic_graph_execute_mode = params.at("dynamic_graph_execute_mode").s();
            if (dynamic_graph_execute_mode != "lazy_recompile" && dynamic_graph_execute_mode != "dynamic_execute") {
              ADP_LOG(ERROR) << "dynamic_graph_execute_mode should be lazy_recompile or dynamic_execute.";
              LOG(ERROR) << "dynamic_graph_execute_mode should be lazy_recompile or dynamic_execute.";
              return errors::Internal("dynamic_graph_execute_mode should be lazy_recompile or dynamic_execute.");
            }
          }
          if (params.count("dynamic_inputs_shape_range")) {
            dynamic_inputs_shape_range = params.at("dynamic_inputs_shape_range").s();
          }
        }
      }
      if (params.count("local_rank_id")) {
        local_rank_id = params.at("local_rank_id").i();
        Status s = CheckLocalRankId(local_rank_id);
        if (!s.ok()) {
          ADP_LOG(ERROR) << s.error_message();
          LOG(ERROR) << s.error_message();
          return errors::Internal(s.error_message());
        }
      }
      if (params.count("local_device_list")) {
        local_device_list = params.at("local_device_list").s();
        Status s = CheckDeviceList(local_device_list);
        if (!s.ok()) {
          ADP_LOG(ERROR) << s.error_message();
          LOG(ERROR) << s.error_message();
          return errors::Internal(s.error_message());
        }
      }
      if (params.count("in_out_pair_flag")) {
        in_out_pair_flag = params.at("in_out_pair_flag").b();
      }
      if (params.count("in_out_pair")) {
        in_out_pair = params.at("in_out_pair").s();
      }

      if (params.count("enable_exception_dump")) {
        enable_exception_dump = params.at("enable_exception_dump").i();
      }
      if (!params.count("op_select_implmode") && !params.count("optypelist_for_implmode")) {
        op_select_implmode = "high_performance";
      } else if (params.count("op_select_implmode") && !params.count("optypelist_for_implmode")) {
        op_select_implmode = params.at("op_select_implmode").s();
        Status s = CheckOpImplMode(op_select_implmode);
        if (!s.ok()) {
          ADP_LOG(FATAL) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
      } else if (params.count("optypelist_for_implmode") && !params.count("op_select_implmode")) {
        ADP_LOG(FATAL) << "when use optypelist_for_implmode, op_select_implmode must be set.";
        LOG(FATAL) << "when use optypelist_for_implmode, op_select_implmode must be set.";
      } else {
        op_select_implmode = params.at("op_select_implmode").s();
        Status s = CheckOpImplMode(op_select_implmode);
        if (!s.ok()) {
          ADP_LOG(FATAL) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
        optypelist_for_implmode = params.at("optypelist_for_implmode").s();
      }
      if (params.count("input_shape") && params.count("dynamic_dims") && params.count("dynamic_node_type")) {
        input_shape = params.at("input_shape").s();
        Status s = CheckInputShape(input_shape);
        if (!s.ok()) {
          ADP_LOG(FATAL) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
        dynamic_dims = params.at("dynamic_dims").s();
        s = CheckDynamicDims(dynamic_dims);
        if (!s.ok()) {
          ADP_LOG(FATAL) << s.error_message();
          LOG(FATAL) << s.error_message();
        }
        dynamic_node_type = params.at("dynamic_node_type").i();
        if (dynamic_node_type < 0 || dynamic_node_type > 1) {
          ADP_LOG(FATAL) << "dynamic_node_type should be 0 or 1.";
          LOG(FATAL) << "dynamic_node_type should be 0 or 1.";
        }
      } else if (!params.count("input_shape") && !params.count("dynamic_dims") && !params.count("dynamic_node_type")) {
        // the three parameters are not set normally.
      } else {
        ADP_LOG(FATAL) << "input_shape, dynamic_dims and dynamic_node_type should use together.";
        LOG(FATAL) << "input_shape, dynamic_dims and dynamic_node_type should use together.";
      }
      if (params.count("buffer_optimize")) {
        buffer_optimize = params.at("buffer_optimize").s();
        if (buffer_optimize != "l2_optimize" && buffer_optimize != "off_optimize") {
          ADP_LOG(FATAL) << "buffer_optimize is valid, should be one of [l2_optimize, off_optimize]";
          LOG(FATAL) << "buffer_optimize is valid, should be one of [l2_optimize, off_optimize]";
        }
      }
      if (params.count("enable_small_channel")) {
        enable_small_channel = params.at("enable_small_channel").i();
      }
      if (params.count("fusion_switch_file")) {
        fusion_switch_file = params.at("fusion_switch_file").s();
      }
      if (params.count("enable_compress_weight") && params.count("compress_weight_conf")) {
        ADP_LOG(FATAL) << "enable_compress_weight can not use with compress_weight_conf.";
        LOG(FATAL) << "enable_compress_weight can not use with compress_weight_conf.";
      }
      if (params.count("enable_compress_weight")) {
        enable_compress_weight = params.at("enable_compress_weight").b();
      }
      if (params.count("compress_weight_conf")) {
        compress_weight_conf = params.at("compress_weight_conf").s();
      }
      if (params.count("op_compiler_cache_mode")) {
        op_compiler_cache_mode = params.at("op_compiler_cache_mode").s();
      }
      if (params.count("op_compiler_cache_dir")) {
        op_compiler_cache_dir = params.at("op_compiler_cache_dir").s();
      }
      if (params.count("debug_dir")) {
        debug_dir = params.at("debug_dir").s();
      }
      if (params.count("hcom_multi_mode")) {
        hcom_multi_mode = params.at("hcom_multi_mode").b();
      }
      if (params.count("session_device_id")) {
        if (params.at("session_device_id").i() >= 0) {
          session_device_id = params.at("session_device_id").i();
        } else {
          ADP_LOG(FATAL) << "session_device_id must be nonnegative integer.";
          LOG(FATAL) << "session_device_id must be nonnegative integer.";
        }
      }
      if (params.count("modify_mixlist")) {
        if (params.count("precision_mode") && params.at("precision_mode").s() == "allow_mix_precision") {
          modify_mixlist = params.at("modify_mixlist").s();
        } else {
          ADP_LOG(ERROR)
              << "modify_mixlist is assigned, please ensure that precision_mode is assigned to 'allow_mix_precision'.";
          LOG(ERROR)
              << "modify_mixlist is assigned, please ensure that precision_mode is assigned to 'allow_mix_precision'.";
          return errors::Internal(
              "modify_mixlist is assigned, please ensure that precision_mode is assigned to 'allow_mix_precision'.");
        }
      }
      if (params.count("op_precision_mode")) {
        op_precision_mode = params.at("op_precision_mode").s();
      }
      if (params.count("device_type")) {
        device_type = params.at("device_type").s();
      }
      if (params.count("hccl_timeout")) {
        hccl_timeout = std::to_string(params.at("hccl_timeout").i());
      }
      if (params.count("op_wait_timeout")) {
        op_wait_timeout = std::to_string(params.at("op_wait_timeout").i());
      }
      if (params.count("op_execute_timeout")) {
        op_execute_timeout = std::to_string(params.at("op_execute_timeout").i());
      }
    }
  }

  // session options
  sess_options["variable_format_optimize"] = std::to_string(variable_format_optimize);
  sess_options["hcom_parallel"] = std::to_string(hcom_parallel);
  sess_options["stream_max_parallel_num"] = stream_max_parallel_num;
  if (!graph_memory_max_size.empty()) {
    sess_options["graph_memory_max_size"] = graph_memory_max_size;
  }
  if (!variable_memory_max_size.empty()) {
    sess_options["variable_memory_max_size"] = variable_memory_max_size;
  }

  sess_options["enable_dump"] = std::to_string(enable_dump);
  sess_options["dump_path"] = dump_path;
  sess_options["dump_step"] = dump_step;
  sess_options["dump_mode"] = dump_mode;
  sess_options["enable_dump_debug"] = std::to_string(enable_dump_debug);
  sess_options["dump_debug_mode"] = dump_debug_mode;
  sess_options["is_tailing_optimization"] = std::to_string(is_tailing_optimization);
  sess_options["op_select_implmode"] = op_select_implmode;
  sess_options["optypelist_for_implmode"] = optypelist_for_implmode;
  sess_options["input_shape"] = input_shape;
  sess_options["dynamic_dims"] = dynamic_dims;
  sess_options["dynamic_node_type"] = std::to_string(dynamic_node_type);
  sess_options["buffer_optimize"] = buffer_optimize;
  sess_options["enable_small_channel"] = std::to_string(enable_small_channel);
  sess_options["fusion_switch_file"] = fusion_switch_file;
  sess_options["enable_compress_weight"] = std::to_string(enable_compress_weight);
  sess_options["compress_weight_conf"] = compress_weight_conf;
  sess_options["hcom_multi_mode"] = std::to_string(hcom_multi_mode);
  sess_options["session_device_id"] = std::to_string(session_device_id);
  sess_options["modify_mixlist"] = modify_mixlist;
  sess_options["op_precision_mode"] = op_precision_mode;
  sess_options["hccl_timeout"] = hccl_timeout;

  init_options_["precision_mode"] = precision_mode;
  if (precision_mode.empty()) {
    init_options_[ge::PRECISION_MODE] = "allow_fp32_to_fp16";
  } else {
    init_options_[ge::PRECISION_MODE] = precision_mode;
  }
  init_options_["profiling_mode"] = std::to_string(profiling_mode);
  init_options_["profiling_options"] = profiling_options;
  init_options_[ge::OPTION_EXEC_PROFILING_MODE] = profiling_mode;
  init_options_[ge::OPTION_EXEC_PROFILING_OPTIONS] = profiling_options;
  init_options_["auto_tune_mode"] = auto_tune_mode;
  init_options_["graph_run_mode"] = std::to_string(graph_run_mode);
  init_options_["op_debug_level"] = std::to_string(op_debug_level);
  init_options_["enable_scope_fusion_passes"] = enable_scope_fusion_passes;
  init_options_["enable_exception_dump"] = std::to_string(enable_exception_dump);
  init_options_["aoe_mode"] = aoe_mode;
  init_options_["work_path"] = work_path;
  init_options_["distribute_config"] = distribute_config;
  init_options_["op_compiler_cache_mode"] = op_compiler_cache_mode;
  init_options_["op_compiler_cache_dir"] = op_compiler_cache_dir;
  init_options_["debug_dir"] = debug_dir;
  init_options_["device_type"] = device_type;
  init_options_["soc_config"] = soc_config;
  if (!soc_config.empty()) {
    init_options_["ge.socVersion"] = soc_config;
  }
  init_options_["op_wait_timeout"] = op_wait_timeout;
  init_options_["op_execute_timeout"] = op_execute_timeout;

  pass_options["do_npu_optimizer"] = std::to_string(do_npu_optimizer);
  pass_options["enable_data_pre_proc"] = std::to_string(enable_dp);
  pass_options["use_off_line"] = std::to_string(use_off_line);
  pass_options["mix_compile_mode"] = std::to_string(mix_compile_mode);
  pass_options["iterations_per_loop"] = std::to_string(iterations_per_loop);
  pass_options["lower_functional_ops"] = std::to_string(lower_functional_ops);
  pass_options["job"] = job;
  pass_options["task_index"] = std::to_string(task_index);
  pass_options["dynamic_input"] = std::to_string(dynamic_input);
  pass_options["dynamic_graph_execute_mode"] = dynamic_graph_execute_mode;
  pass_options["dynamic_inputs_shape_range"] = dynamic_inputs_shape_range;
  pass_options["local_rank_id"] = std::to_string(local_rank_id);
  pass_options["local_device_list"] = local_device_list;
  pass_options["in_out_pair_flag"] = std::to_string(in_out_pair_flag);
  pass_options["in_out_pair"] = in_out_pair;

  if (!node) {
    ADP_LOG(ERROR) << "node is null.";
    LOG(ERROR) << "node is null.";
    return errors::Internal("node is null.");
  }
  std::string attr_name;
  for (const auto &option : sess_options) {
    attr_name = std::string("_") + option.first;
    node->AddAttr(attr_name, option.second);
  }
  for (const auto &option : init_options_) {
    attr_name = std::string("_") + option.first;
    node->AddAttr(attr_name, option.second);
  }
  for (const auto &option : pass_options) {
    attr_name = std::string("_") + option.first;
    node->AddAttr(attr_name, option.second);
  }
  node->AddAttr("_NpuOptimizer", "NpuOptimizer");

  return Status::OK();
}

void NpuAttrs::LogOptions(const std::map<std::string, std::string> &options) {
  for (const auto &option : options) {
    ADP_LOG(INFO) << option.first << ": " << option.second;
  }
}
}  // namespace tensorflow