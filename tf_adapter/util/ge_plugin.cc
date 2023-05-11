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
#include <thread>
#include <mmpa/mmpa_api.h>
#include "nlohmann/json.hpp"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/framework/cancellation.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/omg/parser/parser_api.h"
#include "framework/omg/omg_inner_types.h"
#include "ge/ge_api.h"
#include "tdt/tdt_host_interface.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/npu_attrs.h"
#include "acl/acl_rt.h"
#include "tf_adapter/util/npu_plugin.h"
#include "aoe_tuning_api.h"

using AoeStatus = int32_t;
using AoeFinalizeFunc = AoeStatus (*)();
using json = nlohmann::json;

using namespace tdt;
using namespace tensorflow;
namespace {
const int kFatalSleepTime = 3000;
const int64 kInvalidRankSize = -1;
const int64 kDefaultRankSize = 1;
inline string ToString(ge::Status status) {
  return ::ge::StatusFactory::Instance()->GetErrDesc(status);
}
void GeFinalize() {
  // ge finalize
  ge::Status status = ge::GEFinalize();
  if (status != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] GE finalize failed, ret : " << ToString(status);
    std::string error_message = ge::GEGetErrorMsg();
    LOG(ERROR) << "[GePlugin] GE finalize failed, ret : " << ToString(status) << std::endl
               << "Error Message is : " << std::endl
               << error_message;
  }

  // parser finalize
  ge::Status status_parser = ge::ParserFinalize();
  if (status_parser != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] Parser finalize failed, ret : " << ToString(status_parser);
    LOG(ERROR) << "[GePlugin] Parser finalize failed, ret : " << ToString(status_parser);
  }
}
}  // namespace

GePlugin::GePlugin()

    : device_id_(0), isInit_(false), isGlobal_(false) {
  ADP_LOG(INFO) << "[GePlugin] new constructor";
}

GePlugin::~GePlugin() {
  ADP_LOG(INFO) << "[GePlugin] destroy constructor begin";
  Finalize();
  ADP_LOG(INFO) << "[GePlugin] destroy constructor end";
}

/**
 * @brief: get instance
 */
GePlugin *GePlugin::GetInstance() {
  static GePlugin instance;
  return &instance;
}

void GePlugin::Init(std::map<std::string, std::string> &init_options, const bool is_global) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (isInit_) {
    ADP_LOG(INFO) << "[GePlugin] Ge has already initialized";
    return;
  }
  init_options_ = init_options;

  std::string enable_hf32_execution;
  (void) ReadStringFromEnvVar("ENABLE_HF32_EXECUTION", "", &enable_hf32_execution);
  if (!enable_hf32_execution.empty()) {
    init_options["ge.exec.allow_hf32"] = enable_hf32_execution;
    ADP_LOG(INFO) << "[GePlugin] allow_hf32 : " << init_options["ge.exec.allow_hf32"];
  }

  std::string tf_config;
  (void) ReadStringFromEnvVar("TF_CONFIG", "", &tf_config);
  int exec_hccl_flag = 1;
  if (!tf_config.empty()) {
    json config_info;
    try {
      config_info = json::parse(tf_config);
    } catch (json::exception &e) {
      ADP_LOG(WARNING) << "[GePlugin] Failed to convert TF_CONFIG info from string to json ,reason: " << e.what();
      LOG(WARNING) << "[GePlugin] Failed to convert TF_CONFIG info from string to json ,reason: " << e.what();
    }
    if (config_info.is_object()) {
      if (config_info["task"]["type"] == "ps") {
        ADP_LOG(INFO) << "The ps process does not need to be initialized";
        return;
      }
      if (config_info["task"]["type"] == "evaluator") {
        exec_hccl_flag = 0;
      }
    }
  }
  init_options[OPTION_EXEC_HCCL_FLAG] = std::to_string(exec_hccl_flag);

  ADP_LOG(INFO) << "[GePlugin] graph run mode : " << init_options[ge::OPTION_GRAPH_RUN_MODE];

  Status s = GetEnvDeviceID(device_id_);
  if (!s.ok()) {
    ADP_LOG(FATAL) << s.error_message();
    LOG(FATAL) << s.error_message();
  }
  init_options[ge::OPTION_EXEC_DEVICE_ID] = std::to_string(device_id_);
  ADP_LOG(INFO) << "[GePlugin] device id : " << init_options[ge::OPTION_EXEC_DEVICE_ID];

  std::string env_job_id;
  (void) ReadStringFromEnvVar("JOB_ID", "", &env_job_id);
  if (!env_job_id.empty()) {
    init_options[ge::OPTION_EXEC_JOB_ID] = env_job_id;
  } else {
    ADP_LOG(WARNING) << "[GePlugin] can not find Environment variable : JOB_ID";
    LOG(WARNING) << "[GePlugin] can not find Environment variable : JOB_ID";
  }

  std::string cm_chief_ip;
  (void) ReadStringFromEnvVar("CM_CHIEF_IP", "", &cm_chief_ip);
  (void) ReadInt64FromEnvVar("CM_WORKER_SIZE", kInvalidRankSize, &work_size_num);
  std::string env_rank_table_file;
  (void) ReadStringFromEnvVar("RANK_TABLE_FILE", "", &env_rank_table_file);
  (void) ReadInt64FromEnvVar("RANK_SIZE", kInvalidRankSize, &rank_size_num);
  if (!cm_chief_ip.empty() && !env_rank_table_file.empty()) {
    ADP_LOG(ERROR) << "[GePlugin] CM_CHIEF_IP and RANK_TABLE_FILE cannot be configured at the same time.";
    LOG(ERROR) << "[GePlugin] CM_CHIEF_IP and RANK_TABLE_FILE cannot be configured at the same time.";
  } else if (!cm_chief_ip.empty()) {
    SetCmChiefWorkSizeEnv(init_options, cm_chief_ip);
  } else if (!env_rank_table_file.empty()) {
    SetRankTableFileEnv(init_options, env_rank_table_file);
  } else {
    ADP_LOG(INFO) << "[GePlugin] CM_CHIEF_IP and RANK_TABLE_FILE are all not be configured.";
  }

  std::string cluster_info;
  (void) ReadStringFromEnvVar("HELP_CLUSTER", "", &cluster_info);
  if (!cluster_info.empty()) {
    is_use_hcom = true;
  }

  init_options[ge::OPTION_EXEC_IS_USEHCOM] = std::to_string(is_use_hcom);
  init_options[ge::OPTION_EXEC_DEPLOY_MODE] = std::to_string(deploy_mode);

  // is use hcom configuration
  ADP_LOG(INFO) << "[GePlugin] is_usehcom : " << init_options[ge::OPTION_EXEC_IS_USEHCOM]
                << ", deploy_mode :" << init_options[ge::OPTION_EXEC_DEPLOY_MODE];

  // profiling configuration
  ADP_LOG(INFO) << "[GePlugin] profiling_mode : " << init_options[ge::OPTION_EXEC_PROFILING_MODE]
                << ", profiling_options:" << init_options[ge::OPTION_EXEC_PROFILING_OPTIONS];

  // mix precision configuration
  ADP_LOG(INFO) << "[GePlugin] precision_mode : " << init_options[ge::PRECISION_MODE];

  // debug configuration
  ADP_LOG(INFO) << "[GePlugin] op_debug_level : " << init_options[ge::OP_DEBUG_LEVEL];

  ADP_LOG(INFO) << "[GePlugin] ge.deterministic : " << init_options["ge.deterministic"];

  // scope fusion configuration
  ADP_LOG(INFO) << "[GePlugin] enable_scope_fusion_passes : "
                << init_options[ge::OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES];

  // exception dump configuration
  ADP_LOG(INFO) << "[GePlugin] enable_exception_dump : " << init_options["ge.exec.enable_exception_dump"];

  ADP_LOG(INFO) << "[GePlugin] job_id : " << init_options[ge::OPTION_EXEC_JOB_ID];

  ADP_LOG(INFO) << "[GePlugin] op_compiler_cache_mode : " << init_options["ge.op_compiler_cache_mode"];

  ADP_LOG(INFO) << "[GePlugin] op_compiler_cache_dir : " << init_options["ge.op_compiler_cache_dir"];

  ADP_LOG(INFO) << "[GePlugin] debugDir : " << init_options["ge.debugDir"];

  ADP_LOG(INFO) << "[GePlugin] hcom_multi_mode : " << init_options["ge.hcomMultiMode"];

  init_options["ge.fusionTensorSize"] = std::to_string(GetFusionTensorSize());
  ADP_LOG(INFO) << "[GePlugin] fusionTensorSize : " << init_options["ge.fusionTensorSize"];

  // aoe mode and work path
  if (!init_options["ge.jobType"].empty()) {
    init_options["ge.buildMode"] = "tuning";
  }
  ADP_LOG(INFO) << "[GePlugin] aoe mode : " << init_options["ge.jobType"]
                << ", work path : " << init_options["ge.tuningPath"]
                << ", distribute_config : " << init_options["distribute_config"];

  ADP_LOG(INFO) << "[GePlugin] fusion_switch_file :" << init_options["ge.fusionSwitchFile"];

  ADP_LOG(INFO) << "[GePlugin] op_precision_mode :" << init_options[ge::OP_PRECISION_MODE];

  ADP_LOG(INFO) << "[GePlugin] op_select_implmode :" << init_options[ge::OP_SELECT_IMPL_MODE];

  ADP_LOG(INFO) << "[GePlugin] optypelist_for_implmode :" << init_options[ge::OPTYPELIST_FOR_IMPLMODE];

  bool tdt_uninit_env = false;
  (void) ReadBoolFromEnvVar("ASCEND_TDT_UNINIT", false, &tdt_uninit_env);
  if (!kIsHeterogeneous && !tdt_uninit_env) {
    // Open TsdClient first, then call GEInitialize
    ADP_LOG(INFO) << "[GePlugin] Open TsdClient and Init tdt host.";
    int32_t ret = tdt::TdtOutFeedInit(static_cast<uint32_t>(device_id_));
    if (ret != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
      ADP_LOG(FATAL) << "[GePlugin] Tdt host init failed, tdt error code : " << ret;
      LOG(FATAL) << "[GePlugin] Tdt host init failed, tdt error code : " << ret;
    }
  }

  // ge Initialize
  ge::Status status = ge::GEInitialize(init_options);
  if (status != ge::SUCCESS) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
    ADP_LOG(FATAL) << "[GePlugin] Initialize ge failed, ret : " << ToString(status);
    std::string error_message = ge::GEGetErrorMsg();
    LOG(FATAL) << "[GePlugin] Initialize ge failed, ret : " << ToString(status) << std::endl
               << "Error Message is : " << std::endl
               << error_message;
  }
  domi::GetContext().train_flag = true;
  ADP_LOG(INFO) << "[GePlugin] Initialize ge success.";

  // parser Initialize
  ge::Status status_parser = ge::ParserInitialize(init_options);
  if (status_parser != ge::SUCCESS) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
    ADP_LOG(FATAL) << "[GePlugin] Initialize parser failed, ret : " << ToString(status_parser);
    LOG(FATAL) << "[GePlugin] Initialize parser failed, ret : " << ToString(status_parser);
  }
  ADP_LOG(INFO) << "[GePlugin] Initialize parser success.";
  isInit_ = true;
  isGlobal_ = is_global;
}

void GePlugin::SetRankTableFileEnv(std::map<std::string, std::string> &init_options, std::string &rankTableFile) {
  rank_size_num = (rank_size_num == kInvalidRankSize) ? kDefaultRankSize : rank_size_num;
  if (rank_size_num > UINT32_MAX) {
    rank_size_num = UINT32_MAX;
    ADP_LOG(WARNING) << "[GePlugin] RANK_SIZE is larger than UINT32_MAX, set to UINT32_MAX.";
    LOG(WARNING) << "[GePlugin] RANK_SIZE is larger than UINT32_MAX, set to UINT32_MAX.";
  }
  if (!rankTableFile.empty() && (rank_size_num > 0) && (work_size_num == kInvalidRankSize)) {
    ADP_LOG(INFO) << "[GePlugin] env RANK_TABLE_FILE:" << rankTableFile;
    is_use_hcom = true;
    init_options[ge::OPTION_EXEC_RANK_TABLE_FILE] = rankTableFile;
    std::string env_pod_name;
    (void) ReadStringFromEnvVar("POD_NAME", "", &env_pod_name);
    if (!env_pod_name.empty()) {
      deploy_mode = true;
      init_options[ge::OPTION_EXEC_POD_NAME] = env_pod_name;
    } else {
      std::string env_rank_id;
      (void) ReadStringFromEnvVar("RANK_ID", "", &env_rank_id);
      if (!env_rank_id.empty()) {
        ADP_LOG(INFO) << "[GePlugin] env RANK_ID:" << env_rank_id;
        deploy_mode = false;
        init_options[ge::OPTION_EXEC_RANK_ID] = env_rank_id;
      } else {
        ADP_LOG(ERROR) << "[GePlugin] Can't find rank_id or pod_name in env.";
        LOG(ERROR) << "[GePlugin] Can't find rank_id or pod_name in env.";
      }
    }
  }
}

void GePlugin::SetCmChiefWorkSizeEnv(std::map<std::string, std::string> &init_options, std::string &cmChiefIp) {
  std::string cm_chief_port;
  (void) ReadStringFromEnvVar("CM_CHIEF_PORT", "", &cm_chief_port);
  std::string cm_chief_device;
  (void) ReadStringFromEnvVar("CM_CHIEF_DEVICE", "", &cm_chief_device);
  std::string cm_worker_ip;
  (void) ReadStringFromEnvVar("CM_WORKER_IP", "", &cm_worker_ip);
  std::string cm_worker_size;
  (void) ReadStringFromEnvVar("CM_WORKER_SIZE", "", &cm_worker_size);
  work_size_num = (work_size_num == kInvalidRankSize) ? kDefaultRankSize : work_size_num;
  if (work_size_num > UINT32_MAX) {
    work_size_num = UINT32_MAX;
    ADP_LOG(WARNING) << "[GePlugin] RANK_SIZE is larger than UINT32_MAX, set to UINT32_MAX.";
    LOG(WARNING) << "[GePlugin] RANK_SIZE is larger than UINT32_MAX, set to UINT32_MAX.";
  }
  if (!cmChiefIp.empty() && !cm_chief_port.empty() && !cm_chief_device.empty() && (work_size_num > 0) &&
      (rank_size_num == kInvalidRankSize)) {
    is_use_hcom = true;
    init_options["ge.cmChiefIp"] = cmChiefIp;
    init_options["ge.cmChiefPort"] = cm_chief_port;
    init_options["ge.cmChiefWorkerDevice"] = cm_chief_device;
    if (!cm_worker_ip.empty()) {
      init_options["ge.cmWorkerIp"] = cm_worker_ip;
    }
    if (!cm_worker_size.empty()) {
      init_options["ge.cmWorkerSize"] = cm_worker_size;
    }
  }
}

std::map<std::string, std::string> GePlugin::GetInitOptions() {
  return init_options_;
}

uint64_t GePlugin::GetFusionTensorSize() const {
  const int64 fusion_tensor_size_default = 524288000;
  int64 fusion_tensor_size = fusion_tensor_size_default;
  Status s = ReadInt64FromEnvVar("FUSION_TENSOR_SIZE", fusion_tensor_size_default, &fusion_tensor_size);
  if (s.ok() && fusion_tensor_size >= 0) {
    return static_cast<uint64_t>(fusion_tensor_size);
  }
  return static_cast<uint64_t>(fusion_tensor_size_default);
}

void GePlugin::Finalize() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!isInit_) {
    ADP_LOG(INFO) << "[GePlugin] Ge has already finalized.";
    return;
  }

  // ge finalize
  GeFinalize();

  const char *tdt_uninit_env = std::getenv("ASCEND_TDT_UNINIT");
  bool tdt_init = true;
  if (tdt_uninit_env != nullptr && std::atoi(tdt_uninit_env) == 1) {
    tdt_init = false;
  }
  if (!kIsHeterogeneous && tdt_init) {
    ADP_LOG(INFO) << "[GePlugin] Close TsdClient and destroy tdt.";
    int32_t ret = tdt::TdtOutFeedDestroy();
    if (ret != 0) {
      LOG(ERROR) << "[GePlugin] Close tdt host failed.";
      ADP_LOG(ERROR) << "[GePlugin] Close tdt host failed.";
    }
  }
  isInit_ = false;
}

bool GePlugin::IsGlobal() {
  std::lock_guard<std::mutex> lock(mutex_);
  return isGlobal_;
}

static CancellationManager g_cancellationManager;
Status RegisterNpuCancellationCallback(std::function<void()> callback, std::function<void()> *deregister_fn) {
  CancellationToken token = g_cancellationManager.get_cancellation_token();
  if (!g_cancellationManager.RegisterCallback(token, std::move(callback))) {
    return errors::Cancelled("Operation was cancelled");
  }
  *deregister_fn = [token]() { g_cancellationManager.DeregisterCallback(token); };
  return Status::OK();
}

void PluginInit(std::map<std::string, std::string> &init_options) {
  GePlugin::GetInstance()->Init(init_options, true);
  ADP_LOG(INFO) << "[GePlugin] npu plugin init success.";
}

void PluginFinalize() {
  GePlugin::GetInstance()->Finalize();
  ADP_LOG(INFO) << "[GePlugin] npu plugin finalize success.";
}

void AoeFinalizeIfNeed() {
  auto attr = GePlugin::GetInstance()->GetInitOptions();
  if (attr["ge.jobType"].empty() || attr["ge.tuningPath"].empty()) {
    return;
  }

  ADP_LOG(INFO) << "Start to call aoe finalize when npu close.";
  void *handle = mmDlopen("libaoe_tuning.so", MMPA_RTLD_NOW);
  if (handle == nullptr) {
    ADP_LOG(WARNING) << "open libaoe_tuning.so failed.";
    return;
  }

  auto aoe_finalize = reinterpret_cast<AoeFinalizeFunc>(mmDlsym(handle, "AoeFinalize"));
  if (aoe_finalize == nullptr) {
    ADP_LOG(WARNING) << "load aoe finalize function failed.";
    return;
  }

  (void) aoe_finalize();
  (void) mmDlclose(handle);

  ADP_LOG(INFO) << "Finish to call aoe finalize when npu close.";
}

void NpuClose() {
  ADP_LOG(INFO) << "[GePlugin] Npu close.";
  g_cancellationManager.StartCancel();
  GeFinalize();
  AoeFinalizeIfNeed();
  uint32_t device_id = 0;
  (void) GetEnvDeviceID(device_id);
  if (NpuAttrs::GetUseTdtStatus(device_id)) {
    ADP_LOG(INFO) << "[GePlugin] the process has turned on TDT resource, finalize resource at exit.";
    int32_t tdt_status = TdtInFeedDestroy(device_id);
    if (tdt_status != 0) {
      ADP_LOG(ERROR) << "[GePlugin] Tdt client close failed.";
      LOG(ERROR) << "[GePlugin] Tdt client close failed.";
    } else {
      ADP_LOG(INFO) << "[GePlugin] Tdt client close success.";
      NpuAttrs::SetUseTdtStatus(device_id, false);
    }
  }
  ADP_LOG(INFO) << "[GePlugin] npu finalize resource success.";
}

int32_t InitRdmaPool(size_t size) {
  ge::Status ret = ge::InitRdmaPool(size);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] init rdma pool failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] init rdma pool failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] init rdma pool success.";
  return 0;
}

int32_t RegistRdmaRemoteAddr(const std::vector<ge::HostVarInfo> &var_info) {
  ge::Status ret = ge::RdmaRemoteRegister(var_info);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] rdma remote register failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] rdma remote register failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] rdma remote register success.";
  return 0;
}

int32_t RdmaInitAndRegister(const std::vector<ge::HostVarInfo> &var_info, size_t size) {
  ge::Status ret = ge::InitRdmaPool(size);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] init rdma pool failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] init rdma pool failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] init rdma pool success.";
  ret = ge::RdmaRemoteRegister(var_info);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] rdma remote register failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] rdma remote register failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] rdma remote register success.";
  return 0;
}

int32_t GetVarAddrAndSize(const string &var_name, uint64_t &base_addr, uint64_t &var_size) {
  ge::Status ret = ge::GetVarBaseAddrAndSize(var_name, base_addr, var_size);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] get " << var_name << " base addr and size failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] get " << var_name << " base addr and size failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] get " << var_name << " base addr and size success.";
  return 0;
}

int32_t MallocSharedMem(const ge::TensorInfo &tensor_info, uint64_t &dev_addr, uint64_t &memory_size) {
  ge::Status ret = ge::MallocSharedMemory(tensor_info, dev_addr, memory_size);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] malloc shared memory failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] malloc shared memory failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] malloc shared memory success.";
  return 0;
}

int32_t SetDeviceSatMode(uint32_t mode) {
  aclError ret = aclrtSetDeviceSatMode(aclrtFloatOverflowMode(mode));
  if (ret != ACL_SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] set device sat mode failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] set device sat mode failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] set device sat mode success.";
  return 0;
}

int32_t GetDeviceSatMode() {
  aclrtFloatOverflowMode floatOverflowMode = ACL_RT_OVERFLOW_MODE_UNDEF;
  aclError ret = aclrtGetDeviceSatMode(&floatOverflowMode);
  if (ret != ACL_SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] get device sat mode failed, ret : " << ToString(ret);
    LOG(ERROR) << "[GePlugin] get device sat mode failed, ret : " << ToString(ret);
    return -1;
  }
  ADP_LOG(INFO) << "[GePlugin] get device sat mode success.";
  return static_cast<int32_t>(floatOverflowMode);
}
std::atomic_int GePlugin::graph_counter_ = {0};
