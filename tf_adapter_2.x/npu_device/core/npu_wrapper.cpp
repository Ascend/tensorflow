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

#include <memory>

#include "npu_python.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/dlpack.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/python/eager/pywrap_tensor_conversion.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_ptr.h"
#include "tensorflow/python/util/util.h"

#include "framework/omg/parser/parser_api.h"
#include "ge/ge_api.h"

#include "acl/acl_rt.h"
#include "npu_aoe.h"
#include "npu_device_register.h"
#include "npu_global.h"
#include "npu_logger.h"
#include "npu_utils.h"
#include "npu_thread_pool.h"
#include "npu_run_context.h"

namespace py = pybind11;

namespace {
TFE_Context *InputTFE_Context(const py::handle &ctx) {
  return static_cast<TFE_Context *>(PyCapsule_GetPointer(ctx.ptr(), nullptr));
}
std::atomic_bool graph_engine_started{false};

const std::map<std::string, std::string> kConfigurableOptions = {
  {"graph_run_mode", ge::OPTION_GRAPH_RUN_MODE},
  {"graph_memory_max_size", ge::GRAPH_MEMORY_MAX_SIZE},
  {"variable_memory_max_size", ge::VARIABLE_MEMORY_MAX_SIZE},
  {"variable_format_optimize", "ge.exec.variable_acc"},
  {"enable_scope_fusion_passes", ge::OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES},
  {"fusion_switch_file", ge::FUSION_SWITCH_FILE},
  {"precision_mode", ge::PRECISION_MODE},
  {"op_select_implmode", ge::OP_SELECT_IMPL_MODE},
  {"optypelist_for_implmode", ge::OPTYPELIST_FOR_IMPLMODE},
  {"op_compiler_cache_mode", ge::OP_COMPILER_CACHE_MODE},
  {"op_compiler_cache_dir", ge::OP_COMPILER_CACHE_DIR},
  {"stream_max_parallel_num", ge::STREAM_MAX_PARALLEL_NUM},
  {"hcom_parallel", ge::HCOM_PARALLEL},
  {"hcom_multi_mode", ge::HCOM_MULTI_MODE},
  {"is_tailing_optimization", ge::OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION},
  {"op_debug_level", ge::OP_DEBUG_LEVEL},
  {"op_debug_config", "op_debug_config"},
  {"debug_dir", ge::DEBUG_DIR},
  {"modify_mixlist", ge::MODIFY_MIXLIST},
  {"enable_exception_dump", ge::OPTION_EXEC_ENABLE_EXCEPTION_DUMP},
  {"enable_dump", ge::OPTION_EXEC_ENABLE_DUMP},
  {"dump_path", ge::OPTION_EXEC_DUMP_PATH},
  {"dump_step", ge::OPTION_EXEC_DUMP_STEP},
  {"dump_mode", ge::OPTION_EXEC_DUMP_MODE},
  {"enable_dump_debug", ge::OPTION_EXEC_ENABLE_DUMP_DEBUG},
  {"dump_debug_mode", ge::OPTION_EXEC_DUMP_DEBUG_MODE},
  {"enable_profiling", ge::OPTION_EXEC_PROFILING_MODE},
  {"profiling_options", ge::OPTION_EXEC_PROFILING_OPTIONS},
  {"aoe_mode", "ge.jobType"},
  {"work_path", "ge.tuningPath"},
  {"input_shape", ge::INPUT_SHAPE},
  {"dynamic_node_type", ge::DYNAMIC_NODE_TYPE},
  {"dynamic_dims", ge::kDynamicDims},
  {"enable_small_channel", ge::ENABLE_SMALL_CHANNEL},
  {"deterministic", "ge.deterministic"},
  {"op_precision_mode", "ge.exec.op_precision_mode"},
  {"graph_exec_timeout", "ge.exec.graphExecTimeout"},
  {"logical_device_cluster_deploy_mode", ge::OPTION_EXEC_LOGICAL_DEVICE_CLUSTER_DEPLOY_MODE},
  {"logical_device_id", ge::OPTION_EXEC_LOGICAL_DEVICE_ID},
  {"model_deploy_mode", "ge.exec.modelDeployMode"},
  {"model_deploy_devicelist", "ge.exec.modelDeployDevicelist"},
  {"topo_sorting_mode", "ge.topoSortingMode"},
  {"overflow_flag", "ge.exec.overflow"},
  {"insert_op_file", "ge.insertOpFile"},
  {"customize_dtypes", "ge.customizeDtypes"},
  {"dump_data", "ge.exec.dumpData"},
  {"dump_layer", "ge.exec.dumpLayer"},
  {"aoe_config_file", "ge.aoe_config_file"},
  {"stream_sync_timeout", "stream_sync_timeout"},
  {"event_sync_timeout", "event_sync_timeout"},
  {"external_weight", "ge.externalWeight"},
  // private options
  {"_distribute.rank_id", ge::OPTION_EXEC_RANK_ID},
  {"_distribute.rank_table", ge::OPTION_EXEC_RANK_TABLE_FILE},
  {"resource_config_path", "ge.resourceConfigPath"},
  {"graph_parallel_option_path", "ge.graphParallelOptionPath"},
  {"enable_graph_parallel", "ge.enableGraphParallel"},
  {"atomic_clean_policy", "ge.exec.atomicCleanPolicy"},
  {"static_memory_policy", "ge.exec.staticMemoryPolicy"},
  {"_distribute.cm_chief_ip", ge::OPTION_EXEC_CM_CHIEF_IP},
  {"_distribute.cm_chief_port", ge::OPTION_EXEC_CM_CHIEF_PORT},
  {"_distribute.cm_chief_worker_device", ge::OPTION_EXEC_CM_CHIEF_DEVICE},
  {"_distribute.cm_worker_ip", ge::OPTION_EXEC_CM_WORKER_IP},
  {"_distribute.cm_worker_size", ge::OPTION_EXEC_CM_WORKER_SIZE},
  {"jit_compile", "ge.jit_compile"}};
}  // namespace

#undef PYBIND11_CHECK_PYTHON_VERSION
#define PYBIND11_CHECK_PYTHON_VERSION

namespace {
std::unordered_set<std::string> npu_specify_ops_cache;
constexpr uint32_t kDeviceSatModeLimit = 2U;
}
namespace npu {
bool CheckIsDistribute(std::map<std::string, std::string> &global_options) {
  return ((global_options.find(ge::OPTION_EXEC_RANK_TABLE_FILE) != global_options.end() &&
           global_options.find(ge::OPTION_EXEC_RANK_ID) != global_options.end()) ||
          (global_options.find(ge::OPTION_EXEC_CM_CHIEF_IP) != global_options.end() &&
           global_options.find(ge::OPTION_EXEC_CM_CHIEF_PORT) != global_options.end() &&
           global_options.find(ge::OPTION_EXEC_CM_CHIEF_DEVICE) != global_options.end()));
}
void ParseGlobalOptions(int device_index, const std::map<std::string, std::string> &user_options,
                        std::map<std::string, std::string> &global_options) {
  for (const auto &option : user_options) {
    auto iter = kConfigurableOptions.find(option.first);
    if (iter != kConfigurableOptions.end()) {
      global_options[iter->second] = option.second;
    } else {
      LOG(WARNING) << "Unrecognized graph engine option " << option.first << ":" << option.second;
    }
  }
  if (CheckIsDistribute(global_options)) {
    const static std::string kTrue = "1";
    global_options[ge::OPTION_EXEC_DEPLOY_MODE] = "0";
    global_options[ge::OPTION_EXEC_IS_USEHCOM] = kTrue;
    global_options[ge::OPTION_EXEC_HCCL_FLAG] = kTrue;
    global_options["ge.exec.hccl_tailing_optimize"] = kTrue;
  }

  if (global_options.find("ge.jobType") == global_options.end()) {
    if (!kAoeMode.empty()) {
      global_options["ge.jobType"] = kAoeMode;
      global_options["ge.buildMode"] = "tuning";
    }
  } else {
    global_options["ge.buildMode"] = "tuning";
  }

  global_options[ge::OPTION_EXEC_DEVICE_ID] = std::to_string(device_index);
  if (global_options[ge::OPTION_GRAPH_RUN_MODE] == "0") {
    global_options[ge::ENABLE_SMALL_CHANNEL] = "1";
  }
}

PYBIND11_MODULE(_npu_device_backends, m) {
  (void)m.def("Open",
              [](const py::handle &context, const char *device_name, int device_index,
                 const std::map<std::string, std::string> &user_options,
                 std::map<std::string, std::string> device_options) -> std::string {
                pybind11::gil_scoped_release release;
                static std::map<std::string, std::string> global_options;
                if (!graph_engine_started.exchange(true)) {
                  ParseGlobalOptions(device_index, user_options, global_options);
                  LOG(INFO) << "Start graph engine with options:";
                  for (const auto &option : global_options) {
                    LOG(INFO) << "  " << option.first << ":" << option.second;
                  }
                  auto ge_status = ge::GEInitialize(global_options);
                  if (ge_status != ge::SUCCESS) {
                    return "Failed start graph engine:" + ge::GEGetErrorMsg();
                  }
                  LOG(INFO) << "Start graph engine succeed";
                  ge_status = ge::ParserInitialize(global_options);
                  if (ge_status != ge::SUCCESS) {
                    return "Failed start tensorflow model parser:" + ge::GEGetErrorMsg();
                  }
                  LOG(INFO) << "Start tensorflow model parser succeed";

                  // initialize aoe tuning if need
                  if (!global_options["ge.jobType"].empty()) {
                    auto status = npu::NpuAoe::GetInstance().AoeTuningInitialize(global_options["ge.tuningPath"]);
                    if (!status.ok()) {
                      return status.error_message();
                    }
                  }
                  auto status = npu::global::RtsCtx::CreateGlobalCtx(device_index);
                  if (!status.ok()) {
                    return status.error_message();
                  }
                  status = npu::global::RtsCtx::EnsureInitialized();
                  if (!status.ok()) {
                    return status.error_message();
                  }
                }

                std::string full_name = tensorflow::strings::StrCat(device_name, ":", device_index);
                tensorflow::DeviceNameUtils::ParsedName parsed_name;
                if (!tensorflow::DeviceNameUtils::ParseFullName(full_name, &parsed_name)) {
                  return "Invalid npu device name " + full_name;
                }
                LOG(INFO) << "Create device instance " << full_name << " with extra options:";
                for (const auto &option : device_options) {
                  LOG(INFO) << "  " << option.first << ":" << option.second;
                  (void)global_options.emplace(option.first, option.second);
                }
                NpuThreadPool::GetInstance().Init(kDefaultThreadNum);
                // Currently only support global basic options
                auto status =
                  npu::CreateDevice(InputTFE_Context(context), full_name.c_str(), device_index, global_options);
                pybind11::gil_scoped_acquire acquire;
                return status;
              });

  (void)m.def("Close", []() {
    pybind11::gil_scoped_release release;
    NpuThreadPool::GetInstance().Destroy();
    npu::ReleaseDeviceResource();
    if (graph_engine_started.exchange(false)) {
      auto ge_status = ge::ParserFinalize();
      if (ge_status != ge::SUCCESS) {
        LOG(ERROR) << "Failed stop tensorflow model parser:" << ge::GEGetErrorMsg();
      } else {
        LOG(INFO) << "Stop tensorflow model parser succeed";
      }
      npu::global::dev_memory_shared_lock.lock();
      npu::global::dev_memory_released = true;
      npu::global::dev_memory_shared_lock.unlock();

      ge_status = ge::GEFinalize();
      if (ge_status != ge::SUCCESS) {
        LOG(ERROR) << "Failed stop graph engine:" << ge::GEGetErrorMsg();
      } else {
        LOG(INFO) << "Stop graph engine succeed";
      }

      (void)npu::NpuAoe::GetInstance().AoeTuningFinalize();
      (void)npu::global::RtsCtx::DestroyGlobalCtx();
    }
    pybind11::gil_scoped_acquire acquire;
  });

  (void)m.def("StupidRepeat", [](const char *device_name, int times) {
    for (int i = 0; i < times; i++) {
      LOG(INFO) << device_name;
    }
  });

  (void)m.def("WatchOpRegister", []() {
    npu_specify_ops_cache.clear();
    tensorflow::OpList ops;
    tensorflow::OpRegistry::Global()->Export(true, &ops);
    for (auto &op : ops.op()) {
      (void)npu_specify_ops_cache.insert(op.name());
    }
  });

  (void)m.def("StopWatchOpRegister", []() {
    tensorflow::OpList ops;
    tensorflow::OpRegistry::Global()->Export(true, &ops);
    for (auto &op : ops.op()) {
      if (npu_specify_ops_cache.count(op.name()) == 0) {
        if (global::g_npu_specify_ops.insert(op.name()).second) {
          DLOG() << "Register npu specific op " << op.name();
        }
      }
    }
  });

  (void)m.def("SetNpuLoopSize", [](int64_t loop_size) {
    if (loop_size <= 0) {
      LOG(ERROR) << "Npu loop size must be greater than 0, got " << loop_size;
      return;
    }
    npu::global::g_npu_loop_size = loop_size;
    LOG(INFO) << "Npu loop size is set to " << npu::global::g_npu_loop_size
              << ", it will take effect in the next training loop";
  });

  (void)m.def("SetDeviceSatMode", [](uint32_t mode) {
    if (mode > kDeviceSatModeLimit) {
      LOG(ERROR) << "overflow mode is unvalid" << mode;
      return;
    }
    aclrtSetDeviceSatMode(aclrtFloatOverflowMode(mode));
  });

  (void)m.def("GetDeviceSatMode", []() -> std::int32_t {
    aclrtFloatOverflowMode mode = ACL_RT_OVERFLOW_MODE_UNDEF;
    aclError ret = aclrtGetDeviceSatMode(&mode);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "get device sat mode failed";
      return -1;
    }
    LOG(INFO) << "get deviceSatMode success";
    return static_cast<int32_t>(mode);
  });

  (void)m.def("RunContextOptionsSetMemoryOptimizeOptions", &RunContextOptionsSetMemoryOptimizeOptions);
  (void)m.def("CleanRunContextOptions", &CleanRunContextOptions);
};
}  // namespace npu
