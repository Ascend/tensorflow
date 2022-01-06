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

#include "Python.h"

#ifdef ASCEND_CI_LIMITED_PY37
#undef PyCFunction_NewEx
#endif

#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
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

#include "npu_device_register.h"
#include "npu_global.h"
#include "npu_micros.h"
#include "npu_utils.h"

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
  {"auto_tune_mode", ge::AUTO_TUNE_MODE},
  {"op_select_implmode", ge::OP_SELECT_IMPL_MODE},
  {"optypelist_for_implmode", ge::OPTYPELIST_FOR_IMPLMODE},
  {"op_compiler_cache_mode", ge::OP_COMPILER_CACHE_MODE},
  {"op_compiler_cache_dir", ge::OP_COMPILER_CACHE_DIR},
  {"stream_max_parallel_num", ge::STREAM_MAX_PARALLEL_NUM},
  {"hcom_parallel", ge::HCOM_PARALLEL},
  {"hcom_multi_mode", ge::HCOM_MULTI_MODE},
  {"is_tailing_optimization", ge::OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION},
  {"op_debug_level", ge::OP_DEBUG_LEVEL},
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
  // private options
  {"_distribute.rank_id", ge::OPTION_EXEC_RANK_ID},
  {"_distribute.rank_table", ge::OPTION_EXEC_RANK_TABLE_FILE}};
}  // namespace

#undef PYBIND11_CHECK_PYTHON_VERSION
#define PYBIND11_CHECK_PYTHON_VERSION

namespace npu {
PYBIND11_MODULE(_npu_device_backends, m) {
  m.def("Open",
        [](const py::handle &context, const char *device_name, int device_index,
           std::map<std::string, std::string> user_options,
           std::map<std::string, std::string> device_options) -> std::string {
          pybind11::gil_scoped_release release;
          static std::map<std::string, std::string> global_options;
          if (!graph_engine_started.exchange(true)) {
            for (const auto &option : user_options) {
              auto iter = kConfigurableOptions.find(option.first);
              if (iter != kConfigurableOptions.end()) {
                global_options[iter->second] = option.second;
              } else {
                LOG(WARNING) << "Unrecognized graph engine option " << option.first << ":" << option.second;
              }
            }

            if (global_options.find(ge::OPTION_EXEC_RANK_TABLE_FILE) != global_options.end() &&
                global_options.find(ge::OPTION_EXEC_RANK_ID) != global_options.end()) {
              const static std::string kTrue = "1";
              global_options[ge::OPTION_EXEC_DEPLOY_MODE] = "0";
              global_options[ge::OPTION_EXEC_IS_USEHCOM] = kTrue;
              global_options[ge::OPTION_EXEC_HCCL_FLAG] = kTrue;
              global_options["ge.exec.hccl_tailing_optimize"] = kTrue;
            }

            global_options[ge::OPTION_EXEC_DEVICE_ID] = std::to_string(device_index);
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
            aclrtContext global_rt_ctx = nullptr;
            auto status = [&global_rt_ctx, device_index]() -> tensorflow::Status {
              NPU_REQUIRES_ACL_OK("Acl create rts ctx failed", aclrtCreateContext(&global_rt_ctx, device_index));
              return tensorflow::Status::OK();
            }();
            if (!status.ok()) {
              return status.error_message();
            }
            npu::global::RtsCtx::SetGlobalCtx(global_rt_ctx);
            status = npu::global::RtsCtx::EnsureInitialized();
            if (!status.ok()) {
              return status.error_message();
            }
          }

          std::string full_name = CatStr(device_name, ":", device_index);
          tensorflow::DeviceNameUtils::ParsedName parsed_name;
          if (!tensorflow::DeviceNameUtils::ParseFullName(full_name, &parsed_name)) {
            return "Invalid npu device name " + full_name;
          }
          LOG(INFO) << "Create device instance " << full_name << " with extra options:";
          for (const auto &option : device_options) {
            LOG(INFO) << "  " << option.first << ":" << option.second;
          }
          // Currently only support global basic options
          auto status = npu::CreateDevice(InputTFE_Context(context), full_name.c_str(), device_index, global_options);
          pybind11::gil_scoped_acquire acquire;
          return status;
        });

  m.def("Close", []() {
    pybind11::gil_scoped_release release;
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
    }
    pybind11::gil_scoped_acquire acquire;
  });

  m.def("StupidRepeat", [](const char *device_name, int times) {
    for (int i = 0; i < times; i++) {
      LOG(INFO) << device_name;
    }
  });

  m.def("SetNpuLoopSize", [](int64_t loop_size) {
    if (loop_size <= 0) {
      LOG(ERROR) << "Npu loop size must be greater than 0, got " << loop_size;
      return;
    }
    npu::global::g_npu_loop_size = loop_size;
    LOG(INFO) << "Npu loop size is set to " << npu::global::g_npu_loop_size
              << ", it will take effect in the next training loop";
  });
};
}  // namespace npu