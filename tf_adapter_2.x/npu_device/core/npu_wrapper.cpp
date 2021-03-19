/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: Common depends and micro defines for and only for data preprocess module
 */

#include <memory>

#include "Python.h"
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
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/python/eager/pywrap_tensor_conversion.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_ptr.h"
#include "tensorflow/python/util/util.h"

#include "framework/omg/omg_inner_types.h"
#include "framework/omg/parser/parser_api.h"
#include "ge/ge_api.h"

#include "npu_device_register.h"

namespace py = pybind11;

namespace {
TFE_Context *InputTFE_Context(const py::handle &ctx) {
  return static_cast<TFE_Context *>(PyCapsule_GetPointer(ctx.ptr(), nullptr));
}
std::atomic_bool graph_engine_started{false};
const std::string kTrain = "1";
const std::string kOpen = "1";
}  // namespace

PYBIND11_MODULE(_npu_device_backends, m) {
  m.def("Open",
        [](const py::handle &context, const char *device_name, int device_index,
           std::map<std::string, std::string> global_options,
           std::map<std::string, std::string> session_options) -> std::string {
          pybind11::gil_scoped_release release;
          if (!graph_engine_started.exchange(true)) {
            // 只允许在train模式下工作
            global_options[ge::OPTION_GRAPH_RUN_MODE] = kTrain;
            global_options[ge::OPTION_EXEC_DEVICE_ID] = std::to_string(device_index);
            if (global_options.find(ge::PRECISION_MODE) == global_options.end()) {
              global_options[ge::PRECISION_MODE] = "allow_mix_precision";
            }
            LOG(INFO) << "Start graph engine with options:";
            for (const auto &option : global_options) {
              LOG(INFO) << "  " << option.first << ":" << option.second;
            }
            auto ge_status = ge::GEInitialize(global_options);
            if (ge_status != ge::SUCCESS) {
              return "Failed start graph engine:" + ge::StatusFactory::Instance()->GetErrDesc(ge_status);
            }
            LOG(INFO) << "Start graph engine succeed";
            ge_status = ge::ParserInitialize(global_options);
            if (ge_status != ge::SUCCESS) {
              return "Failed start tensorflow model parser:" + ge::StatusFactory::Instance()->GetErrDesc(ge_status);
            }
            LOG(INFO) << "Start tensorflow model parser succeed";
          }

          std::string full_name = tensorflow::strings::StrCat(device_name, ":", device_index);
          tensorflow::DeviceNameUtils::ParsedName parsed_name;
          if (!tensorflow::DeviceNameUtils::ParseFullName(full_name, &parsed_name)) {
            return "Invalid npu device name " + full_name;
          }
          LOG(INFO) << "Create device instance " << full_name << " with options:";
          for (const auto &option : session_options) {
            LOG(INFO) << "  " << option.first << ":" << option.second;
          }
          auto status = CreateDevice(InputTFE_Context(context), full_name.c_str(), device_index, session_options);
          pybind11::gil_scoped_acquire acquire;
          return status;
        });

  m.def("Close", []() {
    pybind11::gil_scoped_release release;
    ReleaseDeviceResource();
    if (graph_engine_started.exchange(false)) {
      auto ge_status = ge::ParserFinalize();
      if (ge_status != ge::SUCCESS) {
        LOG(ERROR) << "Failed stop tensorflow model parser:" << ge::StatusFactory::Instance()->GetErrDesc(ge_status);
      } else {
        LOG(INFO) << "Stop tensorflow model parser succeed";
      }
      ge_status = ge::GEFinalize();
      if (ge_status != ge::SUCCESS) {
        LOG(ERROR) << "Failed stop graph engine:" << ge::StatusFactory::Instance()->GetErrDesc(ge_status);
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
};
