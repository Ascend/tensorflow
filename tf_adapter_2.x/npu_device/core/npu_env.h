/* Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_NPU_ENV_H
#define TENSORFLOW_NPU_ENV_H

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"

const static bool kDumpExecutionDetail = []() -> bool {
  bool dump_execute_detail = false;
  tensorflow::ReadBoolFromEnvVar("NPU_DEBUG", false, &dump_execute_detail);
  return dump_execute_detail;
}();

const static bool kDumpGraph = []() -> bool {
  bool dump_graph = false;
  tensorflow::ReadBoolFromEnvVar("NPU_DUMP_GRAPH", false, &dump_graph);
  return dump_graph;
}();

const static bool kCustomKernelEnabled = []() -> bool {
  bool use_custom_kernel = true;
  tensorflow::ReadBoolFromEnvVar("NPU_ENABLE_CUSTOM_KERNEL", true, &use_custom_kernel);
  return use_custom_kernel;
}();

const static int64_t kGlobalLoopSize = []() -> int64_t {
  tensorflow::int64 loop_size = 1;
  tensorflow::ReadInt64FromEnvVar("NPU_LOOP_SIZE", 1, &loop_size);
  return loop_size;
}();

const static bool kPerfEnabled = []() -> bool {
  bool perf_enabled = false;
  tensorflow::ReadBoolFromEnvVar("NPU_ENABLE_PERF", false, &perf_enabled);
  return perf_enabled;
}();
#endif  // TENSORFLOW_NPU_ENV_H
