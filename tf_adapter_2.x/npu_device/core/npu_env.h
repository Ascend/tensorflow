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

#ifndef NPU_DEVICE_CORE_NPU_ENV_H
#define NPU_DEVICE_CORE_NPU_ENV_H

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

const static bool kPerfEnabled = []() -> bool {
  bool perf_enabled = false;
  tensorflow::ReadBoolFromEnvVar("NPU_ENABLE_PERF", false, &perf_enabled);
  return perf_enabled;
}();

const static bool kExecuteOpByAcl = []() -> bool {
  bool execute_op_by_acl = true;
  tensorflow::ReadBoolFromEnvVar("NPU_EXECUTE_OP_BY_ACL", true, &execute_op_by_acl);
  return execute_op_by_acl;
}();

const static bool kGraphEngineGreedyMemory = []() -> bool {
  tensorflow::int64 graph_engine_greedy_memory = 0;
  tensorflow::ReadInt64FromEnvVar("GE_USE_STATIC_MEMORY", 0, &graph_engine_greedy_memory);
  return graph_engine_greedy_memory == 1;
}();

#endif  // NPU_DEVICE_CORE_NPU_ENV_H
