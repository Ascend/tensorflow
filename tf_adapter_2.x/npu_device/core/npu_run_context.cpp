/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "npu_run_context.h"

namespace npu {
RunContextOptions &GetRunContextOptions() {
  static thread_local RunContextOptions run_context_options;
  return run_context_options;
}
}  // namespace npu

extern "C" {
void RunContextOptionsSetMemoryOptimizeOptions(const std::string &recompute) {
  npu::GetRunContextOptions().memory_optimize_options.recompute = recompute;
}
void RunContextOptionsSetGraphParallelOptions(const std::string &enable_graph_parallel,
                                              const std::string &config_path) {
  npu::GetRunContextOptions().graph_parallel_configs.config_path = config_path;
  npu::GetRunContextOptions().graph_parallel_configs.enable_graph_parallel = enable_graph_parallel;
}
void CleanRunContextOptions() { npu::GetRunContextOptions().Clean(); }
}