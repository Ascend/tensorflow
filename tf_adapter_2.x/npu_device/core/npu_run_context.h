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

#include <string>

struct MemoryOptimizeOptions {
  MemoryOptimizeOptions() {}
  void Clean() { recompute.clear(); }
  std::string recompute;
};

struct GraphParallelConfigs {
  void Clean() {
    config_path.clear();
    enable_graph_parallel.clear();
  }
  std::string config_path;
  std::string enable_graph_parallel;
};

struct RunContextOptions {
  MemoryOptimizeOptions memory_optimize_options;
  GraphParallelConfigs graph_parallel_configs;
  void Clean() {
    memory_optimize_options.Clean();
    graph_parallel_configs.Clean();
  }
};

RunContextOptions& GetRunContextOptions();

extern "C" {
extern void RunContextOptionsSetMemoryOptimizeOptions(const std::string &recompute);
extern void CleanRunContextOptions();
extern void RunContextOptionsSetGraphParallelOptions(const std::string &enable_graph_parallel,
                                                     const std::string &config_path);
}
