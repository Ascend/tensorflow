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

#ifndef NPU_DEVICE_CORE_NPU_AOE_H
#define NPU_DEVICE_CORE_NPU_AOE_H

#include <map>
#include "aoe_tuning_api.h"
#include "npu_device.h"

namespace npu {
using SessionId = uint64_t;
using AoeInitializeFunc = Aoe::AoeStatus (*)(const std::map<Aoe::AscendString, Aoe::AscendString> &);
using AoeFinalizeFunc = Aoe::AoeStatus (*)();
using AoeCreateSessionFunc = Aoe::AoeStatus (*)(const std::map<Aoe::AscendString, Aoe::AscendString> &, SessionId &);
using AoeDestroySessionFunc = Aoe::AoeStatus (*)(SessionId);
using AoeSetGeSessionFunc = Aoe::AoeStatus (*)(SessionId, ge::Session *);
using AoeSetDependGraphFunc = Aoe::AoeStatus (*)(SessionId, std::vector<ge::Graph> &);
using AoeSetDependGraphsInputsFunc = Aoe::AoeStatus (*)(SessionId, std::vector<std::vector<ge::Tensor>> &);
using AoeSetTuningGraphInputFunc = Aoe::AoeStatus (*)(SessionId, std::vector<ge::Tensor> &);
using AoeSetTuningGraphFunc = Aoe::AoeStatus (*)(SessionId, ge::Graph &);
using AoeTuningGraphFunc = Aoe::AoeStatus (*)(SessionId, const std::map<Aoe::AscendString, Aoe::AscendString> &);

struct AoeFunc {
  AoeInitializeFunc aoe_initialize = nullptr;
  AoeFinalizeFunc aoe_finalize = nullptr;
  AoeCreateSessionFunc aoe_create_session = nullptr;
  AoeDestroySessionFunc aoe_destroy_session = nullptr;
  AoeSetGeSessionFunc aoe_set_gesession = nullptr;
  AoeSetDependGraphFunc aoe_set_dependgraphs = nullptr;
  AoeSetTuningGraphFunc aoe_set_tuninggraph = nullptr;
  AoeTuningGraphFunc aoe_tuning_graph = nullptr;
  AoeSetDependGraphsInputsFunc aoe_set_depend_graphs_inputs = nullptr;
  AoeSetTuningGraphInputFunc aoe_set_tuning_graph_input = nullptr;
};

class NpuAoe {
 public:
  NpuAoe() = default;
  ~NpuAoe();

  static NpuAoe &GetInstance();
  tensorflow::Status AoeTuningInitialize(const std::string &work_path);
  tensorflow::Status RunAoeTuning(NpuDevice &device, TFE_Context *context, bool need_build, uint64_t graph_id,
                                  const std::string &name, const tensorflow::GraphDef &graph_def,
                                  std::vector<TFE_TensorHandle *> &inputs);
  tensorflow::Status AoeTuningFinalize();

  NpuAoe(const NpuAoe&) = delete;
  NpuAoe(NpuAoe &&) = delete;
  NpuAoe& operator=(const NpuAoe&) = delete;
  NpuAoe& operator=(NpuAoe &&) = delete;

 private:
  tensorflow::Status LoadAoeFunc();

  AoeFunc aoe_func_;
  void *handle_ = nullptr;
  int64_t exec_num_ = 0;
  std::map<uint64_t, ge::Graph> ge_graph_;
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_AOE_H