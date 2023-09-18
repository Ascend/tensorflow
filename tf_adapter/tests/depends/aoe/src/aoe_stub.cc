/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "aoe_tuning_api.h"

namespace Aoe {
extern "C" AoeStatus AoeInitialize(const std::map<ge::AscendString, ge::AscendString> &globalOptions) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeFinalize() {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeCreateSession(uint64_t &sessionId) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeDestroySession(uint64_t sessionId) {
  if (sessionId >= 9999) {
    return Aoe::AOE_FAILURE;
  }
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetGeSession(uint64_t sessionId, ge::Session *geSession) {
  if (sessionId >= 9999) {
    return Aoe::AOE_FAILURE;
  }
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetDependGraphs(uint64_t sessionId, const std::vector<ge::Graph> &dependGraphs) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetTuningGraph(uint64_t sessionId, const ge::Graph &tuningGraph) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeTuningGraph(uint64_t sessionId,
                                    const std::map<ge::AscendString, ge::AscendString> &tuningOptions) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetDependGraphsInputs(uint64_t sessionId,
                                              const std::vector<std::vector<ge::Tensor>> &inputs) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetTuningGraphInput(uint64_t sessionId, const std::vector<ge::Tensor> &input) {
  return Aoe::AOE_SUCCESS;
}
} // namespace Aoe