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

#include "toolchain/tuning_tool/aoe_tuning_api.h"

namespace Aoe {
extern "C" Aoe::AoeStatus AoeInitialize(const std::map<Aoe::AscendString, Aoe::AscendString> &globalOptions) {
  return Aoe::AOE_SUCCESS;
}

extern "C" Aoe::AoeStatus AoeFinalize() {
  return Aoe::AOE_SUCCESS;
}

extern "C" Aoe::AoeStatus AoeCreateSession(const std::map<Aoe::AscendString, Aoe::AscendString> &sessionOptions,
                                           Aoe::SessionId &SessionId) {
  return Aoe::AOE_SUCCESS;
}

extern "C" Aoe::AoeStatus AoeDestroySession(Aoe::SessionId SessionId) {
  if (SessionId >= 9999) {
    return Aoe::AOE_FALLURE;
  }
  return Aoe::AOE_SUCCESS;
}

extern "C" Aoe::AoeStatus AoeSetGeSession(Aoe::SessionId SessionId, ge::Session* geSession) {
  if (SessionId >= 9999) {
    return Aoe::AOE_FALLURE;
  }
  return Aoe::AOE_SUCCESS;
}

extern "C" Aoe::AoeStatus AoeSetDependGraphs(Aoe::SessionId SessionId, std::vector<ge::Graph> &dependGraph) {
  return Aoe::AOE_SUCCESS;
}

extern "C" Aoe::AoeStatus AoeSetTuningGraph(Aoe::SessionId SessionId, ge::Graph &tuningGraph) {
  return Aoe::AOE_SUCCESS;
}

extern "C" Aoe::AoeStatus AoeTuningGraph(Aoe::SessionId SessionId,
                                         const std::map<Aoe::AscendString, Aoe::AscendString> &tuningOptions) {
  return Aoe::AOE_SUCCESS;
}

extern "C" Aoe::AoeStatus AoeSetDependGraphsInputs(Aoe::SessionId SessionId,
                                                   std::vector<std::vector<ge::Tensor>> &input) {
  return Aoe::AOE_SUCCESS;
}

extern "C" Aoe::AoeStatus AoeSetTuningGraphInput(Aoe::SessionId SessionId, std::vector<ge::Tensor> &input) {
  return Aoe::AOE_SUCCESS;
}
} // namespace Aoe