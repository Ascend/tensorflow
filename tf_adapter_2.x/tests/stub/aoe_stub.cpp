/* Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "aoe_tuning_api.h"

namespace Aoe {
extern "C" AoeStatus AoeInitialize(const std::map<ge::AscendString, ge::AscendString> &globalOptions) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeFinalize() {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeCreateSession(SessionId &SessionId) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeDestroySession(SessionId SessionId) {
  if (SessionId >= 9999) {
    return Aoe::AOE_FALLURE;
  }
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetGeSession(SessionId SessionId, ge::Session* geSession) {
  if (SessionId >= 9999) {
    return Aoe::AOE_FALLURE;
  }
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetDependGraphs(SessionId SessionId, std::vector<ge::Graph> &dependGraph) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetTuningGraph(SessionId SessionId, ge::Graph &tuningGraph) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeTuningGraph(SessionId SessionId,
                                    const std::map<ge::AscendString, ge::AscendString> &tuningOptions) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetDependGraphsInputs(SessionId SessionId,
                                              std::vector<std::vector<ge::Tensor>> &input) {
  return Aoe::AOE_SUCCESS;
}

extern "C" AoeStatus AoeSetTuningGraphInput(SessionId SessionId, std::vector<ge::Tensor> &input) {
  return Aoe::AOE_SUCCESS;
}
} // namespace Aoe