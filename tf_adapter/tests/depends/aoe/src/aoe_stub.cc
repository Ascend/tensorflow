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

#include "toolchain/tuning_tool/tune_api.h"

extern "C" AoeStatus AoeOnlineInitialize(ge::Session *session, const std::map<std::string, std::string> &option) {
  if (option.empty()) {
    return AOE_FALLURE;
  }
  return AOE_SUCCESS;
}

extern "C" AoeStatus AoeOnlineFinalize() {
  return AOE_SUCCESS;
}

extern "C" AoeStatus AoeOnlineTuning(ge::Graph &tuningGraph, std::vector<ge::Graph> &dependGraph,
    ge::Session *session, const std::map<std::string, std::string> &option) {
  if (option.empty()) {
    return AOE_FALLURE;
  }
  return AOE_SUCCESS;
}