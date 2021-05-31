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

#include "toolchain/slog.h"
#include "toolchain/plog.h"

DLL_EXPORT int CheckLogLevelForC(int moduleId, int logLevel) { return 1; }

void DlogInnerForC(int moduleId, int level, const char *fmt, ...) { return; }

#define DlogForC(moduleId, level, fmt, ...)                                                 \
  do {                                                                                  \
    if(CheckLogLevelForC(moduleId, level) == 1) {                                           \
        DlogInnerForC(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);   \
     }                                                                                  \
  } while (0)

DLL_EXPORT int DlogReportFinalize() { return 0; }