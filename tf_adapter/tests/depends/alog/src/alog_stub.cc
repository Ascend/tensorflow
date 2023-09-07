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
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <stdarg.h>

#define MSG_LENGTH_STUB 1024
int g_logLevel = 0xffffffff;

DLL_EXPORT void SetLogLevelForC(int logLevel) {
  g_logLevel = logLevel;
}

DLL_EXPORT void ClearLogLevelForC() {
  g_logLevel = 0xffffffff;
}

DLL_EXPORT int CheckLogLevelForC(int moduleId, int logLevel) {
  if (logLevel >= g_logLevel) {
    return 1;
  } else {
    return 0;
  }
}

void DlogInnerForC(int moduleId, int level, const char *fmt, ...) {
  int len;
  char msg[MSG_LENGTH_STUB] = {0};
  snprintf(msg,MSG_LENGTH_STUB,"[moduleId:%d] [level:%d] ", moduleId, level);
  va_list ap;

  va_start(ap,fmt);
  len = strlen(msg);
  vsnprintf(msg + len, MSG_LENGTH_STUB- len, fmt, ap);
  va_end(ap);

  printf("\r\n%s",msg);
  fflush(stdout);
  return;
}

DLL_EXPORT int DlogReportFinalize() { return 0; }