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

#include "mmpa/mmpa_api.h"

INT32 mmAccess2(const CHAR *path_name, INT32 mode) {
  if (path_name == NULL) { return EN_INVALID_PARAM; }

  INT32 ret = access(path_name, mode);
  if (ret != EN_OK) { return EN_ERROR; }
  return EN_OK;
}

INT32 mmGetPid() {
  return (INT32)getpid();
}

INT32 mmMkdir(const CHAR *path_name, mmMode_t mode) {
  if (path_name == NULL) { return EN_INVALID_PARAM; }

  INT32 ret = mkdir(path_name, mode);
  if (ret != EN_OK) { return EN_ERROR; }
  return EN_OK;
}

INT32 mmIsDir(const CHAR *file_name) {
  if (file_name == NULL) { return EN_INVALID_PARAM; }
  struct stat file_stat;
  (void)memset_s(&file_stat, sizeof(file_stat), 0, sizeof(file_stat));
  INT32 ret = lstat(file_name, &file_stat);
  if (ret < MMPA_ZERO) { return EN_ERROR; }

  if (!S_ISDIR(file_stat.st_mode)) { return EN_ERROR; } 
  return EN_OK;
}

INT32 mmRealPath(const CHAR *path, CHAR *real_path, INT32 real_path_len) {
  INT32 ret = EN_OK;
  if ((real_path == NULL) || (path == NULL) || (real_path_len < MMPA_MAX_PATH)) {
    return EN_INVALID_PARAM;
  }

  CHAR *ptr = realpath(path, real_path);
  if (ptr == NULL) { ret = EN_ERROR; }
  return ret;
}

VOID *mmDlopen(const CHAR *file_name, INT32 mode) {
  if ((file_name == NULL) || (mode < MMPA_ZERO)) { return NULL; }
  return dlopen(file_name, mode);
}

CHAR *mmDlerror(void) {
  return dlerror();
}

VOID *mmDlsym(VOID *handle, const CHAR *func_name) {
  if ((handle == NULL) || (func_name == NULL)) { return NULL; }
  return dlsym(handle, func_name);
}

INT32 mmDlclose(VOID *handle) {
  if (handle == NULL) { return EN_INVALID_PARAM; }

  INT32 ret = dlclose(handle);
  if (ret != EN_OK) { return EN_ERROR; }
  return EN_OK;
}