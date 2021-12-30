/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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

#ifndef TENSORFLOW_COMMON_COMMON_H_
#define TENSORFLOW_COMMON_COMMON_H_

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tf_adapter/common/adp_logger.h"
#include <string>

#define CHECK_NOT_NULL(v)                                                                                              \
  if ((v) == nullptr) {                                                                                                \
    ADP_LOG(ERROR) << #v " is nullptr.";                                                                               \
    LOG(ERROR) << #v " is nullptr.";                                                                                   \
    return;                                                                                                            \
  }

#define REQUIRES_NOT_NULL(v)                                                                                           \
  if ((v) == nullptr) {                                                                                                \
    ADP_LOG(ERROR) << #v " is nullptr.";                                                                               \
    LOG(ERROR) << #v " is nullptr.";                                                                                   \
    return errors::Internal(#v " is nullptr.");                                                                        \
  }

#define REQUIRES_STATUS_OK(s)                                                                                          \
  if (!(s).ok()) {                                                                                                     \
    return (s);                                                                                                        \
  }

constexpr int ADAPTER_ENV_MAX_LENTH = 1024 * 1024;

#define ADAPTER_LOG_IF_ERROR(...)                                                                                      \
  do {                                                                                                                 \
    const ::tensorflow::Status status = (__VA_ARGS__);                                                                 \
    if (TF_PREDICT_FALSE(!status.ok()))                                                                                \
      LOG(INFO) << status.ToString();                                                                                  \
  } while (0)

inline std::string CatStr(const tensorflow::strings::AlphaNum &a) {
  return StrCat(a);
}

inline std::string CatStr(const tensorflow::strings::AlphaNum &a,
                          const tensorflow::strings::AlphaNum &b) {
  return StrCat(a, b);
}

inline std::string CatStr(const tensorflow::strings::AlphaNum &a,
                          const tensorflow::strings::AlphaNum &b,
                          const tensorflow::strings::AlphaNum &c) {
  return StrCat(a, b, c);
}

inline std::string CatStr(const tensorflow::strings::AlphaNum &a,
                          const tensorflow::strings::AlphaNum &b,
                          const tensorflow::strings::AlphaNum &c,
                          const tensorflow::strings::AlphaNum &d) {
  return StrCat(a, b, c, d);
}

template <typename... AV>
inline std::string CatStr(const tensorflow::strings::AlphaNum &a,
                          const tensorflow::strings::AlphaNum &b,
                          const tensorflow::strings::AlphaNum &c,
                          const tensorflow::strings::AlphaNum &d,
                          const tensorflow::strings::AlphaNum &e,
                          const AV &... args) {
  return StrCat(a, b, c, d, e, args...);
}

#endif  // TENSORFLOW_COMMON_COMMON_H_
