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

#ifndef TENSORFLOW_ADP_LOGGER_H
#define TENSORFLOW_ADP_LOGGER_H

#include <sstream>

#define FMK_MODULE_NAME static_cast<int>(FMK)

#define LOG_DEPRECATED_WITH_REPLACEMENT(old, replacement)                                                              \
  do {                                                                                                                 \
    ADP_LOG(WARNING) << "The " #old " option IS DEPRECATED. It will be removed in a future version. Please "           \
                        "use " #replacement " instead";                                                                \
  } while (false)

#define LOG_DEPRECATED(old)                                                                                            \
  do {                                                                                                                 \
    ADP_LOG(WARNING) << "The " #old " option IS DEPRECATED. It will be removed in a future version.";                  \
  } while (false)

namespace npu {
constexpr const char *ADP_MODULE_NAME = "TF_ADAPTER";
const int ADP_DEBUG = 0;
const int ADP_INFO = 1;
const int ADP_WARNING = 2;
const int ADP_ERROR = 3;
const int ADP_EVENT = 16;
const int ADP_FATAL = 32;

class AdapterLogger : public std::basic_ostringstream<char> {
 public:
  AdapterLogger(const char *fname, int line, int severity) : severity_(severity) {
    *this << " [" << fname << ":" << line << "]"
          << " ";
  }
  ~AdapterLogger() noexcept override;

 private:
  int severity_;
};
}  // namespace npu

#define ADP_LOG_INFO npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_INFO)
#define ADP_LOG_WARNING npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_WARNING)
#define ADP_LOG_ERROR npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_ERROR)
#define ADP_LOG_EVENT npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_EVENT)
#define ADP_LOG_DEBUG npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_DEBUG)
#define ADP_LOG_FATAL npu::AdapterLogger(__FILE__, __LINE__, npu::ADP_FATAL)

#define ADP_LOG(LEVEL) ADP_LOG_##LEVEL
#endif
