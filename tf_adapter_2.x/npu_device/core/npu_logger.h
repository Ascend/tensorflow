/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef TENSORFLOW_NPU_LOGGER_H
#define TENSORFLOW_NPU_LOGGER_H

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"

#include "npu_env.h"

#define DLOG() \
  if (kDumpExecutionDetail) LOG(INFO)

namespace npu {
// TODO:日志适配层，需要对接slog，当前未使用，复用的tensorflow
class Logger : public std::basic_ostringstream<char> {
 public:
  Logger(const char *f, int line) { *this << f << ":" << line << " "; }
  ~Logger() override { std::cerr << str() << std::endl; }
};

class Timer : public std::basic_ostringstream<char> {
 public:
  template <typename... Args>
  explicit Timer(Args... args) {
    *this << tensorflow::strings::StrCat(args...) << " cost ";
  };
  ~Timer() override = default;
  void Start() {
    if (TF_PREDICT_FALSE(kPerfEnabled)) {
      start_ = tensorflow::Env::Default()->NowMicros();
    }
    started_ = true;
  }
  void Stop() {
    if (started_ && TF_PREDICT_FALSE(kPerfEnabled)) {
      *this << (tensorflow::Env::Default()->NowMicros() - start_) / 1000 << " ms";
      LOG(INFO) << str();
    }
    started_ = false;
  }

 private:
  uint64_t start_{0};
  bool started_{false};
};
}  // namespace npu

#endif  // TENSORFLOW_NPU_DEVICE_ACL_BACKENDS_H
