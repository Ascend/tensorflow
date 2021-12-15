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

#ifndef WORKSPACE_TEST_FUNCTION_LIBRARY_H
#define WORKSPACE_TEST_FUNCTION_LIBRARY_H

#include <mutex>
#include <string>
#include <vector>

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"

class FunctionStrLibrary {
 public:
  static FunctionStrLibrary &Instance() {
    static FunctionStrLibrary library;
    return library;
  }

  void Add(const std::string &readable_str) {
    tensorflow::FunctionDef def;
    CHECK(tensorflow::protobuf::TextFormat::ParseFromString(readable_str, &def));
    std::unique_lock<std::mutex> lk(mu_);
    function_defs_.emplace_back(def.SerializeAsString());
  }

  std::vector<std::string> Get() {
    std::unique_lock<std::mutex> lk(mu_);
    return function_defs_;
  }

 private:
  FunctionStrLibrary() = default;
  ~FunctionStrLibrary() = default;
  std::mutex mu_;
  std::vector<std::string> function_defs_;
};

#define REGISTER_TEST_FUNC(str) REGISTER_TEST_FUNC_1(__COUNTER__, (str))
#define REGISTER_TEST_FUNC_1(ctr, str) REGISTER_TEST_FUNC_2(ctr, (str))
#define REGISTER_TEST_FUNC_2(ctr, str)         \
  static int __registered_func##ctr = []() {   \
    FunctionStrLibrary::Instance().Add((str)); \
    return 0;                                  \
  }();

#endif  // WORKSPACE_TEST_FUNCTION_LIBRARY_H
