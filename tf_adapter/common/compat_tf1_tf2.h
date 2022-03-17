/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef TENSORFLOW_COMMON_COMPAT_TF1_TF2_H_
#define TENSORFLOW_COMMON_COMPAT_TF1_TF2_H_

#include "tensorflow/core/platform/tstring.h"

namespace npu {
namespace compat_tf1_tf2 {
#ifdef TF_VERSION_TF2
using string = tensorflow::tstring;
#else
using string = tensorflow::string;
#endif
}  // namespace compat_tf1_tf2
}  // namespace npu

#if defined(TF_VERSION_TF2)
#define STATUS_FUNCTION_ONLY_TF2(F)                                                                                    \
  tensorflow::Status F {                                                                                               \
    return tensorflow::Status::OK();                                                                                   \
  }
#else
#define STATUS_FUNCTION_ONLY_TF2(F)
#endif

#if !defined(TF_VERSION_TF2)
#define STATUS_FUNCTION_ONLY_TF1(F)                                                                                    \
  tensorflow::Status F {                                                                                               \
    return tensorflow::Status::OK();                                                                                   \
  }
#else
#define STATUS_FUNCTION_ONLY_TF1(F)
#endif

#if defined(TF_VERSION_TF2)
#define VOID_FUNCTION_ONLY_TF2(F)                                                                                      \
  void F {}
#else
#define VOID_FUNCTION_ONLY_TF2(F)
#endif

#if !defined(TF_VERSION_TF2)
#define VOID_FUNCTION_ONLY_TF1(F)                                                                                      \
  void F {}
#else
#define VOID_FUNCTION_ONLY_TF1(F)
#endif

#endif  // TENSORFLOW_COMMON_COMPAT_TF1_TF2_H_
