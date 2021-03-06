/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef TENSORFLOW_COMMON_COMMON_H_
#define TENSORFLOW_COMMON_COMMON_H_

#include "tensorflow/core/platform/env.h"
#include "tf_adapter/common/adp_logger.h"

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
    return errors::InvalidArgument(#v " is nullptr.");                                                                 \
  }

#define REQUIRES_STATUS_OK(s)                                                                                          \
  if (!s.ok()) { return s; }

#define ADAPTER_ENV_MAX_LENTH 1024 * 1024
#endif  // TENSORFLOW_COMMON_COMMON_H_
