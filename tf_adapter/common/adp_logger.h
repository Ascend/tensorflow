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

#ifndef TENSORFLOW_ADP_LOGGER_H
#define TENSORFLOW_ADP_LOGGER_H

#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>

#define FMK_MODULE_NAME static_cast<int>(FMK)
#define ADP_MODULE_NAME "TF_ADAPTER"

const int ADP_DEBUG = 0;
const int ADP_INFO = 1;
const int ADP_WARNING = 2;
const int ADP_ERROR = 3;
const int ADP_EVENT = 16;
const int ADP_FATAL = 32;


class AdapterLogger : public std::basic_ostringstream<char> {
 public:
  AdapterLogger(const char* fname, int line, int severity) {
    *this << " [" << fname << ":" << line << "]" << " ";
    severity_ = severity;
  }
  ~AdapterLogger();

 private:
  int severity_;
};


#define _ADP_LOG_INFO AdapterLogger(__FILE__, __LINE__, ADP_INFO)
#define _ADP_LOG_WARNING AdapterLogger(__FILE__, __LINE__, ADP_WARNING)
#define _ADP_LOG_ERROR AdapterLogger(__FILE__, __LINE__, ADP_ERROR)
#define _ADP_LOG_EVENT AdapterLogger(__FILE__, __LINE__, ADP_EVENT)
#define _ADP_LOG_DEBUG AdapterLogger(__FILE__, __LINE__, ADP_DEBUG)
#define _ADP_LOG_FATAL AdapterLogger(__FILE__, __LINE__, ADP_FATAL)

#define ADP_LOG(LEVEL) _ADP_LOG_##LEVEL
#endif