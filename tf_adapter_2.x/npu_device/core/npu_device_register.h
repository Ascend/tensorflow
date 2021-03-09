/**
* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
* Description: Common depends and micro defines for and only for data preprocess module
*/

#ifndef TENSORFLOW_NPU_DEVICE_REGISTER_H_
#define TENSORFLOW_NPU_DEVICE_REGISTER_H_

#include "tensorflow/c/eager/c_api.h"
#include <map>
#include <string>

std::string CreateDevice(TFE_Context *context, const char *device_name, int device_index,
                 const std::map<std::string, std::string> &session_options);

void ReleaseDeviceResource();

#endif  // TENSORFLOW_C_EAGER_NPU_DEVICE_TESTUTIL_H_
