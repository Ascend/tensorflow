/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: Common depends and micro defines for and only for data preprocess module
 */

#include "tensorflow/core/platform/mutex.h"

namespace npu {
namespace global {
tensorflow::mutex dev_memory_shared_lock;
bool dev_memory_released{false};
}  // namespace global
}  // namespace npu