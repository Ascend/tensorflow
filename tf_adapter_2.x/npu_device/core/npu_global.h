/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: Common depends and micro defines for and only for data preprocess module
 */

#ifndef WORKSPACE_NPU_GLOBAL_H
#define WORKSPACE_NPU_GLOBAL_H

#include "tensorflow/core/platform/mutex.h"

namespace npu {
namespace global {
// 控制Device内存释放的全局读写锁
extern tensorflow::mutex dev_memory_shared_lock;
extern bool dev_memory_released TF_GUARDED_BY(dev_memory_shared_lock);
}  // namespace global
}  // namespace npu

#endif  // WORKSPACE_NPU_GLOBAL_H
