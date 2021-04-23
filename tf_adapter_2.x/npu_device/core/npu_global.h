/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: Common depends and micro defines for and only for data preprocess module
 */

#ifndef WORKSPACE_NPU_GLOBAL_H
#define WORKSPACE_NPU_GLOBAL_H

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"

#include "acl/acl_rt.h"

namespace npu {
namespace global {
// 控制Device内存释放的全局读写锁
extern tensorflow::mutex dev_memory_shared_lock;
extern bool dev_memory_released TF_GUARDED_BY(dev_memory_shared_lock);

// Rts ctx管理器
class RtsCtx {
 public:
  static void SetGlobalCtx(aclrtContext global_ctx);
  static tensorflow::Status EnsureInitialized();

 private:
  static aclrtContext global_ctx_;
  static std::atomic_bool global_ctx_set_;
};

}  // namespace global
}  // namespace npu

#endif  // WORKSPACE_NPU_GLOBAL_H
