/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: Common depends and micro defines for and only for data preprocess module
 */

#include "tensorflow/core/platform/mutex.h"

#include "npu_global.h"
#include "npu_logger.h"
#include "npu_micros.h"

namespace npu {
namespace global {
tensorflow::mutex dev_memory_shared_lock;
bool dev_memory_released{false};

void RtsCtx::SetGlobalCtx(aclrtContext global_ctx) {
  static std::atomic_bool already_set{false};
  if (!already_set.exchange(true)) {
    global_ctx_ = global_ctx;
    global_ctx_set_ = true;
  }
}

// 存在rtMalloc和rtFree在不同线程操作的情况，这里保证全局唯一的ctx，因而不保证任何在NPU初始化完成前对rts的接口调用成功
tensorflow::Status RtsCtx::EnsureInitialized() {
  if (global_ctx_set_) {
    NPU_REQUIRES_ACL_OK("Acl set current thread ctx failed", aclrtSetCurrentContext(global_ctx_));
  }
  return tensorflow::Status::OK();
}

aclrtContext RtsCtx::global_ctx_{nullptr};
std::atomic_bool RtsCtx::global_ctx_set_{false};
}  // namespace global
}  // namespace npu