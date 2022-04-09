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

#include "npu_global.h"

#include "tensorflow/core/util/env_var.h"

#include "npu_logger.h"
#include "npu_micros.h"

namespace npu {
namespace global {
std::atomic_int64_t g_npu_loop_size{[]() -> int64_t {
  tensorflow::int64 loop_size = 1;
  tensorflow::ReadInt64FromEnvVar("NPU_LOOP_SIZE", 1, &loop_size);
  if (loop_size <= 0) {
    LOG(ERROR) << "Npu loop size must be greater than 0, got " << loop_size << " set by env 'NPU_LOOP_SIZE'";
    return 1;
  }
  return loop_size;
}()};

std::unordered_set<std::string> g_npu_specify_ops;

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
  static thread_local bool already_set{false};
  if (!already_set && global_ctx_set_) {
    NPU_REQUIRES_ACL_OK("Acl set current thread ctx failed", aclrtSetCurrentContext(global_ctx_));
    already_set = true;
  }
  return tensorflow::Status::OK();
}

aclrtContext RtsCtx::global_ctx_{nullptr};
std::atomic_bool RtsCtx::global_ctx_set_{false};

std::map<int, NpuCtx::Ctx> NpuCtx::npu_ctx_;

void NpuCtx::SetDeviceCtx(int id, TFE_Context *ctx, NpuDevice *device) {
  auto &npu_ctx = npu_ctx_[id];
  npu_ctx.ctx = ctx;
  npu_ctx.device = device;
}
tensorflow::Status NpuCtx::GetDeviceCtx(int id, TFE_Context **ctx, NpuDevice **device) {
  auto iter = npu_ctx_.find(id);
  NPU_REQUIRES(iter != npu_ctx_.end(),
               tensorflow::errors::Internal("Device instance on device ", id, " has not been created"));
  *ctx = iter->second.ctx;
  *device = iter->second.device;
  return tensorflow::Status::OK();
}
}  // namespace global
}  // namespace npu