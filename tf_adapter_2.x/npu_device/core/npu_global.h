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

#ifndef WORKSPACE_NPU_GLOBAL_H
#define WORKSPACE_NPU_GLOBAL_H

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"

#include "acl/acl_rt.h"

namespace npu {
namespace global {
// 全局Device循环次数设置
extern std::atomic_int64_t g_npu_loop_size;
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
