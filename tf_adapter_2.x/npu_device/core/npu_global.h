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

#ifndef NPU_DEVICE_CORE_NPU_GLOBAL_H
#define NPU_DEVICE_CORE_NPU_GLOBAL_H

#include <map>
#include <vector>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"

#include "acl/acl_rt.h"

namespace npu {
class NpuDevice;
class HdcChannel;
namespace global {
// 全局Device循环次数设置
extern std::atomic_int64_t g_npu_loop_size;
// 全局NPU自定义OP
extern std::unordered_set<std::string> g_npu_specify_ops;
// 控制Device内存释放的全局读写锁
extern tensorflow::mutex dev_memory_shared_lock;
extern bool dev_memory_released TF_GUARDED_BY(dev_memory_shared_lock);

// Rts ctx管理器
class RtsCtx {
 public:
  static tensorflow::Status CreateGlobalCtx(int32_t device_index);
  static tensorflow::Status EnsureInitialized();
  static tensorflow::Status DestroyGlobalCtx();
 private:
  static aclrtContext global_ctx_;
  static tensorflow::mutex global_ctx_mutex_;
};

class NpuCtx {
 public:
  static void SetDeviceCtx(int id, TFE_Context *ctx, NpuDevice *device);
  static tensorflow::Status GetDeviceCtx(int id, TFE_Context **ctx, NpuDevice **device);
  struct Ctx {
    TFE_Context *ctx;
    NpuDevice *device;
  };

 private:
  static std::map<int, NpuCtx::Ctx> npu_ctx_;
};

class GlobalHdcChannel {
 public:
  static GlobalHdcChannel &GetInstance() {
    static GlobalHdcChannel Instance;
    return Instance;
  }

  void Get(const std::string &name, std::vector<std::shared_ptr<npu::HdcChannel>> &channels);

  tensorflow::Status Create(const std::string &name, int64_t channel_capacity, const std::vector<int> &device_ids);

  void Destroy(const std::string &name);

 private:
  std::map<std::string, std::vector<std::shared_ptr<npu::HdcChannel>>> global_channels_;
  std::mutex global_channels_mu_;
};

}  // namespace global
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_GLOBAL_H
