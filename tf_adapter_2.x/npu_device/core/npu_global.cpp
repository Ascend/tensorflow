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
#include "npu_hdc.h"

namespace npu {
namespace global {
std::atomic_int64_t g_npu_loop_size{[]() -> int64_t {
  tensorflow::int64 loop_size = 1;
  (void)tensorflow::ReadInt64FromEnvVar("NPU_LOOP_SIZE", 1, &loop_size);
  if (loop_size <= 0) {
    LOG(ERROR) << "Npu loop size must be greater than 0, got " << loop_size << " set by env 'NPU_LOOP_SIZE'";
    return 1;
  }
  return loop_size;
}()};

std::unordered_set<std::string> g_npu_specify_ops;

tensorflow::mutex dev_memory_shared_lock;
bool dev_memory_released = false;

void RtsCtx::SetGlobalCtx(aclrtContext global_ctx) {
  static std::atomic_bool already_set{false};
  if (!already_set.exchange(true)) {
    global_ctx_ = global_ctx;
    global_ctx_set_ = true;
  }
}

// 存在rtMalloc和rtFree在不同线程操作的情况，也存在同一线程会切换context的场景
// 这里保证全局唯一的ctx，且对device资源操作时都设置这个全局ctx
tensorflow::Status RtsCtx::EnsureInitialized() {
  if (global_ctx_set_) {
    NPU_REQUIRES_ACL_OK("Acl set current thread ctx failed", aclrtSetCurrentContext(global_ctx_));
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
  const decltype(npu_ctx_)::const_iterator iter = npu_ctx_.find(id);
  NPU_REQUIRES(iter != npu_ctx_.cend(),
               tensorflow::errors::Internal("Device instance on device ", id, " has not been created"));
  *ctx = iter->second.ctx;
  *device = iter->second.device;
  return tensorflow::Status::OK();
}

void GlobalHdcChannel::Get(const std::string &name, std::vector<std::shared_ptr<npu::HdcChannel>> &channels) {
  std::unique_lock<std::mutex> lk(global_channels_mu_);
  auto iter = global_channels_.find(name);
  if (iter != global_channels_.end()) {
    channels = iter->second;
  }
}

tensorflow::Status GlobalHdcChannel::Create(const std::string &name, int64_t channel_capacity,
                                            const std::vector<int> &device_ids) {
  const int64_t kInvalidCpacity = -1L;
  if (channel_capacity == kInvalidCpacity) {
    return tensorflow::errors::Internal("Overflow, tensor size exceeds the maximum value of uint64.");
  }
  std::vector<std::shared_ptr<npu::HdcChannel>> channels;
  channels.resize(device_ids.size());
  uint32_t count = 0U;
  for (size_t i = 0UL; i < device_ids.size(); i++) {
    if (!npu::HdcChannel::Create(static_cast<uint32_t>(device_ids[i]), name, static_cast<size_t>(channel_capacity),
                                 &channels[i])
           .ok()) {
      break;
    }
    count++;
  }
  if (count == device_ids.size()) {
    DLOG() << "Create hdc channel with capacity success.";
  } else if (count == 0U) {
    DLOG() << "Current version not support create hdc channel with capacity by acl.";
    return tensorflow::Status::OK();
  } else {
    return tensorflow::errors::Internal("Failed create hdc channel with capacity.");
  }
  std::unique_lock<std::mutex> lk(global_channels_mu_);
  (void)global_channels_.insert(std::make_pair(name, channels));
  return tensorflow::Status::OK();
}

void GlobalHdcChannel::Destroy(const std::string &name) {
  std::unique_lock<std::mutex> lk(global_channels_mu_);
  auto iter = global_channels_.find(name);
  if (iter != global_channels_.end()) {
    for (const auto &channel : iter->second) {
      channel->Destroy();
    }
    (void)global_channels_.erase(name);
  }
}
}  // namespace global
}  // namespace npu