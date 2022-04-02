/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef KERNELS_UTIL_HOST_THREAD_POOL_H_
#define KERNELS_UTIL_HOST_THREAD_POOL_H_

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <functional>
#include <condition_variable>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
class HostThreadPool {
 public:
  HostThreadPool();
  Status Init(uint32_t device_id);
  void PushTask(std::function<void()> closure);
  void StopThreadPool();
  ~HostThreadPool();
 private:
  void ParallelForCopyThread();
  std::mutex queue_lock_;
  std::condition_variable queue_var_;
  std::vector<std::unique_ptr<Thread>> copy_thread_pool_;
  std::queue<std::function<void()>> task_queue_;
  std::atomic<bool> thread_stop_flag_;
  uint32_t device_id_;
};
}
#endif