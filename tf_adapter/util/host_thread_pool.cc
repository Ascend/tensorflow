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

#include "host_thread_pool.h"
#include "acl/acl_rt.h"
#include "tf_adapter/common/adp_logger.h"

namespace {
  const uint32_t MAX_THREAD_NUM = 4U;
}
namespace tensorflow {
  HostThreadPool::HostThreadPool() : thread_stop_flag_(false) {}

  HostThreadPool::~HostThreadPool() {}

  Status HostThreadPool::Init(uint32_t device_id) {
    ADP_LOG(INFO) << "Start to start thread pool";
    device_id_ = device_id;
    copy_thread_pool_.resize(MAX_THREAD_NUM);
    if (Env::Default() == nullptr) {
      ADP_LOG(ERROR) << "Env default is nullptr.";
      return errors::InvalidArgument("Init memory pool failed");
    }
    for (size_t i = 0UL; i < copy_thread_pool_.size(); i++) {
      if (copy_thread_pool_[i] == nullptr) {
        std::string thread_name = "thread_pool" + std::to_string(i);
        copy_thread_pool_[i].reset(
            Env::Default()->StartThread({}, thread_name, [this]() { ParallelForCopyThread(); }));
      }
    }
    return Status::OK();
  }

  void HostThreadPool::ParallelForCopyThread() {
    ADP_LOG(INFO) << "Start parallel copy thread.";
    auto ret = aclrtSetDevice(device_id_);
    if (ret != ACL_ERROR_NONE) {
      ADP_LOG(ERROR) << "Set device failed, device_id: " << device_id_;
      return;
    }
    std::function<void()> closure;
    while (!thread_stop_flag_.load()) {
      {
        std::unique_lock<std::mutex> lck(queue_lock_);
        queue_var_.wait(lck, [this]() { return ((!task_queue_.empty()) || (thread_stop_flag_.load())); });
        if (thread_stop_flag_.load()) {
          queue_var_.notify_all();
          break;
        }
        closure = task_queue_.front();
        task_queue_.pop();
      }
      closure();
    }
    ADP_LOG(INFO) << "Copy thread is finished";
  }

  void HostThreadPool::PushTask(std::function<void()> closure) {
    std::unique_lock<std::mutex> lck(queue_lock_);
    task_queue_.push(closure);
    queue_var_.notify_one();
  }

  void HostThreadPool::StopThreadPool() {
    thread_stop_flag_.store(true);
    queue_var_.notify_all();
  }
}