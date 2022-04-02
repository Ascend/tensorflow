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

#include "memory_pool.h"
#include "host_allocator.h"
#include "securec.h"
#include <vector>
#include <string>
#include "acl/acl_rt.h"
#include "tf_adapter/common/adp_logger.h"

namespace {
  const uint32_t MAX_THREAD_NUM = 4U;
  const uint64_t PARALLEL_MEMORY_TRHESHOLD = 10 * 1024 * 1024;
  std::atomic<uint32_t> event_num(0U);
}
namespace tensorflow {
  MemoryPool::MemoryPool() : thread_stop_flag_(false) {}
  Status MemoryPool::Init(uint32_t device_id) {
    device_id_ = device_id;
    ADP_LOG(INFO) << "Start to start thread pool";
    copy_thread_pool_.resize(MAX_THREAD_NUM);
    if (Env::Default() == nullptr) {
      ADP_LOG(ERROR) << "Env default is nullptr.";
      return errors::InvalidArgument("Init memory pool failed");
    }
    for (size_t i = 0UL; i < copy_thread_pool_.size(); i++) {
      if (!copy_thread_pool_[i]) {
        std::string thread_name = "thread_pool" + std::to_string(i);
        copy_thread_pool_[i].reset(
            Env::Default()->StartThread({}, thread_name, [this]() { ParallelForCopyThread(); }));
      }
    }
    return Status::OK();
  }

  MemoryPool::~MemoryPool() {}

  void MemoryPool::ParallelForCopyThread() {
    ADP_LOG(INFO) << "Start parallel copy thread.";
    auto ret = aclrtSetDevice(device_id_);
    if (ret != ACL_ERROR_NONE) {
      ADP_LOG(ERROR) << "Set device failed, device_id: " << device_id_;
      return;
    }
    std::function<void()> closure;
    while (!thread_stop_flag_) {
      {
        std::unique_lock<std::mutex> lck(queue_lock_);
        queue_var_.wait(lck, [this]() { return (!task_queue_.empty()); });
        closure = task_queue_.front();
        task_queue_.pop();
        queue_var_.notify_all();
      }
      closure();
      event_num++;
    }
    ADP_LOG(INFO) << "Copy thread is finished";
  }

  void MemoryPool::ParallelCopy(void *dst_ptr, uint64_t dst_size,
                                const char *src_ptr, uint64_t src_size) {
    event_num.store(0U);
    uint64_t block_size = (src_size / MAX_THREAD_NUM);
    uint64_t remain_size = (src_size % MAX_THREAD_NUM);
    uint64_t start = 0UL;
    uint64_t end = 0UL;
    std::function<void()> closure;
    for (uint32_t i = 0U; i < MAX_THREAD_NUM; i++) {
      start = i * block_size;
      end = (i + 1) * block_size;
      if (i == MAX_THREAD_NUM - 1) {
        end += remain_size;
      }
      closure = [start, end, &dst_ptr, &dst_size, &src_ptr, &src_size] () {
        void *dst = dst_ptr + start;
        const char *src = src_ptr + start;
        int64_t len = end - start;
        if (len < 0) {
          ADP_LOG(ERROR) << "Len is less than zero, len:" << len;
          return;
        }
        auto ret = aclrtMemcpy(dst, dst_size - start, src, len, ACL_MEMCPY_HOST_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
          ADP_LOG(ERROR) << "Memcpy failed, start:" << start << ", len: " << len;
          return;
        }
      };
      task_queue_.push(closure);
    }
    while ((event_num.load() < MAX_THREAD_NUM) && (!thread_stop_flag_)) {}
  }

  bool MemoryPool::CopyTensor(void *memory_ptr, uint64_t memory_size,
                              std::vector<Tensor> &args, std::vector<Tensor> &result_args) {
    uint64_t offset = 0ULL;
    for (size_t i = 0UL; i < args.size(); i++) {
      const char *src_ptr = args[i].tensor_data().data();
      uint64_t src_size = args[i].tensor_data().size();
      void *dst_ptr = memory_ptr + offset;
      uint64_t dst_size = memory_size - offset;
      if (src_size > PARALLEL_MEMORY_TRHESHOLD) {
        ParallelCopy(dst_ptr, dst_size, src_ptr, src_size);
      } else {
        auto ret = aclrtMemcpy(dst_ptr, dst_size, src_ptr, src_size, ACL_MEMCPY_HOST_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
          ADP_LOG(ERROR) << "Memcpy failed, dst_size:" << dst_size << ", src_size: " << src_size;
          return false;
        }
      }
      HostAllocator *a = new (std::nothrow) HostAllocator(dst_ptr);
      if (a == nullptr) {
        ADP_LOG(ERROR) << "Allocator memory failed";
        return false;
      }
      Tensor temp_tensor(a, args[i].dtype(), args[i].shape());
      result_args.emplace_back(std::move(temp_tensor));
      offset += src_size;
    }
    return true;
  }

  Status MemoryPool::MallocMemory(std::vector<Tensor> &args, std::vector<Tensor> &result_args, uint64_t args_size) {
    MemoryBlock temp_block(nullptr, args_size);
    bool queue_is_empty = false;
    {
      std::lock_guard<std::mutex> lck(memory_pool_lock_);
      auto free_it = free_memory_list_.begin();
      while (free_it != free_memory_list_.end()) {
        if (free_it->data_size >= args_size) {
          temp_block = (*free_it);
          free_it = free_memory_list_.erase(free_it);
          break;
        }
        free_it++;
      }
      if ((temp_block.ptr == nullptr) && (!free_memory_list_.empty())) {
        if (!FreeMemoryList(free_memory_list_)) {
          ADP_LOG(ERROR) << "Release free host memory failed";
          return errors::InvalidArgument("Release free host memory failed");
        }
      }
    }

    if (temp_block.ptr == nullptr) {
      auto ret = aclrtMallocHost(&temp_block.ptr, args_size);
      if (ret != ACL_ERROR_NONE) {
        ADP_LOG(ERROR) << "rtMalloc host memory failed";
        return errors::InvalidArgument("rtMalloc host memory failed");
      }
    }

    if (!CopyTensor(temp_block.ptr, args_size, args, result_args)) {
      ADP_LOG(ERROR) << "Copy tensor to memory failed";
      std::lock_guard<std::mutex> lck(memory_pool_lock_);
      free_memory_list_.push_back(temp_block);
      return errors::InvalidArgument("Copy tensor to memory failed");
    }
    std::lock_guard<std::mutex> lck(memory_pool_lock_);
    used_memory_list_.push_back(temp_block);
    return Status::OK();
  }

  void MemoryPool::ReleaseMemory() {
    std::lock_guard<std::mutex> lck(memory_pool_lock_);
    if (used_memory_list_.empty()) {
      return;
    }
    MemoryBlock head = used_memory_list_.front();
    used_memory_list_.pop_front();
    free_memory_list_.push_back(head);
  }

  Status MemoryPool::FreeAllMemory() {
    thread_stop_flag_ = true;
    std::lock_guard<std::mutex> lck(memory_pool_lock_);
    if ((!FreeMemoryList(free_memory_list_)) || !FreeMemoryList(used_memory_list_)) {
      ADP_LOG(ERROR) << "Release host memory pool failed";
      return errors::InvalidArgument("Release host memory pool failed");
    }
    return Status::OK();
  }

  bool MemoryPool::FreeMemoryList(std::list<MemoryBlock> &memory_list) {
    auto memory_it = memory_list.begin();
    while (memory_it != memory_list.end()) {
      auto ret = aclrtFreeHost(memory_it->ptr);
      if (ret != ACL_ERROR_NONE) {
        ADP_LOG(ERROR) << "rtFree host memory failed";
        return false;
      }
      memory_it++;
    }
    memory_list.clear();
    return true;
  }
}