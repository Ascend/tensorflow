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
#include "securec.h"
#include <vector>
#include <string>
#include "tf_adapter/common/adp_logger.h"

namespace tensorflow {
  MemoryPool::MemoryPool() {}

  MemoryPool::~MemoryPool() {}

  Status MemoryPool::MallocMemory(void *&buffer,
                                  uint64_t args_size) {
    MemoryBlock temp_block(nullptr, args_size);
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
      temp_block.ptr = malloc(args_size);
      if (temp_block.ptr == nullptr) {
        ADP_LOG(ERROR) << "rtMalloc host memory failed";
        return errors::InvalidArgument("rtMalloc host memory failed");
      }
    }
    buffer = temp_block.ptr;
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
      free(memory_it->ptr);
      memory_it++;
    }
    memory_list.clear();
    return true;
  }
}