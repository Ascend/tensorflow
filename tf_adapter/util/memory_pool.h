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

#ifndef KERNELS_UTIL_MEMORY_POOL_H_
#define KERNELS_UTIL_MEMORY_POOL_H_

#include <cstdlib>
#include <cstdint>
#include <memory>
#include <atomic>
#include <list>
#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
struct MemoryBlock {
  void *ptr;
  uint64_t data_size;
  MemoryBlock(void *in_ptr, uint64_t in_size) : ptr(in_ptr), data_size(in_size) {}
};

class MemoryPool {
 public:
  MemoryPool();
  Status MallocMemory(void *&buffer,
                      uint64_t args_size);
  void ReleaseMemory();
  Status FreeAllMemory();
  ~MemoryPool();
 private:
  bool FreeMemoryList(std::list<MemoryBlock> &memory_list);
  std::mutex memory_pool_lock_;
  std::list<MemoryBlock> used_memory_list_;
  std::list<MemoryBlock> free_memory_list_;
};
}
#endif