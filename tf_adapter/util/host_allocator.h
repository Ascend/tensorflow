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

#ifndef TENSORFLOW_HOST_ALLOCATOR_H_
#define TENSORFLOW_HOST_ALLOCATOR_H_

#include "tensorflow/core/framework/allocator.h"
#include <string>

namespace tensorflow {
  class HostAllocator : public Allocator {
  public:
    explicit HostAllocator(void *addr);
    ~HostAllocator() override;
    std::string Name() override;
    void *AllocateRaw(size_t alignment, size_t num_bytes) override;
    void *AllocateRaw(size_t alignment, size_t num_bytes,
                      const AllocationAttributes &allocation_attr) override;
    void DeallocateRaw(void *ptr) override;
  private:
    void *addr_;
  };
}
#endif