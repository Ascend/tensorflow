/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#include <cstdint>
#include <thread>
#include <mutex>
#include <functional>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor.h"

#ifndef TENSORFLOW_CORE_KERNELS_NPU_TENSOR_H_
#define TENSORFLOW_CORE_KERNELS_NPU_TENSOR_H_
namespace tensorflow {
namespace data {
class NpuAllocator : public Allocator {
public:
  static NpuAllocator* CreateNpuAllocator(void *addr, const std::function<void(void *)> del) {
    return new (std::nothrow)NpuAllocator(kNpuAllocatorName, addr, del);
  }

  static NpuAllocator* CreateCpuAllocator(void *addr, const std::function<void(void *)> del) {
    return new (std::nothrow)NpuAllocator(kCpuAllocatorName, addr, del);
  }

  ~NpuAllocator() override {
    delete_(addr_);
  }

  std::string Name() override {
    return kNpuAllocatorName;
  }

  static bool IsNpuAllocator(const std::string name) {
    return (name.compare(kNpuAllocatorName) == 0) ||
        (name.compare(kCpuAllocatorName) == 0);
  }

  static bool IsNpuAllocator(Tensor &tensor) {
    TensorDescription tensorDesc;
    tensor.FillDescription(&tensorDesc);
    if (tensorDesc.has_allocation_description()) {
      return IsNpuAllocator(tensorDesc.allocation_description().allocator_name());
    }
    return false;
  }

  static uint64_t GetAlignment() { return static_cast<uint64_t>(kAllocatorAlignment); }
  static uint64_t AlignSize(uint64_t size) {
    uint64_t alignment = static_cast<uint64_t>(kAllocatorAlignment);
    return ((size + alignment - 1) / alignment) * alignment;
  }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    (void)alignment;
    (void)num_bytes;
    return addr_;
  }

  void DeallocateRaw(void* ptr) override {
    (void)ptr;
    delete this;
  }

private:
  explicit NpuAllocator(const std::string name, void *addr, const std::function<void(void *)> del)
      : name_(name),
        addr_(addr),
        delete_(del) {
    ADP_LOG(INFO) << "NpuAllocator: name = " << name;
  };
  const std::string name_;
  void *addr_;
  std::function<void(void *)> delete_;
  static constexpr const char* const kNpuAllocatorName = "NpuAllocator";
  static constexpr const char* const kCpuAllocatorName = "CpuAllocator";
};
}  // namespace data
}  // namespace tensorflow
#endif // TENSORFLOW_CORE_KERNELS_NPU_TENSOR_H_