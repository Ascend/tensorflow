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
  static NpuAllocator* CreateNpuAllocator(void *addr, std::function<void(void *)> del) {
    return new (std::nothrow)NpuAllocator(kNpuAllocatorName, addr, del);
  }

  static NpuAllocator* CreateCpuAllocator(void *addr, std::function<void(void *)> del) {
    return new (std::nothrow)NpuAllocator(kCpuAllocatorName, addr, del);
  }

  ~NpuAllocator() override {
    delete_(addr_);
  }

  std::string Name() override {
    return kNpuAllocatorName;
  }

  static bool IsNpuAllocator(std::string name) {
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

  static int64_t GetAlignment() { return kAllocatorAlignment; }
  static int64_t AlignSize(int64_t size) { return ((size + kAllocatorAlignment - 1) / kAllocatorAlignment) * kAllocatorAlignment; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    (void)alignment;
    (void)num_bytes;
    return addr_;
  }

  void DeallocateRaw(void* ptr) override { delete this; }

 private:
  explicit NpuAllocator(std::string name, void *addr, std::function<void(void *)> del)
      : name_(name),
        addr_(addr),
        delete_(del) {
    ADP_LOG(INFO) << "NpuAllocator: name = " << name << ", addr = " << addr;
  };
  const std::string name_;
  void *addr_;
  std::function<void(void *)> delete_;
  static constexpr char *kNpuAllocatorName = "NpuAllocator";
  static constexpr char *kCpuAllocatorName = "CpuAllocator";
};
}  // namespace data
}  // namespace tensorflow
#endif // TENSORFLOW_CORE_KERNELS_NPU_TENSOR_H_