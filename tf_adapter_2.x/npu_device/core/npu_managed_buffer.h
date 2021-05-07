/* Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_NPU_MANAGED_BUFFER_H
#define TENSORFLOW_NPU_MANAGED_BUFFER_H

#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"

#include "graph/types.h"

class NpuManagedBuffer {
 public:
  static void Destroy(NpuManagedBuffer *buf);

  static tensorflow::Status Create(ge::Format fmt, const tensorflow::TensorShape &shape, tensorflow::DataType dtype,
                                   NpuManagedBuffer **buf);
  static tensorflow::Status Create(ge::Format format, const std::vector<int64_t> &shape, ge::DataType data_type,
                                   NpuManagedBuffer **buf);
  static tensorflow::Status Create(ge::Format format, const std::vector<int64_t> &shape, ge::DataType data_type,
                                   ge::Format origin_format, const std::vector<int64_t> &origin_shape,
                                   NpuManagedBuffer **buf);
  static tensorflow::Status Create(ge::Format format, const std::vector<int64_t> &shape, ge::DataType data_type,
                                   ge::Format origin_format, const std::vector<int64_t> &origin_shape, void *addr,
                                   size_t size, void *arg, void (*deallocator)(void *, size_t, void *),
                                   NpuManagedBuffer **buf);

  // 将输入的CPU Tensor的数据填充到当前buffer管理的NPU内存上，CPU
  // Tensor的格式和type与buffer的成员origin_data_type_和origin_format_一致
  tensorflow::Status AssembleFrom(const tensorflow::Tensor *tensor);

  // 将当前buffer管理的NPU内存上的数据填充到输入的CPU Tensor的数据地址上，CPU
  // Tensor的格式和type与buffer的成员origin_data_type_和origin_format_一致
  tensorflow::Status AssembleTo(const tensorflow::Tensor *tensor);

  bool SameRepresentation() { return origin_format_ == format_ && origin_data_type_ == data_type_; }

  std::string DebugString() const;

  class Guarder {
   public:
    explicit Guarder(NpuManagedBuffer *buf) : buf_(buf) {}
    ~Guarder() { NpuManagedBuffer::Destroy(buf_); }

   private:
    NpuManagedBuffer *buf_;
  };

 private:
  NpuManagedBuffer() = default;
  ~NpuManagedBuffer();
  tensorflow::Status TransRepresentationOnNpu(NpuManagedBuffer *dst_buff);  // 在NPU上完成从存储到原始的格式和类型转换
  tensorflow::Status HToD(void *host_data, size_t size);  // 将输入的Host内存搬运到管理的NPU内存上
  tensorflow::Status DToH(void *host_data, size_t max_len);  // 将管理的NPU内存上的数据搬运到输入的Host内存上

  ge::DataType origin_data_type_{};  // 原始数据类型，即对应的CPU Tensor的数据类型
  ge::Format origin_format_{};  // 原始内存排布，即对应的CPU Tensor的维度信息，一般都是ND，可能是NCHW或者NHWC
  std::vector<int64_t> origin_shape_;  // 原始维度信息，即对应的CPU Tensor的原始维度
  ge::DataType data_type_{};           // 在NPU上的存储数据类型
  ge::Format format_{};                // 在NPU上的存储格式
  std::vector<int64_t> shape_;         // 对应NPU上的存储格式的维度值

  size_t size_{};                                  // NPU上占用的内存大小
  void *data_{};                                   // NPU地址指针
  void (*deallocator_)(void *, size_t, void *){};  // NP内存的释放函数，内存可能会来自于内存池或者rtMalloc
  void *deallocator_arg_{};                        // 地址释放时传给释放函数的参数
};

// NpuManagedBuffer是Host的对象，是CPU Tensor管理的对象，是NPU内存的Host句柄，应当在析构函数中释放NPU内存
static void NpuManagedBufferDeallocator(void *data, size_t len, void *arg) {
  NpuManagedBuffer::Destroy(reinterpret_cast<NpuManagedBuffer *>(data));
}

#endif  // TENSORFLOW_NPU_TENSOR_H