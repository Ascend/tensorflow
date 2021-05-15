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

#include "npu_global.h"
#include "npu_managed_buffer.h"
#include "npu_logger.h"
#include "npu_micros.h"
#include "npu_utils.h"

#include "acl/acl_op_compiler.h"
#include "acl/acl_rt.h"

#include "tensorflow/core/common_runtime/dma_helper.h"

namespace {
class NpuMemory {
 public:
  static tensorflow::Status Malloc(size_t size, void **memory) {
    if (size == 0) {
      *memory = nullptr;
      return tensorflow::Status::OK();
    }
    NPU_REQUIRES_ACL_OK("Malloc npu memory failed for size " + std::to_string(size),
                        aclrtMalloc(memory, size, ACL_MEM_MALLOC_HUGE_FIRST));
    npu_memory_usage_ += size;
    DLOG() << "Malloced npu memory " << reinterpret_cast<uintptr_t>(*memory) << ", size " << size << ", usage "
           << npu_memory_usage_;
    return tensorflow::Status::OK();
  }
  static void Free(void *memory, size_t size, void *arg) {
    npu::global::dev_memory_shared_lock.lock_shared();
    if (!npu::global::dev_memory_released) {
      if (aclrtFree(memory) == ACL_ERROR_NONE) {
        npu_memory_usage_ -= size;
        DLOG() << "Freed npu memory " << reinterpret_cast<uintptr_t>(memory) << ", size " << size << ", usage "
               << npu_memory_usage_;
      } else {
        LOG(ERROR) << "Failed to free npu memory " << reinterpret_cast<uintptr_t>(memory) << ", size " << size
                   << ", usage " << npu_memory_usage_;
      }
    } else {
      DLOG() << "Skipped free npu memory " << reinterpret_cast<uintptr_t>(memory) << " as device reset, size " << size
             << ", usage " << npu_memory_usage_;
    }
    npu::global::dev_memory_shared_lock.unlock_shared();
  }

 private:
  static std::atomic_int64_t npu_memory_usage_;
};

std::atomic_int64_t NpuMemory::npu_memory_usage_{0};

class RtsStreamGuard {
 public:
  explicit RtsStreamGuard(aclrtStream stream) : stream_(stream) {}
  ~RtsStreamGuard() {
    if (stream_ != nullptr) {
      aclrtDestroyStream(stream_);
      stream_ = nullptr;
    }
  }

 private:
  aclrtStream stream_;
};

tensorflow::Status CreateAclTensorDesc(ge::DataType dtype, ge::Format format, const std::vector<int64_t> &shape,
                                       std::shared_ptr<aclTensorDesc> *desc) {
  aclDataType acl_dtype;
  aclFormat acl_format;
  NPU_REQUIRES_OK(MapGeType2Acl(dtype, &acl_dtype));
  NPU_REQUIRES_OK(MapGeFormat2Acl(format, &acl_format));
  aclTensorDesc *acl_desc = aclCreateTensorDesc(acl_dtype, shape.size(), shape.data(), acl_format);
  NPU_REQUIRES(acl_desc != nullptr, tensorflow::errors::Internal("Failed create acl tensor desc"));
  desc->reset(acl_desc, [](aclTensorDesc *desc) { aclDestroyTensorDesc(desc); });
  return tensorflow::Status::OK();
}

tensorflow::Status CreateAclDataBuffer(void *data, size_t size, std::shared_ptr<aclDataBuffer> *buf) {
  aclDataBuffer *acl_buf = aclCreateDataBuffer(data, size);
  NPU_REQUIRES(acl_buf != nullptr, tensorflow::errors::Internal("Failed create acl data buffer"));
  buf->reset(acl_buf, [](aclDataBuffer *buf) { aclDestroyDataBuffer(buf); });
  return tensorflow::Status::OK();
}

tensorflow::Status CreateTransFormatAttr(ge::Format src, ge::Format dst, std::shared_ptr<aclopAttr> *attr) {
  aclopAttr *acl_attr = aclopCreateAttr();
  NPU_REQUIRES(acl_attr != nullptr, tensorflow::errors::Internal("Failed create acl op attr"));
  attr->reset(acl_attr, [](aclopAttr *attr) { aclopDestroyAttr(attr); });

  NPU_REQUIRES_ACL_OK("Acl set op attr src_format failed",
                      aclopSetAttrString(acl_attr, "src_format", GetFormatName(src)));

  NPU_REQUIRES_ACL_OK("Acl set op attr dst_format failed",
                      aclopSetAttrString(acl_attr, "dst_format", GetFormatName(dst)));
  return tensorflow::Status::OK();
}

tensorflow::Status CreateCastDtypeAttr(ge::DataType src, ge::DataType dst, std::shared_ptr<aclopAttr> *attr) {
  aclopAttr *acl_attr = aclopCreateAttr();
  NPU_REQUIRES(acl_attr != nullptr, tensorflow::errors::Internal(""));
  attr->reset(acl_attr, [](aclopAttr *attr) { aclopDestroyAttr(attr); });

  NPU_REQUIRES_ACL_OK("Acl set op attr dst_type failed",
                      aclopSetAttrInt(acl_attr, "dst_type", static_cast<int32_t>(dst)));
  return tensorflow::Status::OK();
}

tensorflow::Status ScheduleCastDtypeTask(aclrtStream stream, ge::Format format, const std::vector<int64_t> &shape,
                                         ge::DataType src_dt, ge::DataType dst_dt, void *src_data, void *dst_data,
                                         size_t src_len, size_t dst_len) {
  // TODO: 在一些cube格式的极端场景下，data type转换后，shape也会跟着转，这里暂时没有考虑这种场景
  std::shared_ptr<aclTensorDesc> input_desc;
  NPU_REQUIRES_OK(CreateAclTensorDesc(src_dt, format, shape, &input_desc));
  aclTensorDesc *input_descs[] = {input_desc.get()};

  std::shared_ptr<aclDataBuffer> input_data;
  NPU_REQUIRES_OK(CreateAclDataBuffer(src_data, src_len, &input_data));
  aclDataBuffer *input_dbs[] = {input_data.get()};

  std::shared_ptr<aclTensorDesc> output_desc;
  NPU_REQUIRES_OK(CreateAclTensorDesc(dst_dt, format, shape, &output_desc));
  aclTensorDesc *output_ds[] = {output_desc.get()};

  std::shared_ptr<aclDataBuffer> output_data;
  NPU_REQUIRES_OK(CreateAclDataBuffer(dst_data, dst_len, &output_data));
  aclDataBuffer *output_dbs[] = {output_data.get()};

  std::shared_ptr<aclopAttr> attr;
  NPU_REQUIRES_OK(CreateCastDtypeAttr(src_dt, dst_dt, &attr));
  NPU_REQUIRES_ACL_OK("Acl compile and execute \'Cast\' op failed",
                      aclopCompileAndExecute("Cast", 1, input_descs, input_dbs, 1, output_ds, output_dbs, attr.get(),
                                             ACL_ENGINE_AICORE, ACL_COMPILE_SYS, nullptr, stream));
  return tensorflow::Status::OK();
}

tensorflow::Status ScheduleTransFormatTask(aclrtStream stream, ge::DataType src_dt, ge::Format src_format,
                                           const std::vector<int64_t> &src_shape, ge::Format dst_format,
                                           const std::vector<int64_t> &dst_shape, void *src_data, void *dst_data,
                                           size_t src_len, size_t dst_len) {
  std::shared_ptr<aclTensorDesc> input_desc;
  NPU_REQUIRES_OK(CreateAclTensorDesc(src_dt, src_format, src_shape, &input_desc));
  aclTensorDesc *input_descs[] = {input_desc.get()};

  std::shared_ptr<aclDataBuffer> input_data;
  NPU_REQUIRES_OK(CreateAclDataBuffer(src_data, src_len, &input_data));
  aclDataBuffer *input_dbs[] = {input_data.get()};

  std::shared_ptr<aclTensorDesc> output_desc;
  NPU_REQUIRES_OK(CreateAclTensorDesc(src_dt, dst_format, dst_shape, &output_desc));
  aclTensorDesc *output_ds[] = {output_desc.get()};

  std::shared_ptr<aclDataBuffer> output_data;
  NPU_REQUIRES_OK(CreateAclDataBuffer(dst_data, dst_len, &output_data));
  aclDataBuffer *output_dbs[] = {output_data.get()};

  std::shared_ptr<aclopAttr> attr;
  NPU_REQUIRES_OK(CreateTransFormatAttr(src_format, dst_format, &attr));
  NPU_REQUIRES_ACL_OK("Acl compile and execute \'TransData\' op failed",
                      aclopCompileAndExecute("TransData", 1, input_descs, input_dbs, 1, output_ds, output_dbs,
                                             attr.get(), ACL_ENGINE_AICORE, ACL_COMPILE_SYS, nullptr, stream));
  return tensorflow::Status::OK();
}
}  // namespace

NpuManagedBuffer::~NpuManagedBuffer() {
  if (deallocator_ && size_ > 0) {
    deallocator_(data_, size_, deallocator_arg_);
  }
}

tensorflow::Status NpuManagedBuffer::Create(ge::Format fmt, const tensorflow::TensorShape &shape,
                                            tensorflow::DataType dtype, NpuManagedBuffer **buf) {
  std::vector<int64_t> dims;
  for (auto dim_size : shape.dim_sizes()) {
    dims.push_back(dim_size);
  }
  ge::DataType ge_type;
  NPU_REQUIRES_OK(MapTfType2Ge(dtype, &ge_type));
  return Create(fmt, dims, ge_type, buf);
}

tensorflow::Status NpuManagedBuffer::Create(ge::Format format, const std::vector<int64_t> &dims, ge::DataType data_type,
                                            NpuManagedBuffer **buf) {
  return Create(format, dims, data_type, format, dims, buf);
}

tensorflow::Status NpuManagedBuffer::Create(ge::Format format, const std::vector<int64_t> &shape,
                                            ge::DataType data_type, ge::Format origin_format,
                                            const std::vector<int64_t> &origin_shape, NpuManagedBuffer **buf) {
  NPU_REQUIRES_OK(npu::global::RtsCtx::EnsureInitialized());
  size_t total_bytes;
  int dtype_size = ge::GetSizeByDataType(data_type);
  NPU_REQUIRES(dtype_size > 0,
               tensorflow::errors::Internal("Data type size invalid ", dtype_size, " for ge type enum ", data_type));
  total_bytes = dtype_size;
  for (auto dim_size : shape) {
    if (dim_size == 0) {
      total_bytes = 0;
      break;
    }
    NPU_REQUIRES(dim_size >= 0, tensorflow::errors::InvalidArgument("Dim size invalid for shape ", VecToString(shape)));
    NPU_REQUIRES(total_bytes <= total_bytes * dim_size,
                 tensorflow::errors::InvalidArgument("Total bytes overflow for shape ", VecToString(shape)));
    total_bytes *= dim_size;
  }
  void *data = nullptr;
  NPU_REQUIRES_OK(NpuMemory::Malloc(total_bytes, &data));
  auto status =
    Create(format, shape, data_type, origin_format, origin_shape, data, total_bytes, nullptr, NpuMemory::Free, buf);
  if (!status.ok()) {
    NpuMemory::Free(data, total_bytes, nullptr);
  }
  return status;
}

tensorflow::Status NpuManagedBuffer::Create(ge::Format format, const std::vector<int64_t> &shape,
                                            ge::DataType data_type, ge::Format origin_format,
                                            const std::vector<int64_t> &origin_shape, void *addr, size_t size,
                                            void *arg, void (*deallocator)(void *, size_t, void *),
                                            NpuManagedBuffer **buf) {
  *buf = new (std::nothrow) NpuManagedBuffer();
  if (*buf == nullptr) {
    return tensorflow::errors::Internal("Failed malloc host npu buffer handle");
  }
  (*buf)->format_ = format;
  (*buf)->shape_ = shape;
  (*buf)->data_type_ = data_type;
  (*buf)->origin_format_ = origin_format;
  (*buf)->origin_data_type_ = data_type;
  (*buf)->origin_shape_ = origin_shape;

  (*buf)->data_ = addr;
  (*buf)->size_ = size;
  (*buf)->deallocator_arg_ = arg;
  (*buf)->deallocator_ = deallocator;

  return tensorflow::Status::OK();
}

void NpuManagedBuffer::Destroy(NpuManagedBuffer *buf) { delete buf; }

tensorflow::Status NpuManagedBuffer::AssembleTo(const tensorflow::Tensor *tensor) {
  NPU_REQUIRES_OK(npu::global::RtsCtx::EnsureInitialized());
  NPU_REQUIRES(tensor != nullptr,
               tensorflow::errors::InvalidArgument("Failed assemble npu buffer to cpu as dst cpu tensor is nullptr"));
  DLOG() << "Npu buffer " << DebugString() << " assemble to " << tensor->DebugString();
  tensorflow::DataType dtype;
  NPU_REQUIRES_OK(MapGeType2Tf(origin_data_type_, &dtype));
  NPU_REQUIRES(dtype == tensor->dtype(),
               tensorflow::errors::InvalidArgument("Data type mismatch when assemble npu buffer to cpu, npu ",
                                                   tensorflow::DataTypeString(dtype), " vs. cpu ",
                                                   tensorflow::DataTypeString(tensor->dtype())));
  if (size_ == 0) {
    return tensorflow::Status::OK();
  }
  if (SameRepresentation()) {
    NPU_REQUIRES_OK(DToH(const_cast<char *>(tensor->tensor_data().data()), tensor->TotalBytes()));
  } else {
    NpuManagedBuffer *buf;
    NPU_REQUIRES_OK(Create(origin_format_, origin_shape_, origin_data_type_, &buf));
    NpuManagedBuffer::Guarder guarder(buf);
    NPU_REQUIRES_OK(TransRepresentationOnNpu(buf));
    buf->DToH(const_cast<char *>(tensor->tensor_data().data()), tensor->TotalBytes());
  }
  return tensorflow::Status::OK();
}

tensorflow::Status NpuManagedBuffer::AssembleFrom(const tensorflow::Tensor *tensor) {
  NPU_REQUIRES_OK(npu::global::RtsCtx::EnsureInitialized());
  NPU_REQUIRES(tensor != nullptr,
               tensorflow::errors::InvalidArgument("Failed assemble npu buffer from cpu as dst cpu tensor is nullptr"));
  DLOG() << "Npu buffer " << DebugString() << " assemble from " << tensor->DebugString();
  tensorflow::DataType dtype;
  NPU_REQUIRES_OK(MapGeType2Tf(origin_data_type_, &dtype));
  NPU_REQUIRES(dtype == tensor->dtype(),
               tensorflow::errors::InvalidArgument("Data type mismatch when assemble npu buffer from cpu, npu ",
                                                   tensorflow::DataTypeString(dtype), " vs. cpu ",
                                                   tensorflow::DataTypeString(tensor->dtype())));
  if (size_ == 0) {
    return tensorflow::Status::OK();
  }
  if (SameRepresentation()) {
    NPU_REQUIRES_OK(HToD(const_cast<char *>(tensor->tensor_data().data()), tensor->TotalBytes()));
  } else {
    NpuManagedBuffer *buf;
    NPU_REQUIRES_OK(Create(origin_format_, origin_shape_, origin_data_type_, &buf));
    NpuManagedBuffer::Guarder guarder(buf);
    NPU_REQUIRES_OK(buf->HToD(const_cast<char *>(tensor->tensor_data().data()), tensor->TotalBytes()));
    NPU_REQUIRES_OK(buf->TransRepresentationOnNpu(this));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status NpuManagedBuffer::TransRepresentationOnNpu(NpuManagedBuffer *dst_buff) {
  DLOG() << "Trans representation on npu, format " << GetFormatName(format_) << " to "
         << GetFormatName(dst_buff->format_) << ", data type " << data_type_ << " to " << dst_buff->data_type_;
  NPU_REQUIRES(format_ != dst_buff->format_ || data_type_ != dst_buff->data_type_, tensorflow::errors::Internal(""));

  aclrtStream rts = nullptr;
  NPU_REQUIRES_ACL_OK("Acl create stream failed", aclrtCreateStream(&rts));
  RtsStreamGuard rts_guard(rts);
  if (format_ == dst_buff->format_) {
    NPU_REQUIRES_OK(ScheduleCastDtypeTask(rts, format_, shape_, data_type_, dst_buff->data_type_, data_,
                                          dst_buff->data_, size_, dst_buff->size_));
  } else if (data_type_ == dst_buff->data_type_) {
    NPU_REQUIRES_OK(ScheduleTransFormatTask(rts, data_type_, format_, shape_, dst_buff->format_, dst_buff->shape_,
                                            data_, dst_buff->data_, size_, dst_buff->size_));
  } else {
    NpuManagedBuffer *buf;
    NPU_REQUIRES_OK(Create(format_, shape_, dst_buff->data_type_, &buf));
    NpuManagedBuffer::Guarder guarder(buf);
    NPU_REQUIRES_OK(ScheduleCastDtypeTask(rts, format_, shape_, data_type_, dst_buff->data_type_, data_, buf->data_,
                                          size_, buf->size_));
    NPU_REQUIRES_OK(ScheduleTransFormatTask(rts, buf->data_type_, buf->format_, buf->shape_, dst_buff->format_,
                                            dst_buff->shape_, buf->data_, dst_buff->data_, buf->size_,
                                            dst_buff->size_));
  }
  NPU_REQUIRES_ACL_OK("Acl synchronize stream failed", aclrtSynchronizeStream(rts));
  return tensorflow::Status::OK();
}

tensorflow::Status NpuManagedBuffer::HToD(void *host_data, size_t size) {
  NPU_REQUIRES(size <= size_, tensorflow::errors::Internal("Failed copy host buffer to npu as size mismatch npu ",
                                                           size_, " vs. cpu ", size));
  NPU_REQUIRES_ACL_OK("Acl rt-memcpy host to device failed",
                      aclrtMemcpy(data_, size_, host_data, size, ACL_MEMCPY_HOST_TO_DEVICE));
  return tensorflow::Status::OK();
}

tensorflow::Status NpuManagedBuffer::DToH(void *host_data, size_t size) {
  NPU_REQUIRES(size >= size_, tensorflow::errors::Internal("Failed copy npu buffer to host as size mismatch npu ",
                                                           size_, " vs. cpu ", size));
  NPU_REQUIRES_ACL_OK("Acl rt-memcpy device to host failed",
                      aclrtMemcpy(host_data, size, data_, size_, ACL_MEMCPY_DEVICE_TO_HOST));
  return tensorflow::Status::OK();
}

std::string NpuManagedBuffer::DebugString() const {
  std::stringstream ss;
  tensorflow::DataType origin_type;
  tensorflow::DataType storage_type;
  (void)MapGeType2Tf(origin_data_type_, &origin_type);
  (void)MapGeType2Tf(data_type_, &storage_type);
  ss << "origin " << GetFormatName(origin_format_) << " " << tensorflow::DataTypeString(origin_type)
     << VecToString(origin_shape_) << ", storage " << GetFormatName(origin_format_) << " "
     << tensorflow::DataTypeString(storage_type) << VecToString(shape_);
  return ss.str();
}