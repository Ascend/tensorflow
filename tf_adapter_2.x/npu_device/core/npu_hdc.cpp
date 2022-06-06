/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include <securec.h>
#include "npu_hdc.h"
#include "npu_logger.h"
#include "npu_micros.h"
#include "npu_thread_pool.h"
#include "npu_utils.h"

namespace {
constexpr size_t kParallelMemCopyThreshold = 10 * 1024 * 1024UL;

/**
 * @brief: parallel mem copy
 */
tensorflow::Status Copy2ContinuousMem(void *dst_ptr, void *src_ptr, const size_t src_size) {
  if (dst_ptr == nullptr || src_ptr == nullptr) {
    return tensorflow::errors::Internal("dst_ptr or src_ptr is null before do parallel memory copy.");
  }

  if (src_size <= kParallelMemCopyThreshold) {
    if (memcpy_s(dst_ptr, src_size, src_ptr, src_size) != EOK) {
      return tensorflow::errors::Internal("memory copy failed before hdc send data, dst:", dst_ptr,
                                          ", dst_size:", src_size, ", src:", src_ptr, ", src_size:", src_size);
    }
    return tensorflow::Status::OK();
  }

  std::atomic<uint32_t> parallel_cpy_count{0U};
  uint32_t enqueue_count = 0;
  size_t block_size = src_size / npu::kDefaultThreadNum;
  size_t remained_size = src_size % npu::kDefaultThreadNum;
  std::vector<tensorflow::Status> copy_results(npu::kDefaultThreadNum);
  for (int32_t i = 0; i < npu::kDefaultThreadNum; i++) {
    if (i == npu::kDefaultThreadNum - 1) {
      block_size += remained_size;
    }
    auto &ret = copy_results[i];
    std::function<void()> closure = [dst_ptr, src_ptr, block_size, &ret, &parallel_cpy_count]() {
      ret = npu::LoopCopy(dst_ptr, src_ptr, block_size);
      parallel_cpy_count++;
    };
    NPU_REQUIRES_OK(npu::ThreadPool::GetInstance().EnqueueTask(closure));
    enqueue_count++;
    dst_ptr += block_size;
    src_ptr += block_size;
  }
  while (parallel_cpy_count < enqueue_count) {
  }
  for (auto ret : copy_results) {
    if (!ret.ok()) {
      return ret;
    }
  }
  return tensorflow::Status::OK();
}
}  // namespace

namespace npu {
/**
 * @brief: mapping acl data type to tf
 * @param acl_type: acl data type
 * @param tf_type: tensorflow data type
 */
tensorflow::Status HdcChannel::MappingAclDtypeToTf(const aclDataType &acl_type, tensorflow::DataType &tf_type) const {
  const static std::map<aclDataType, tensorflow::DataType> type_mapping = {
    {ACL_FLOAT, tensorflow::DT_FLOAT},   {ACL_FLOAT16, tensorflow::DT_HALF},  {ACL_INT8, tensorflow::DT_INT8},
    {ACL_INT32, tensorflow::DT_INT32},   {ACL_UINT8, tensorflow::DT_UINT8},   {ACL_INT16, tensorflow::DT_INT16},
    {ACL_UINT16, tensorflow::DT_UINT16}, {ACL_UINT32, tensorflow::DT_UINT32}, {ACL_INT64, tensorflow::DT_INT64},
    {ACL_UINT64, tensorflow::DT_UINT64}, {ACL_DOUBLE, tensorflow::DT_DOUBLE}, {ACL_BOOL, tensorflow::DT_BOOL},
    {ACL_STRING, tensorflow::DT_STRING}};
  auto found = type_mapping.find(acl_type);
  if (found == type_mapping.end()) {
    return tensorflow::errors::Internal("Hdc channel receive unsupported data type", acl_type);
  }
  tf_type = found->second;
  return tensorflow::Status::OK();
}

/**
 * @brief: assemble acl tensor to tensor
 * @param item: acl tdt data item
 * @param tensors: tensorflow tensors
 * @param call_by_channel_receive: if call by channel receive or not
 */
tensorflow::Status HdcChannel::AssembleAclTensor2Tensor(acltdtDataItem *item,
                                                        std::vector<tensorflow::Tensor> &tensors) const {
  acltdtTensorType acl_type = acltdtGetTensorTypeFromItem(item);
  if (acl_type == ACL_TENSOR_DATA_END_OF_SEQUENCE) {
    LOG(INFO) << "Hdc channel received end-of-sequence for out-feed op.";
    return tensorflow::Status::OK();
  } else if (acl_type == ACL_TENSOR_DATA_ABNORMAL) {
    LOG(INFO) << "Hdc channel received abnormal for out-feed op.";
    return tensorflow::Status::OK();
  } else if (acl_type == ACL_TENSOR_DATA_UNDEFINED) {
    LOG(INFO) << "Hdc channel received undefined message type for out-feed op.";
    return tensorflow::errors::Internal("Hdc channel received undefined message type for out-feed op.");
  }
  tensorflow::DataType tf_type = tensorflow::DT_INVALID;
  TF_RETURN_IF_ERROR(MappingAclDtypeToTf(acltdtGetDataTypeFromItem(item), tf_type));
  size_t dim_num = acltdtGetDimNumFromItem(item);
  size_t acl_data_len = acltdtGetDataSizeFromItem(item);
  const char *acl_data = static_cast<char *>(acltdtGetDataAddrFromItem(item));
  if (acl_data == nullptr) {
    return tensorflow::errors::Internal("Failed get data addr from item");
  }
  if (tf_type == tensorflow::DT_STRING) {
    if (dim_num != 0) {
      return tensorflow::errors::Internal("Hdc channel receive unsupported non-scalar string type");
    }
    tensorflow::Tensor tensor(tf_type, tensorflow::TensorShape({}));
    tensor.scalar<tensorflow::tstring>()() = std::string(acl_data, acl_data_len);
    tensors.emplace_back(std::move(tensor));
  } else if (DataTypeCanUseMemcpy(tf_type)) {
    std::vector<int64_t> dims;
    dims.resize(dim_num);
    if (acltdtGetDimsFromItem(item, dims.data(), dim_num) != ACL_ERROR_NONE) {
      return tensorflow::errors::Internal("Failed get dim-size from hdc channel data");
    }
    tensorflow::TensorShape tf_shape;
    for (auto dim : dims) {
      tf_shape.AddDim(dim);
    }
    tensorflow::Tensor tensor = tensorflow::Tensor(tf_type, tf_shape);
    auto tensor_data = tensor.data();
    auto tensor_size = tensor.tensor_data().size();
    if (tensor_size != acl_data_len) {
      return tensorflow::errors::Internal("Hdc channel receive size mismatch tensor size acl:", acl_data_len,
                                          " vs. tensorflow:", tensor_size);
    }
    (void)memcpy(tensor_data, acl_data, tensor_size);
    tensors.emplace_back(std::move(tensor));
  } else {
    return tensorflow::errors::InvalidArgument("Hdc channel receive un-copyable tensorflow data type",
                                               DataTypeString(tf_type));
  }
  return tensorflow::Status::OK();
}

/**
 * @brief: assemble acl dataset to tensors
 * @param acl_dataset: acl tdt dataset
 * @param out_tensors: tensorflow tensors
 * @param call_by_channel_receive: if call by channel receive of not
 */
tensorflow::Status HdcChannel::AssembleAclDataset2Tensors(acltdtDataset *acl_dataset,
                                                          std::vector<tensorflow::Tensor> &out_tensors) const {
  for (size_t i = 0; i < acltdtGetDatasetSize(acl_dataset); i++) {
    auto acl_data = acltdtGetDataItem(acl_dataset, i);
    if (acl_data == nullptr) {
      return tensorflow::errors::Internal("Acl get tensor data from dataset failed when receive tensor data.");
    }
    TF_RETURN_IF_ERROR(AssembleAclTensor2Tensor(acl_data, out_tensors));
  }
  return tensorflow::Status::OK();
}

/**
 * @brief: destroy acl dataset
 * @param acl_dataset: acl tdt dataset
 * @param include_data_item: if include data item or not
 */
tensorflow::Status HdcChannel::DestroyAclDataset(acltdtDataset *acl_dataset, bool include_data_item) const {
  if (include_data_item) {
    for (size_t i = 0; i < acltdtGetDatasetSize(acl_dataset); i++) {
      if (acltdtDestroyDataItem(acltdtGetDataItem(acl_dataset, i)) != ACL_ERROR_NONE) {
        return tensorflow::errors::Internal("Acl destroy tensor data failed.");
      }
    }
  }
  if (acltdtDestroyDataset(acl_dataset) != ACL_ERROR_NONE) {
    return tensorflow::errors::Internal("Acl destroy tensor dataset failed.");
  }
  return tensorflow::Status::OK();
}

/**
 * @brief: recv tensor by acl
 * @param tensors: tensorflow tensors
 */
tensorflow::Status HdcChannel::RecvTensorByAcl(std::vector<tensorflow::Tensor> &tensors) const {
  auto acl_dataset = acltdtCreateDataset();
  if (acl_dataset == nullptr) {
    return tensorflow::errors::Internal("Failed create hdc channel.");
  }

  auto acl_status = acltdtReceiveTensor(handle_.acl_handle, acl_dataset, -1 /* no timeout */);
  if (acl_status != ACL_ERROR_NONE) {
    NPU_LOG_IF_ERROR(DestroyAclDataset(acl_dataset, false));
    return tensorflow::errors::Internal("Failed receive data from hdc channel, acl status:", acl_status);
  }

  auto status = AssembleAclDataset2Tensors(acl_dataset, tensors);
  if (!status.ok()) {
    NPU_LOG_IF_ERROR(DestroyAclDataset(acl_dataset, false));
    return status;
  }
  TF_RETURN_IF_ERROR(DestroyAclDataset(acl_dataset, false));
  return tensorflow::Status::OK();
}

/**
 * @brief: mapping tf data type to acl
 * @tf_type: tensorflow data type
 * @acl_type: acl data type
 */
tensorflow::Status HdcChannel::MappingTfDtypeToAcl(const tensorflow::DataType tf_type, aclDataType &acl_type) const {
  const static std::map<tensorflow::DataType, aclDataType> type_mapping = {
    {tensorflow::DT_FLOAT, ACL_FLOAT},   {tensorflow::DT_HALF, ACL_FLOAT16},  {tensorflow::DT_INT8, ACL_INT8},
    {tensorflow::DT_INT32, ACL_INT32},   {tensorflow::DT_UINT8, ACL_UINT8},   {tensorflow::DT_INT16, ACL_INT16},
    {tensorflow::DT_UINT16, ACL_UINT16}, {tensorflow::DT_UINT32, ACL_UINT32}, {tensorflow::DT_INT64, ACL_INT64},
    {tensorflow::DT_UINT64, ACL_UINT64}, {tensorflow::DT_DOUBLE, ACL_DOUBLE}, {tensorflow::DT_BOOL, ACL_BOOL},
    {tensorflow::DT_STRING, ACL_STRING}};
  auto found = type_mapping.find(tf_type);
  if (found == type_mapping.end()) {
    return tensorflow::errors::Internal("Unsupported tensorflow data type ", DataTypeString(tf_type), " by acl.");
  }
  acl_type = found->second;
  return tensorflow::Status::OK();
}

/**
 * @brief: assemble tensors to acl dataset
 * @param acl_type: acl tdt tensor type
 * @param tensors: tensorflow tensors
 * @param acl_dataset: acl tdt dataset
 */
tensorflow::Status HdcChannel::AssembleTensors2AclDataset(acltdtTensorType acl_type,
                                                          const std::vector<tensorflow::Tensor> &tensors,
                                                          acltdtDataset *acl_dataset) const {
  if (TF_PREDICT_FALSE(acl_type != ACL_TENSOR_DATA_TENSOR)) {
    acltdtDataItem *acl_data = acltdtCreateDataItem(acl_type, nullptr, 0, ACL_BOOL /* whatever */, nullptr, 0);
    if (acl_data == nullptr) {
      return tensorflow::errors::Internal("Acl create tensor item failed when send end-of-sequence.");
    }
    if (acltdtAddDataItem(acl_dataset, acl_data) != ACL_ERROR_NONE) {
      if (acltdtDestroyDataItem(acl_data) != ACL_ERROR_NONE) {
        LOG(ERROR) << "Acl destroy tensor data item failed when send data with type "
                   << (acl_type == ACL_TENSOR_DATA_END_OF_SEQUENCE ? "ACL_TENSOR_DATA_END_OF_SEQUENCE"
                                                                   : "ACL_TENSOR_DATA_ABNORMAL");
      }
      return tensorflow::errors::Internal("Acl add tensor data to dataset failed when send data with type ", acl_type);
    }
    return tensorflow::Status::OK();
  }
  size_t total_size = 0UL;
  for (auto &tensor : tensors) {
    total_size += tensor.TotalBytes();
  }
  tensors_buffer_.resize(std::max(tensors_buffer_.size(), total_size));

  size_t offset = 0UL;
  for (auto &tensor : tensors) {
    aclDataType acl_data_type;
    TF_RETURN_IF_ERROR(MappingTfDtypeToAcl(tensor.dtype(), acl_data_type));
    acltdtDataItem *acl_data = nullptr;
    if (DataTypeCanUseMemcpy(tensor.dtype())) {
      auto dims = tensor.shape().dim_sizes();
      void *tensor_data = tensor.data();
      if (IsNeedContinuousMem()) {
        size_t src_size = tensor.TotalBytes();
        tensor_data = tensors_buffer_.data() + offset;
        NPU_REQUIRES_OK(Copy2ContinuousMem(tensor_data, tensor.data(), src_size));
        offset += src_size;
      }
      acl_data = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR,
                                      (dims.empty() ? nullptr : reinterpret_cast<const int64_t *>(dims.data())),
                                      dims.size(), acl_data_type, tensor_data, tensor.tensor_data().size());
    } else if (tensor.dtype() == tensorflow::DT_STRING) {
      if (tensor.dims() != 0) {
        return tensorflow::errors::Internal("Acl send got unexpected non-scalar string tensor with dim ",
                                            tensor.dims());
      }
      auto value = static_cast<tensorflow::tstring *>(tensor.data());
      // for scalar type, *dims is nullptr and dim_num is 0
      acl_data = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, nullptr, 0, acl_data_type,
                                      const_cast<char *>(value->c_str()), value->size());
    } else {
      return tensorflow::errors::Internal("Acl send got unexpected data type ", DataTypeString(tensor.dtype()));
    }
    if (acl_data == nullptr) {
      return tensorflow::errors::Internal("Acl create tensor item failed when send tensor data ", tensor.DebugString());
    }
    if (acltdtAddDataItem(acl_dataset, acl_data) != ACL_ERROR_NONE) {
      if (acltdtDestroyDataItem(acl_data) != ACL_ERROR_NONE) {
        LOG(ERROR) << "Acl destroy tensor data item failed when send data with type ACL_TENSOR_DATA_TENSOR";
      }
      return tensorflow::errors::Internal("Acl add tensor data to dataset failed when send tensor data.");
    }
  }
  return tensorflow::Status::OK();
}

/**
 * @brief: assemble tensors to acl dataset
 * @param acl_type: acl tdt tensor type
 * @param tensors: tensorflow tensors
 * @param output_acl_dataset: acl tdt dataset
 */
tensorflow::Status HdcChannel::AssembleTensors2AclDataset(acltdtTensorType acl_type,
                                                          const std::vector<tensorflow::Tensor> &tensors,
                                                          acltdtDataset **output_acl_dataset) const {
  auto acl_dataset = acltdtCreateDataset();
  if (acl_dataset == nullptr) {
    return tensorflow::errors::Internal("Acl create tensor dataset failed");
  }
  auto status = AssembleTensors2AclDataset(acl_type, tensors, acl_dataset);
  if (!status.ok()) {
    NPU_LOG_IF_ERROR(DestroyAclDataset(acl_dataset));
    return status;
  }
  *output_acl_dataset = acl_dataset;
  return tensorflow::Status::OK();
}

/**
 * @brief: send tensors by acl
 * @param acl_type: acl tdt tensor type
 * @param tensors: tensorflow tensors
 */
tensorflow::Status HdcChannel::SendTensorsByAcl(acltdtTensorType acl_type,
                                                const std::vector<tensorflow::Tensor> &tensors) const {
  acltdtDataset *acl_dataset = nullptr;

  TF_RETURN_IF_ERROR(AssembleTensors2AclDataset(acl_type, tensors, &acl_dataset));

  auto acl_status = ACL_ERROR_NONE;
  do {
    acl_status = acltdtSendTensor(handle_.acl_handle, acl_dataset, -1 /* no timeout */);
  } while ((acl_status == ACL_ERROR_RT_QUEUE_EMPTY || acl_status == ACL_ERROR_RT_QUEUE_FULL));
  TF_RETURN_IF_ERROR(DestroyAclDataset(acl_dataset));
  if (acl_status != ACL_ERROR_NONE) {
    return tensorflow::errors::Internal("Acl send data failed, acl status:", acl_status);
  }

  return tensorflow::Status::OK();
}

/**
 * @brief: create hdc channel
 * @param device_id: device id
 * @param name: channel name
 * @param guarded_channel: shared point to HdcChannel
 */
tensorflow::Status HdcChannel::Create(uint32_t device_id, const std::string &name,
                                      std::shared_ptr<HdcChannel> *guarded_channel) {
  guarded_channel->reset(new (std::nothrow) HdcChannel(device_id, name));
  NPU_REQUIRES(*guarded_channel,
               tensorflow::errors::Internal("Failed create hdc channel ", name, " on device ", device_id));
  NPU_REQUIRES_OK((*guarded_channel)->Init());
  return tensorflow::Status::OK();
}

tensorflow::Status HdcChannel::Create(uint32_t device_id, const std::string &name, size_t capacity,
                                      std::shared_ptr<HdcChannel> *guarded_channel) {
  guarded_channel->reset(new (std::nothrow) HdcChannel(device_id, name, capacity));
  NPU_REQUIRES(*guarded_channel,
               tensorflow::errors::Internal("Failed create hdc channel ", name, " on device ", device_id));
  return (*guarded_channel)->Init();
}

HdcChannel::~HdcChannel() { Destroy(); }

/**
 * @brief: send tensors
 */
tensorflow::Status HdcChannel::SendTensors(const std::vector<tensorflow::Tensor> &tensors) const {
  return SendTensorsByAcl(ACL_TENSOR_DATA_TENSOR, tensors);
}

/**
 * @brief: recv tensors
 */
tensorflow::Status HdcChannel::RecvTensors(std::vector<tensorflow::Tensor> &tensors) const {
  return RecvTensorByAcl(tensors);
}

/**
 * @brief: notify finish
 */
tensorflow::Status HdcChannel::NotifyFinish() const { return SendTensorsByAcl(ACL_TENSOR_DATA_END_OF_SEQUENCE, {}); }

/**
 * @brief: notify abnormal
 */
tensorflow::Status HdcChannel::NotifyAbnormal() const { return SendTensorsByAcl(ACL_TENSOR_DATA_ABNORMAL, {}); }

/**
 * @brief: destroy hdc channel
 */
void HdcChannel::Destroy() {
  if (!destroyed_.exchange(true)) {
    if (handle_.acl_handle != nullptr && acltdtDestroyChannel(handle_.acl_handle) != ACL_ERROR_NONE) {
      LOG(ERROR) << "Failed close hdc channel " << name_;
    } else {
      DLOG() << "Hdc channel " << name_ << " closed";
    }
  }
}

HdcChannel::HdcChannel(uint32_t device_id, std::string name) : device_id_(device_id), name_(std::move(name)) {}

HdcChannel::HdcChannel(uint32_t device_id, std::string name, size_t capacity)
    : device_id_(device_id), name_(std::move(name)), limited_capacity_(true), capacity_(capacity) {}
/**
 * @brief: init hdc channel
 */
tensorflow::Status HdcChannel::Init() {
  if (limited_capacity_) {
    handle_.acl_handle = acltdtCreateChannelWithCapacity(device_id_, name_.c_str(), capacity_);
    handle_.is_mbuf = true;
  } else {
    handle_.acl_handle = acltdtCreateChannel(device_id_, name_.c_str());
  }
  if (handle_.acl_handle == nullptr) {
    return tensorflow::errors::Internal("Failed create hdc channel by acl");
  }
  return tensorflow::Status::OK();
}
}  // namespace npu