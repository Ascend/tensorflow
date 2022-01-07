/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

#include "tf_adapter/util/acl_channel.h"
#include "securec.h"
#include "acl/error_codes/rt_error_codes.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/npu_attrs.h"
namespace tensorflow {
Status MappingTfDtypeToAcl(const tensorflow::DataType tf_type, aclDataType &acl_type) {
  const static std::map<tensorflow::DataType, aclDataType> type_mapping = {
      {DT_FLOAT, ACL_FLOAT},  {DT_HALF, ACL_FLOAT16},  {DT_INT8, ACL_INT8},     {DT_INT32, ACL_INT32},
      {DT_UINT8, ACL_UINT8},  {DT_INT16, ACL_INT16},   {DT_UINT16, ACL_UINT16}, {DT_UINT32, ACL_UINT32},
      {DT_INT64, ACL_INT64},  {DT_UINT64, ACL_UINT64}, {DT_DOUBLE, ACL_DOUBLE}, {DT_BOOL, ACL_BOOL},
      {DT_STRING, ACL_STRING}};
  auto found = type_mapping.find(tf_type);
  if (found == type_mapping.end()) {
    return errors::Internal("Unsupported tf data type", DataTypeString(tf_type), " by acl.");
  }
  acl_type = found->second;
  return Status::OK();
}

Status MappingAclDtypeToTf(const aclDataType &acl_type, tensorflow::DataType &tf_type) {
  const static std::map<aclDataType, tensorflow::DataType> type_mapping = {
      {ACL_FLOAT, DT_FLOAT},  {ACL_FLOAT16, DT_HALF},  {ACL_INT8, DT_INT8},     {ACL_INT32, DT_INT32},
      {ACL_UINT8, DT_UINT8},  {ACL_INT16, DT_INT16},   {ACL_UINT16, DT_UINT16}, {ACL_UINT32, DT_UINT32},
      {ACL_INT64, DT_INT64},  {ACL_UINT64, DT_UINT64}, {ACL_DOUBLE, DT_DOUBLE}, {ACL_BOOL, DT_BOOL},
      {ACL_STRING, DT_STRING}};
  auto found = type_mapping.find(acl_type);
  if (found == type_mapping.end()) {
    return errors::Internal("Acl channel receive unsupported data type", acl_type);
  }
  tf_type = found->second;
  return Status::OK();
}

Status AssembleAclTensor2Tensor(acltdtDataItem *item, std::vector<Tensor> &tensors, bool call_by_channel_receive) {
  acltdtTensorType acl_type = acltdtGetTensorTypeFromItem(item);
  if (acl_type == ACL_TENSOR_DATA_END_OF_SEQUENCE) {
    LOG(INFO) << "Acl channel received end-of-sequence for out-feed op.";
    return Status::OK();
  } else if (acl_type == ACL_TENSOR_DATA_ABNORMAL) {
    LOG(INFO) << "Acl channel received abnormal for out-feed op.";
    return Status::OK();
  } else if (acl_type == ACL_TENSOR_DATA_UNDEFINED) {
    LOG(INFO) << "Acl channel received undefined message type for out-feed op.";
    return errors::Internal("Acl channel received undefined message type for out-feed op.");
  }
  tensorflow::DataType tf_type;
  TF_RETURN_IF_ERROR(MappingAclDtypeToTf(acltdtGetDataTypeFromItem(item), tf_type));
  size_t dim_num = acltdtGetDimNumFromItem(item);
  size_t acl_data_len = acltdtGetDataSizeFromItem(item);
  char *acl_data = reinterpret_cast<char *>(acltdtGetDataAddrFromItem(item));
  if (acl_data == nullptr) {
    return errors::Internal("Acl get data addr from item failed when receive tensor data.");
  }
  if (!kIsNewDataTransfer && call_by_channel_receive) {
    acl_data = const_cast<char *>(reinterpret_cast<std::string *>(acl_data)->c_str());
  }
  if (tf_type == DT_STRING) {
    if (dim_num != 0) {
      return errors::Internal("Acl channel receive unsupported non-scalar string type");
    }
    Tensor tensor(tf_type, TensorShape({}));
    tensor.scalar<string>()() = std::move(string(acl_data, acl_data_len));
    tensors.emplace_back(std::move(tensor));
  } else if (DataTypeCanUseMemcpy(tf_type)) {
    std::vector<int64_t> dims;
    dims.resize(dim_num);
    if (acltdtGetDimsFromItem(item, dims.data(), dim_num) != ACL_ERROR_NONE) {
      return errors::Internal("Failed get dim-size from acl channel data");
    }
    TensorShape tf_shape;
    for (auto dim : dims) {
      tf_shape.AddDim(dim);
    }
    Tensor tensor = Tensor(tf_type, tf_shape);
    auto tensor_data = const_cast<char *>(tensor.tensor_data().data());
    auto tensor_size = tensor.tensor_data().size();
    if (tensor_size != acl_data_len) {
      return errors::Internal("Acl channel receive size mismatch tensor size acl:", acl_data_len,
                              "vs. tf:", tensor_size);
    }
    do {
      auto copy_size = (tensor_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : tensor_size;
      LOG(INFO) << "tensor data:" << reinterpret_cast<uintptr_t>(tensor_data) << ", tensor_size:" << tensor_size
                << ", acl_data:" << reinterpret_cast<uintptr_t>(acl_data) << ", copy_size:" << copy_size;
      if (memcpy_s(tensor_data, tensor_size, acl_data, copy_size) != EOK) {
        return errors::Internal("Failed copy acl channel data to tensorflow.");
      }
      tensor_size -= copy_size;
      tensor_data += copy_size;
      acl_data += copy_size;
    } while (tensor_size > 0);
    tensors.emplace_back(std::move(tensor));
  } else {
    return errors::InvalidArgument("Acl channel receive uncopyable tf data type", DataTypeString(tf_type));
  }
  return Status::OK();
}

Status AssembleAclDataset2Tensors(acltdtDataset *acl_dataset, std::vector<Tensor> &out_tensors,
                                  bool call_by_channel_receive) {
  for (size_t i = 0; i < acltdtGetDatasetSize(acl_dataset); i++) {
    auto acl_data = acltdtGetDataItem(acl_dataset, i);
    if (acl_data == nullptr) {
      return errors::Internal("Acl get tensor data from dataset failed when receive tensor data.");
    }
    TF_RETURN_IF_ERROR(AssembleAclTensor2Tensor(acl_data, out_tensors, call_by_channel_receive));
  }
  return Status::OK();
}

Status AssembleTensors2AclDataset(acltdtTensorType acl_type, const std::vector<Tensor> &tensors,
                                  acltdtDataset **output_acl_dataset) {
  auto acl_dataset = acltdtCreateDataset();
  if (acl_dataset == nullptr) {
    return errors::Internal("Acl create tensor dataset failed");
  }
  auto status = AssembleTensors2AclDataset(acl_type, tensors, acl_dataset);
  if (!status.ok()) {
    ADAPTER_LOG_IF_ERROR(DestroyAclDataset(acl_dataset));
    return status;
  }
  *output_acl_dataset = acl_dataset;
  return Status::OK();
}

Status AssembleTensors2AclDataset(acltdtTensorType acl_type, const std::vector<Tensor> &tensors,
                                  acltdtDataset *acl_dataset) {
  if (TF_PREDICT_FALSE(acl_type != ACL_TENSOR_DATA_TENSOR)) {
    acltdtDataItem *acl_data = acltdtCreateDataItem(acl_type, nullptr, 0, ACL_BOOL /* whatever */, nullptr, 0);
    if (acl_data == nullptr) {
      return errors::Internal("Acl create tensor item failed when send end-of-sequence.");
    }
    if (acltdtAddDataItem(acl_dataset, acl_data) != ACL_ERROR_NONE) {
      if (acltdtDestroyDataItem(acl_data) != ACL_ERROR_NONE) {
        LOG(ERROR) << "Acl destroy tensor data item failed when send data with type "
                   << (acl_type == ACL_TENSOR_DATA_END_OF_SEQUENCE ? "ACL_TENSOR_DATA_END_OF_SEQUENCE"
                                                                   : "ACL_TENSOR_DATA_ABNORMAL");
      }
      return errors::Internal("Acl add tensor data to dataset failed when send data with type ", acl_type);
    }
    return Status::OK();
  }
  for (auto &tensor : tensors) {
    aclDataType acl_data_type;
    TF_RETURN_IF_ERROR(MappingTfDtypeToAcl(tensor.dtype(), acl_data_type));
    acltdtDataItem *acl_data = nullptr;
    if (DataTypeCanUseMemcpy(tensor.dtype())) {
      auto dims = tensor.shape().dim_sizes();
      acl_data = acltdtCreateDataItem(
          ACL_TENSOR_DATA_TENSOR, (dims.empty() ? nullptr : reinterpret_cast<const int64_t *>(dims.data())),
          dims.size(), acl_data_type, const_cast<char *>(tensor.tensor_data().data()), tensor.tensor_data().size());
    } else if (tensor.dtype() == DT_STRING) {
      if (tensor.dims() != 0) {
        return errors::Internal("Acl send got unexpected non-scalar string tensor with dim ", tensor.dims());
      }
      auto value = reinterpret_cast<string *>(const_cast<char *>(tensor.tensor_data().data()));
      // for scalar type, *dims is nullptr and dim_num is 0
      acl_data = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, nullptr, 0, acl_data_type,
                                      const_cast<char *>(value->c_str()), value->size());
    } else {
      return errors::Internal("Acl send got unexpected data type ", DataTypeString(tensor.dtype()));
    }
    if (acl_data == nullptr) {
      return errors::Internal("Acl create tensor item failed when send tensor data ", tensor.DebugString());
    }
    if (acltdtAddDataItem(acl_dataset, acl_data) != ACL_ERROR_NONE) {
      if (acltdtDestroyDataItem(acl_data) != ACL_ERROR_NONE) {
        ADP_LOG(ERROR) << "Acl destroy tensor data item failed when send data with type ACL_TENSOR_DATA_TENSOR.";
      }
      return errors::Internal("Acl add tensor data to dataset failed when send tensor data.");
    }
  }
  return Status::OK();
}

Status DestroyAclDataset(acltdtDataset *acl_dataset, bool include_data_item) {
  if (include_data_item) {
    for (size_t i = 0; i < acltdtGetDatasetSize(acl_dataset); i++) {
      if (acltdtDestroyDataItem(acltdtGetDataItem(acl_dataset, i)) != ACL_ERROR_NONE) {
        return errors::Internal("Acl destroy tensor data failed.");
      }
    }
  }
  if (acltdtDestroyDataset(acl_dataset) != ACL_ERROR_NONE) {
    return errors::Internal("Acl destroy tensor dataset failed.");
  }
  return Status::OK();
}

Status RecvTensorByAcl(acltdtChannelHandle *acl_handle, std::vector<Tensor> &tensors) {
  auto acl_dataset = acltdtCreateDataset();
  if (acl_dataset == nullptr) {
    return errors::Internal("Failed create acl channel.");
  }
  auto acl_status = acltdtReceiveTensor(acl_handle, acl_dataset, -1 /* no timeout */);
  if (acl_status != ACL_ERROR_NONE && acl_status != ACL_ERROR_RT_QUEUE_EMPTY) {
    ADAPTER_LOG_IF_ERROR(DestroyAclDataset(acl_dataset, false));
    return errors::Internal("Failed receive data from acl channel, acl status:", acl_status);
  }

  auto status = AssembleAclDataset2Tensors(acl_dataset, tensors, true /* call by channel receive */);
  if (!status.ok()) {
    ADAPTER_LOG_IF_ERROR(DestroyAclDataset(acl_dataset, false));
    return status;
  }
  TF_RETURN_IF_ERROR(DestroyAclDataset(acl_dataset, false));
  return Status::OK();
}
// When calling SendTensorsByAcl and its'return is the queue is full or
// empty (actually no event, drv wants us to treat it as a no event,
// because they cannot return no evnet code , only empty). The above 2
// cases , we need to push data into dequeue to sent again.
Status SendTensorsByAcl(const acltdtChannelHandle *acl_handle, acltdtTensorType acl_type,
                        const std::vector<Tensor> &tensors, bool &is_need_resend) {
  acltdtDataset *acl_dataset = nullptr;
  TF_RETURN_IF_ERROR(AssembleTensors2AclDataset(acl_type, tensors, &acl_dataset));
  const int32_t kTimeOut = 1000;
  auto acl_status = acltdtSendTensor(acl_handle, acl_dataset, kTimeOut);
  TF_RETURN_IF_ERROR(DestroyAclDataset(acl_dataset));
  if (acl_status == ACL_ERROR_RT_QUEUE_EMPTY || acl_status == ACL_ERROR_RT_QUEUE_FULL) {
    is_need_resend = true;
    ADP_LOG(EVENT) << "Send data failed , need send data again.";
    return Status::OK();
  }
  if (acl_status != ACL_ERROR_NONE) {
    return errors::Internal("Acl send data failed, acl status:", acl_status);
  }
  return Status::OK();
}

acltdtChannelHandle *CreateAclTdtRecvChannel(uint32_t device_id, const std::string &channel_name,
                                             const size_t capacity) {
  if (kIsNewDataTransfer) {
    return acltdtCreateChannelWithCapacity(device_id, channel_name.c_str(), capacity);
  }
  const static std::string kReceivePrefix = "TF_RECEIVE_";
  return acltdtCreateChannel(device_id, (kReceivePrefix + channel_name).c_str());
}

Status StopRecvTensorByAcl(acltdtChannelHandle **handle, const std::string &channel_name) {
  if (kIsNewDataTransfer) {
    if (acltdtDestroyChannel(*handle) != ACL_ERROR_NONE) {
      return errors::Internal("Failed destroy acl data channel for host queue:", channel_name);
    } else {
      *handle = nullptr;
    }
  } else {
    if (acltdtStopChannel(*handle) != ACL_ERROR_NONE) {
      return errors::Internal("Failed stop acl data channel for host queue:", channel_name);
    }
  }
  ADP_LOG(INFO) << "Success to stop recv tensor by acl.";
  return Status::OK();
}
}  // namespace tensorflow