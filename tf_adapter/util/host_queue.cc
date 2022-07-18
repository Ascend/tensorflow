/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
#include "tf_adapter/util/host_queue.h"

#include <securec.h>
#include <unordered_map>
#include "mmpa/mmpa_api.h"
#include "runtime/dev.h"
#include "runtime/rt_mem_queue.h"
#include "graph/def_types.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/util/acl_channel.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/ge_plugin.h"
#include "tf_adapter/util/util.h"
#include "tf_adapter_2.x/npu_device/core/npu_micros.h"

namespace tensorflow {
namespace {
std::mutex queue_id_to_trans_id_map_mutex;
std::unordered_map<uint32_t, uint64_t> queue_id_to_trans_id_map;

constexpr int64_t kMaxDimSize = 32;

#pragma pack(push, 1)
struct RuntimeTensorDesc {
  uint64_t data_addr;
  int64_t data_offset_size;
  int64_t dtype;
  int64_t shape[kMaxDimSize + 1];           // shape:Dim_Num|DIM0|DIM1|...|DIM31
  int64_t original_shape[kMaxDimSize + 1];  // original_shape:Dim_Num|DIM0|DIM1|...|DIM31
  int64_t format;
  int64_t sub_format;
  uint8_t reserved[456];  // padding to 1024 bytes
};
#pragma pack(pop)

struct ItemInfo {
  int32_t version;
  int32_t data_type;
  uint32_t cur_cnt;
  uint32_t cnt;
  int32_t tensor_type;
  uint32_t dim_num;
  char reserved[32];
  uint64_t data_len;
};

struct DataItemInfo {
  ItemInfo ctrl_info;
  std::vector<int64_t> dims;
  void *data_ptr;
};

struct MsgInfo {
  uint64_t trans_id;
  uint16_t version;
  uint16_t msg_type;
  int32_t ret_code;
  uint64_t timestamp;
  char rsv[40];
};

const static uint32_t kMaxValue = 128U;
const static uint32_t kMaxQueueDepth = 0x4fffffffU;
const static uint64_t kMbufHeadMaxSize = 256UL;
const static uint32_t kMbufHeadEndOfSequencePos = 128U;
const static uint8_t kEndOfSequenceFlag = 0x5A;

Status CheckSymbols() {
  if (rtMemQueueCreate == nullptr) { return errors::Internal("rtMemQueueCreate not found"); }
  if (rtMemQueueDestroy == nullptr) { return errors::Internal("rtMemQueueDestroy not found"); }
  if (rtMemQueueInit == nullptr) { return errors::Internal("rtMemQueueInit not found"); }
  if (rtMemQueueEnQueue == nullptr) { return errors::Internal("rtMemQueueEnQueue not found"); }
  if (rtMemQueueDeQueue == nullptr) { return errors::Internal("rtMemQueueDeQueue not found"); }
  if (rtMbufInit == nullptr) { return errors::Internal("rtMbufInit not found"); }
  if (rtMbufAlloc == nullptr) { return errors::Internal("rtMbufAlloc not found"); }
  if (rtMbufFree == nullptr) { return errors::Internal("rtMbufFree not found"); }
  if (rtMbufGetBuffAddr == nullptr) { return errors::Internal("rtMbufGetBuffAddr not found"); }
  if (rtMbufGetBuffSize == nullptr) { return errors::Internal("rtMbufGetBuffSize not found"); }
  if (rtMbufGetPrivInfo == nullptr) { return errors::Internal("rtMbufGetPrivInfo not found"); }
  return Status::OK();
}

Status GetDataTypeByTensorType(acltdtTensorType tensor_type, int32_t &data_type) {
  const static std::unordered_map<acltdtTensorType, int32_t> type_map = {
    {ACL_TENSOR_DATA_TENSOR, 0}, {ACL_TENSOR_DATA_END_OF_SEQUENCE, 1}, {ACL_TENSOR_DATA_ABNORMAL, 2}};
  auto ret = type_map.find(tensor_type);
  if (ret == type_map.end()) {
    ADP_LOG(ERROR) << "invalid tensor_type: " << static_cast<int32_t>(tensor_type);
    return errors::Internal("invalid tensor type : ", tensor_type);
  }

  data_type = ret->second;
  ADP_LOG(INFO) << "get data type[" << data_type << "] by tensor type[" << static_cast<int32_t>(tensor_type)
    << "] success";
  return Status::OK();
}

Status AddDataItemInfo(acltdtTensorType tdt_data_type, int32_t tensor_type, const int64_t *dims, size_t dim_size,
                       void *data_ptr, uint64_t data_len, std::vector<DataItemInfo> &items) {
  DataItemInfo item = {};
  int32_t data_type = 0;
  TF_RETURN_IF_ERROR(GetDataTypeByTensorType(tdt_data_type, data_type));
  item.ctrl_info.data_type = data_type;
  item.ctrl_info.tensor_type = tensor_type;
  item.ctrl_info.dim_num = static_cast<uint32_t>(dim_size);
  item.ctrl_info.data_len = data_len;
  item.dims.clear();
  for (size_t i = 0UL; i < dim_size; ++i) {
    item.dims.push_back(dims[i]);
  }
  item.data_ptr = data_ptr;
  items.push_back(item);
  return Status::OK();
}

Status MappingTensors2DataItemInfos(acltdtTensorType acl_type, const std::vector<Tensor> &tensors,
                                    std::vector<DataItemInfo> &items,
                                    std::vector<std::unique_ptr<uint8_t[]>> &buff_list) {
  if (acl_type != ACL_TENSOR_DATA_TENSOR) {
    return AddDataItemInfo(acl_type, static_cast<int32_t>(ACL_BOOL), nullptr, 0UL, nullptr, 0UL, items);
  }

  for (auto &tensor : tensors) {
    aclDataType acl_data_type;
    TF_RETURN_IF_ERROR(MappingTfDtypeToAcl(tensor.dtype(), acl_data_type));
    auto dims = tensor.shape().dim_sizes();
    if (DataTypeCanUseMemcpy(tensor.dtype())) {
      TF_RETURN_IF_ERROR(AddDataItemInfo(ACL_TENSOR_DATA_TENSOR, static_cast<int32_t>(acl_data_type),
                                         (dims.empty() ? nullptr : reinterpret_cast<const int64_t *>(dims.data())),
                                         dims.size(), const_cast<char *>(tensor.tensor_data().data()),
                                         tensor.tensor_data().size(), items));
    } else if (tensor.dtype() == DT_STRING) {
      if (tensor.dims() == 0) {
        auto value = ge::PtrToPtr<char, tensorflow::tstring>(const_cast<char *>(tensor.tensor_data().data()));
        TF_RETURN_IF_ERROR(AddDataItemInfo(ACL_TENSOR_DATA_TENSOR, static_cast<int32_t>(ACL_STRING), nullptr, 0UL,
                                           const_cast<char *>(value->c_str()), value->size(), items));
      } else {
        uint8_t *data_ptr = nullptr;
        uint64_t data_size = 0UL;
        std::vector<int64_t> str_dims;
        TF_RETURN_IF_ERROR(GetDtStringTensorData(tensor, data_ptr, data_size, str_dims, buff_list));
        TF_RETURN_IF_ERROR(AddDataItemInfo(ACL_TENSOR_DATA_TENSOR, static_cast<int32_t>(ACL_STRING), str_dims.data(),
                                           str_dims.size(), data_ptr, data_size, items));
      }
    } else {
      return errors::Internal("unexpected data type ", DataTypeString(tensor.dtype()));
    }
  }
  return Status::OK();
}

Status SerializeDataItemInfo(std::vector<DataItemInfo> &items, void *&buff, const acltdtTensorType &acl_type) {
  size_t cnt = items.size();
  size_t total_size = 0UL;
  for (size_t i = 0UL; i < cnt; ++i) {
    items[i].ctrl_info.cur_cnt = static_cast<uint32_t>(i);
    items[i].ctrl_info.cnt = static_cast<uint32_t>(cnt);
    total_size += sizeof(ItemInfo) + items[i].ctrl_info.dim_num * sizeof(int64_t) + items[i].ctrl_info.data_len;
  }

  if (kIsHeterogeneous) {
    total_size += sizeof(RuntimeTensorDesc);
  }
  auto rt_error = rtMbufAlloc(&buff, total_size);
  if (rt_error != ACL_RT_SUCCESS) {
    return errors::Internal("call rtMbufAlloc with size[", total_size, "] failed, ret = ", rt_error);
  }

  void *data = nullptr;
  rt_error = rtMbufGetBuffAddr(buff, &data);
  if (rt_error != ACL_RT_SUCCESS) {
    (void)rtMbufFree(buff);
    return errors::Internal("call rtMbufAlloc with size[", total_size, "] failed, ret = ", rt_error);
  }

  void *head_buf = nullptr;
  uint64_t head_size = 0UL;
  rt_error = rtMbufGetPrivInfo(buff, &head_buf, &head_size);
  if (rt_error != ACL_RT_SUCCESS) {
    (void)rtMbufFree(buff);
    return errors::Internal("call rtMbufGetPrivInfo failed, ret = ", rt_error);
  }
  if ((head_buf != nullptr) && (head_size > kMbufHeadEndOfSequencePos)) {
    ADP_LOG(INFO) << "host queue set end_of_sequence mbuf head.";
    uint8_t *end_of_sequence =
        reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(head_buf) + kMbufHeadEndOfSequencePos);
    if (acl_type == ACL_TENSOR_DATA_END_OF_SEQUENCE) {
      *end_of_sequence = kEndOfSequenceFlag;
    } else {
      *end_of_sequence = 0U;
    }
  }

  if ((head_buf != nullptr) && (head_size >= sizeof(MsgInfo))) {
    MsgInfo *msg_info = reinterpret_cast<MsgInfo *>(static_cast<uint8_t *>(head_buf) + head_size - sizeof(MsgInfo));
    msg_info->ret_code = 0;
  }
  if (kIsHeterogeneous) {
    // need skip RuntimeTensorDesc, this data for GetNext op, so the RuntimeTensorDesc can be left blank
    data = ge::ValueToPtr(ge::PtrToValue(data) + sizeof(RuntimeTensorDesc));
  }
  size_t offset = 0UL;
  for (size_t i = 0UL; i < cnt; ++i) {
    // can not use memcpy_s here, data size may over 2G
    // total_size is calculate by item info, could not overflow here
    (void)memcpy(ge::ValueToPtr(ge::PtrToValue(data) + offset), &items[i].ctrl_info, sizeof(ItemInfo));
    offset += sizeof(ItemInfo);

    for (size_t j = 0UL; j < items[i].ctrl_info.dim_num; ++j) {
      (void)memcpy(ge::ValueToPtr(ge::PtrToValue(data) + offset), &(items[i].dims[j]), sizeof(int64_t));
      offset += sizeof(int64_t);
    }

    if (items[i].ctrl_info.data_len == 0UL) { continue; }

    (void)memcpy(ge::ValueToPtr(ge::PtrToValue(data) + offset), items[i].data_ptr, items[i].ctrl_info.data_len);
    offset += items[i].ctrl_info.data_len;
  }

  return Status::OK();
}
}  // namespace

Status HostQueueSetTransId(uint32_t queue_id, void *&buff) {
  void *head_buff = nullptr;
  uint64_t head_size = 0UL;
  const auto ret = rtMbufGetPrivInfo(buff, &head_buff, &head_size);
  NPU_REQUIRES(ret == ACL_RT_SUCCESS, errors::Internal("call rtMbufGetPrivInfo failed, ret = ", ret));
  if (head_size >= sizeof(MsgInfo)) {
    MsgInfo *msg_info = ge::PtrToPtr<char, MsgInfo>(static_cast<char *>(head_buff) + head_size - sizeof(MsgInfo));
    const std::lock_guard<std::mutex> lk(queue_id_to_trans_id_map_mutex);
    msg_info->trans_id = ++queue_id_to_trans_id_map[queue_id];
    ADP_LOG(INFO) << "host queue[" << queue_id << "] set trans id[" << msg_info->trans_id << "] success";
  }
  return Status::OK();
}

Status HostQueueInit(const std::string &name, const uint32_t &depth, uint32_t &queue_id) {
  TF_RETURN_IF_ERROR(CheckSymbols());
  std::map<std::string, std::string> init_options = NpuAttrs::GetInitOptions();
  NpuAttrs::LogOptions(init_options);
  GePlugin::GetInstance()->Init(init_options, false);

  NPU_REQUIRES(name.size() + 1 <= RT_MQ_MAX_NAME_LEN,
               errors::Internal("invalid queue name length[", name.size(), "], should less than 128"));

  NPU_REQUIRES(depth < kMaxQueueDepth,
               errors::Internal("invalid queue depth[", depth, "], should less than ", kMaxQueueDepth));

  auto rt_error = rtSetDevice(0);
  NPU_REQUIRES(rt_error == ACL_RT_SUCCESS, errors::Internal("call rtSetDevice device[0] failed, ret=", rt_error));
  ADP_LOG(INFO) << "call rtSetDevice device[0] success";

  rt_error = rtMemQueueInit(0);
  NPU_REQUIRES(((rt_error == ACL_RT_SUCCESS) || (rt_error == ACL_ERROR_RT_REPEATED_INIT)),
               errors::Internal("call rtMemQueueInit device[0] failed, ret=", rt_error));

  ADP_LOG(INFO) << "call rtMemQueueInit with device[0] success";

  rtMemQueueAttr_t attr = {};
  (void)memset_s(attr.name, RT_MQ_MAX_NAME_LEN, 0, RT_MQ_MAX_NAME_LEN);
  auto ret = memcpy_s(attr.name, RT_MQ_MAX_NAME_LEN, name.c_str(), name.size() + 1);
  NPU_REQUIRES(ret == EOK, errors::Internal("call memcpy_s queue name failed, ret=", ret));

  attr.depth = depth;
  attr.workMode = RT_MQ_MODE_DEFAULT;
  attr.flowCtrlFlag = false;
  attr.flowCtrlDropTime = 0U;
  attr.overWriteFlag = false;
  rt_error = rtMemQueueCreate(0, &attr, &queue_id);
  NPU_REQUIRES(rt_error == ACL_RT_SUCCESS, errors::Internal("call rtMemQueueCreate device[0] failed, ret=", rt_error));

  ADP_LOG(INFO) << "call rtMemQueueCreate with device[0] queue[" << queue_id << "] success";

  rtMemBuffCfg_t buff_cfg = {0};
  rt_error = rtMbufInit(&buff_cfg);
  NPU_REQUIRES(((rt_error == ACL_RT_SUCCESS) || (rt_error == ACL_ERROR_RT_REPEATED_INIT)),
               errors::Internal("call rtMbufInit failed, ret=", ret));

  const std::lock_guard<std::mutex> lk(queue_id_to_trans_id_map_mutex);
  (void) queue_id_to_trans_id_map.emplace(queue_id, 0UL);
  return Status::OK();
}

void HostQueueDestroy(const uint32_t &queue_id) {
  ADP_LOG(INFO) << "begin host queue: " << queue_id << " destroy";
  auto rt_error = rtSetDevice(0);
  if (rt_error != ACL_RT_SUCCESS) {
    ADP_LOG(ERROR) << "call rtSetDevice device[0] failed, ret=" << rt_error;
  }

  rt_error = rtMemQueueDestroy(0, queue_id);
  if (rt_error != ACL_RT_SUCCESS) {
    ADP_LOG(ERROR) << "call rtMemQueueDestroy device[0] queue[" << queue_id << "] failed, ret=" << rt_error;
  }

  const std::lock_guard<std::mutex> lk(queue_id_to_trans_id_map_mutex);
  (void) queue_id_to_trans_id_map.erase(queue_id);
}

Status MappingTensor2Buff(const acltdtTensorType &acl_type, const std::vector<tensorflow::Tensor> &tensors,
                          void *&buff) {
  std::vector<DataItemInfo> items;
  std::vector<std::unique_ptr<uint8_t[]>> buff_list;
  TF_RETURN_IF_ERROR(MappingTensors2DataItemInfos(acl_type, tensors, items, buff_list));
  TF_RETURN_IF_ERROR(SerializeDataItemInfo(items, buff, acl_type));
  return Status::OK();
}

Status HostQueueSendData(uint32_t queue_id, void *buff, bool &need_resend) {
  need_resend = false;
  auto rt_error = rtSetDevice(0);
  NPU_REQUIRES(rt_error == ACL_RT_SUCCESS, errors::Internal("call rtSetDevice device[0] failed, ret=", rt_error));
  rt_error = rtMemQueueEnQueue(0, queue_id, buff);
  if (rt_error == RT_ERROR_NONE) {
    return Status::OK();
  } else if (rt_error == ACL_ERROR_RT_QUEUE_FULL) {
    need_resend = true;
    ADP_LOG(INFO) << "queue[" << queue_id << "] is full, need call rtMemQueueEnQueue again";
  } else {
    HostQueueFreeBuff(buff);
    return errors::Internal("host enqueue queue[", queue_id, "] failed, ret = ", rt_error);
  }

  return Status::OK();
}

void HostQueueFreeBuff(void *buff) {
  auto rt_error = rtMbufFree(buff);
  if (rt_error != ACL_RT_SUCCESS) {
    ADP_LOG(ERROR) << "call rtMbufFree failed, ret=" << rt_error;
  }
}
}  // namespace tensorflow