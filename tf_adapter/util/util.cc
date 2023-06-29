/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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

#include "tf_adapter/util/util.h"

#include <numeric>
#include <vector>
#include <string>
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/common/compat_tf1_tf2.h"
#include "inc/metadef/inc/graph/def_types.h"
#include "securec.h"
namespace tensorflow {
namespace {
const std::string ATTR_VALUE_SCOPE_NAME = "_without_npu_compile";
}

Status GetDtStringTensorData(const Tensor &tensor, uint8_t *&data_ptr, uint64_t &data_size,
                             std::vector<int64_t> &dims, std::vector<std::unique_ptr<uint8_t[]>> &buff_list) {
  for (int i = 0; i < tensor.dims(); ++i) { dims.emplace_back(tensor.dim_size(i)); }
  int64_t total_nums = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
  uint64_t total_size = 0UL;
  for (int64_t i = 0; i < total_nums; ++i) { total_size += tensor.flat<tstring>()(i).size(); }
  uint64_t buff_size = sizeof(ge::StringHead) * static_cast<uint64_t>(total_nums) + total_size;
  std::unique_ptr<uint8_t[]> buffer(new (std::nothrow) uint8_t[buff_size]);
  REQUIRES_NOT_NULL(buffer);
  buff_list.push_back(std::move(buffer));

  uint8_t *base_ptr = buff_list.back().get();
  uint64_t offset = sizeof(ge::StringHead) * static_cast<uint64_t>(total_nums);
  for (uint64_t i = 0UL; i < static_cast<uint64_t>(total_nums); ++i) {
    ge::StringHead *head = ge::PtrToPtr<uint8_t, ge::StringHead>(base_ptr + i * sizeof(ge::StringHead));
    head->addr = offset;
    head->len = tensor.flat<tstring>()(i).size();
    // Skip data copy for empty tensor
    if (head->len == 0ULL) { continue; }
    auto status = LoopCopy(ge::PtrToPtr<uint8_t, char>(base_ptr + offset), (buff_size - offset),
                           const_cast<char *>(tensor.flat<tstring>()(i).data()), head->len);
    if (!status.ok()) { return status; }
    offset += head->len;
  }
  data_ptr = buff_list.back().get();
  data_size = buff_size;
  return Status::OK();
}

Status MappingDTStringTensor2DataItem(const Tensor &tensor, tdt::DataItem &item,
                                      std::vector<std::unique_ptr<uint8_t[]>> &buff_list) {
  if (tensor.dims() == 0) {
    std::string value = tensor.scalar<npu::compat_tf1_tf2::string>()();
    item.dataLen_ = tensor.scalar<npu::compat_tf1_tf2::string>()().size();
    item.dataPtr_ = std::shared_ptr<void>(const_cast<char *>(value.data()), [](const void *elem) {
      (void)elem;
    });
    return Status::OK();
  }

  uint8_t *data_ptr = nullptr;
  uint64_t data_size = 0UL;
  std::vector<int64_t> dims;
  TF_RETURN_IF_ERROR(GetDtStringTensorData(tensor, data_ptr, data_size, dims, buff_list));
  item.dataPtr_ = std::shared_ptr<void>(data_ptr, [](const void *ptr) {
    (void)ptr;
  });
  item.dataLen_ = data_size;
  return Status::OK();
}

Status MappingDtStringTensor2AclDataItem(const Tensor &tensor, acltdtDataItem *&acl_data,
                                         std::vector<std::unique_ptr<uint8_t[]>> &buff_list) {
  if (tensor.dims() == 0) {
    auto value = ge::PtrToPtr<char, tensorflow::tstring>(const_cast<char *>(tensor.tensor_data().data()));
    // for scalar type, *dims is nullptr and dim_num is 0
    acl_data = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, nullptr, 0, ACL_STRING,
                                    const_cast<char *>(value->c_str()), value->size());
    return Status::OK();
  }

  uint8_t *data_ptr = nullptr;
  uint64_t data_size = 0UL;
  std::vector<int64_t> dims;
  TF_RETURN_IF_ERROR(GetDtStringTensorData(tensor, data_ptr, data_size, dims, buff_list));
  acl_data = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, dims.data(), dims.size(),
                                  ACL_STRING, data_ptr, data_size);
  return Status::OK();
}

Status LoopCopy(char *dst_ptr, size_t dst_size, char *src_ptr, size_t src_size) {
  if (dst_size < src_size) {
    return tensorflow::errors::Internal("Loop memory copy failed. dst_size:", dst_size, ", src_size:", src_size);
  }

  size_t copy_size = 0UL;
  size_t org_src_size = src_size;
  do {
    size_t src_copy_size = (src_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : src_size;
    if (memcpy_s(dst_ptr, src_copy_size, src_ptr, src_copy_size) != EOK) {
      return tensorflow::errors::Internal("Loop memory copy failed , dst_size:", src_copy_size,
                                          ", src_size:", src_copy_size);
    }
    copy_size += src_copy_size;
    dst_ptr += src_copy_size;
    src_ptr += src_copy_size;
    src_size -= src_copy_size;
  } while (copy_size < org_src_size);
  return tensorflow::Status::OK();
}

bool IsWithoutNpuScope(const NodeDef &node_def) {
  if (node_def.attr().count(ATTR_VALUE_SCOPE_NAME) > 0) { return node_def.attr().at(ATTR_VALUE_SCOPE_NAME).b(); }
  return false;
}

bool IsWithoutNpuScope(const Node *node) {
  return IsWithoutNpuScope(node->def());
}

bool IsVariableOrResourceVariable(const Node * const node) {
  if (node->IsVariable() || node->type_string() == "VarHandleOp") {
    return true;
  }
  return false;
}

bool IsVariableExecuteOnHost(const Node * const node, const std::string &variable_location) {
  if (variable_location == "Host" && IsVariableOrResourceVariable(node)) {
    return true;
  }
  return false;
}
} // namespace tensorflow