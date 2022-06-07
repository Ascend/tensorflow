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

#ifndef NPU_DEVICE_CORE_NPU_HDC_H
#define NPU_DEVICE_CORE_NPU_HDC_H

#include "acl/acl_tdt.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"

namespace npu {
struct HdcChannelHandle {
  bool is_mbuf;
  acltdtChannelHandle *acl_handle;
  HdcChannelHandle() : is_mbuf(false), acl_handle(nullptr) {}
};

class HdcChannel {
 public:
  static tensorflow::Status Create(uint32_t device_id, const std::string &name,
                                   std::shared_ptr<HdcChannel> *guarded_channel);

  static tensorflow::Status Create(uint32_t device_id, const std::string &name, size_t capacity,
                                   std::shared_ptr<HdcChannel> *guarded_channel);

  ~HdcChannel();

  tensorflow::Status SendTensors(const std::vector<tensorflow::Tensor> &tensors) const;

  tensorflow::Status RecvTensors(std::vector<tensorflow::Tensor> &tensors) const;

  tensorflow::Status NotifyFinish() const;

  tensorflow::Status NotifyAbnormal() const;

  void Destroy();

  bool IsNeedContinuousMem() const { return handle_.is_mbuf; }

 private:
  HdcChannel(uint32_t device_id, std::string name);

  HdcChannel(uint32_t device_id, std::string name, size_t capacity);

  tensorflow::Status Init();

  tensorflow::Status MappingTfDtypeToAcl(const tensorflow::DataType tf_type, aclDataType &acl_type) const;

  tensorflow::Status MappingAclDtypeToTf(const aclDataType &acl_type, tensorflow::DataType &tf_type) const;

  tensorflow::Status AssembleAclTensor2Tensor(acltdtDataItem *item, std::vector<tensorflow::Tensor> &tensors) const;

  tensorflow::Status AssembleAclDataset2Tensors(acltdtDataset *acl_dataset,
                                                std::vector<tensorflow::Tensor> &out_tensors) const;

  tensorflow::Status AssembleTensors2AclDataset(acltdtTensorType acl_type,
                                                const std::vector<tensorflow::Tensor> &tensors,
                                                acltdtDataset **output_acl_dataset) const;

  tensorflow::Status AssembleTensors2AclDataset(acltdtTensorType acl_type,
                                                const std::vector<tensorflow::Tensor> &tensors,
                                                acltdtDataset *acl_dataset) const;

  tensorflow::Status DestroyAclDataset(acltdtDataset *acl_dataset, bool include_data_item = true) const;

  tensorflow::Status RecvTensorByAcl(std::vector<tensorflow::Tensor> &tensors) const;

  tensorflow::Status SendTensorsByAcl(acltdtTensorType acl_type, const std::vector<tensorflow::Tensor> &tensors) const;

  HdcChannelHandle handle_;
  int32_t device_id_;
  std::string name_;
  bool limited_capacity_{false};
  size_t capacity_{0U};
  std::atomic_bool destroyed_{false};
  mutable std::vector<uint8_t> tensors_buffer_;
};
}  // end namespace npu

#endif  // NPU_DEVICE_CORE_NPU_HDC_H