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

#ifndef TENSORFLOW_ACL_CHANNEL_H_
#define TENSORFLOW_ACL_CHANNEL_H_

#include "acl/acl_tdt.h"
#include "tensorflow/core/framework/tensor.h"
namespace tensorflow {
const static int32_t kTimeOut = 1000;
Status MappingTfDtypeToAcl(const tensorflow::DataType tf_type, aclDataType &acl_type);

Status MappingAclDtypeToTf(const aclDataType &acl_type, tensorflow::DataType &tf_type);

Status AssembleAclTensor2Tensor(acltdtDataItem *item, std::vector<Tensor> &tensors, bool call_by_channel_receive);

Status AssembleAclDataset2Tensors(acltdtDataset *acl_dataset, std::vector<Tensor> &out_tensors,
                                  bool call_by_channel_receive);

Status AssembleTensors2AclDataset(acltdtTensorType acl_type, const std::vector<Tensor> &tensors,
                                  acltdtDataset **acl_dataset);

Status AssembleTensors2AclDataset(acltdtTensorType acl_type, const std::vector<Tensor> &tensors,
                                  acltdtDataset *acl_dataset);

Status DestroyAclDataset(acltdtDataset *acl_dataset, bool include_data_item = true);

Status RecvTensorByAcl(acltdtChannelHandle *acl_handle, std::vector<Tensor> &tensors);

Status SendTensorsByAcl(const acltdtChannelHandle *acl_handle, acltdtTensorType acl_type, const std::vector<Tensor> &tensors,
                        bool &is_need_resend);

Status StopRecvTensorByAcl(acltdtChannelHandle **handle, const std::string channel_name);

acltdtChannelHandle *CreateAclTdtRecvChannel(uint32_t device_id, const std::string channel_name, const size_t capacity);

} // namespace tensorflow

#endif // TENSORFLOW_ACL_CHANNEL_H_
