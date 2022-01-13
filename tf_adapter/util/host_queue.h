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

#ifndef TENSORFLOW_HOST_QUEUE_H_
#define TENSORFLOW_HOST_QUEUE_H_

#include "acl/acl_tdt.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
Status HostQueueInit(const std::string &name, const uint32_t &depth, uint32_t &queue_id);

Status MappingTensor2Buff(const acltdtTensorType &acl_type, const std::vector<tensorflow::Tensor> &tensors,
                          void *&buff);

Status HostQueueSendData(uint32_t queue_id, void *buff, bool &need_resend);

void HostQueueFreeBuff(void *buff);

void HostQueueDestroy(const uint32_t &queue_id);

}  // namespace tensorflow
#endif  // TENSORFLOW_HOST_QUEUE_H_