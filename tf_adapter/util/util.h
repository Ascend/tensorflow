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

#ifndef TENSORFLOW_UTILS_H_
#define TENSORFLOW_UTILS_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "inc/tdt/data_common.h"
#include "tf_adapter/util/host_queue.h"

namespace tensorflow {
Status GetDtStringTensorData(const Tensor &tensor, uint8_t *&data_ptr, uint64_t &data_size,
                             std::vector<int64_t> &dims, std::vector<std::unique_ptr<uint8_t[]>> &buff_list);

Status MappingDTStringTensor2DataItem(const Tensor &tensor, tdt::DataItem &item,
                                      std::vector<std::unique_ptr<uint8_t[]>> &buff_list);

Status MappingDtStringTensor2AclDataItem(const Tensor &tensor, acltdtDataItem *&acl_data,
                                         std::vector<std::unique_ptr<uint8_t[]>> &buff_list);

bool IsWithoutNpuScope(const NodeDef &node_def);
bool IsWithoutNpuScope(const Node *node);
bool IsVariableOrResourceVariable(const Node * const node);
bool IsVariableExecuteOnHost(const Node * const node, const std::string &variable_location);
} // namespace tensorflow
#endif // TENSORFLOW_UTILS_H_