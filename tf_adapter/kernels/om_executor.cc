/*
* Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "om_executor.h"

namespace tensorflow {
OmExecutor::OmExecutor(const std::string &om_path) {}

Status OmExecutor::Create(const std::string &om_path, std::unique_ptr<OmExecutor> &executor) {
  // todo: OM 执行器创建逻辑
  executor.reset(new (std::nothrow) OmExecutor(om_path));
  if (executor == nullptr) {
    return errors::Internal("Failed create executor for om ", om_path);
  }
  return Status::OK();
}

Status OmExecutor::Execute(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  // todo: OM执行逻辑，in为TF的Host tensor，outputs为空vector
  return Status::OK();
}
}  // namespace tensorflow
