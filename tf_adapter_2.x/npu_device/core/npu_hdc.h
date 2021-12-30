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

#ifndef TENSORFLOW_NPU_HDC_H
#define TENSORFLOW_NPU_HDC_H

#include "acl/acl_tdt.h"
#include "tensorflow/core/framework/tensor.h"
namespace npu {
class HdcChannel {
 public:
  static tensorflow::Status Create(uint32_t device_id, const std::string &name,
                                   std::shared_ptr<HdcChannel> *guarded_channel);

  ~HdcChannel();

  tensorflow::Status SendTensors(const std::vector<tensorflow::Tensor> &tensors) const;

  tensorflow::Status RecvTensors(std::vector<tensorflow::Tensor> &tensors) const;

  tensorflow::Status NotifyFinish() const;

  tensorflow::Status NotifyAbnormal() const;

  void Destroy();

 private:
  HdcChannel(uint32_t device_id, std::string name);
  tensorflow::Status Init();
  acltdtChannelHandle *handle_;
  int32_t device_id_;
  std::string name_;
  std::atomic_bool destroyed_{false};
};
}  // end namespace npu

#endif  // TENSORFLOW_NPU_HDC_H
