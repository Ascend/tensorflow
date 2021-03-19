/**
* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
* Description: Common depends and micro defines for and only for data preprocess module
*/

#ifndef TENSORFLOW_NPU_HDC_H
#define TENSORFLOW_NPU_HDC_H

#include <utility>

#include "acl/acl_tdt.h"
#include "tensorflow/core/framework/tensor.h"

#include "npu_micros.h"

class HdcChannel {
 public:
  static tensorflow::Status Create(uint32_t device_id, const std::string& name, std::shared_ptr<HdcChannel> *guarded_channel);

  ~HdcChannel();

  tensorflow::Status SendTensors(const std::vector<tensorflow::Tensor> &tensors);

  tensorflow::Status NotifyFinish();

  tensorflow::Status NotifyAbnormal();

  void Destroy();

 private:
  HdcChannel(uint32_t device_id, std::string name);
  tensorflow::Status Init();
  acltdtChannelHandle *handle_;
  int32_t device_id_;
  std::string name_;
  std::atomic_bool destroyed_{false};
};

#endif  //TENSORFLOW_NPU_HDC_H
