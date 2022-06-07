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

#include "npu_logger.h"
#include "npu_micros.h"

namespace npu {
tensorflow::Status NpuStdoutReceiver::Start() {
  std::unique_lock<std::mutex> lk(mu_);
  if (started_) {
    LOG(INFO) << "Npu stdout receiver has already started on device " << device_id_;
    return tensorflow::Status::OK();
  }
  const static size_t kNpuCerrChannelCapacity = 32U;
  NPU_REQUIRES_OK(npu::HdcChannel::Create(device_id_, "_npu_log", kNpuCerrChannelCapacity, &channel_));
  std::thread t([this]() {
    while (true) {
      std::vector<tensorflow::Tensor> tensors;
      auto status = channel_->RecvTensors(tensors);
      if (stopping_) {
        DLOG() << "Exit npu stdout receive thread of device " << device_id_ << " as stopping";
        break;
      }
      if (!status.ok()) {
        LOG(ERROR) << "Npu stdout receiver on device " << device_id_ << " error " << status.error_message();
        break;
      }
      for (auto &tensor : tensors) {
        LOG(INFO) << "[NPU:" << device_id_ << "] " << tensor.DebugString();
      }
    }
    DLOG() << "Npu stdout receive thread of device " << device_id_ << " exited";
  });
  thread_.swap(t);
  started_ = true;
  LOG(INFO) << "Npu stdout receiver of device " << device_id_ << " started";
  return tensorflow::Status::OK();
}

tensorflow::Status NpuStdoutReceiver::Stop() {
  std::unique_lock<std::mutex> lk(mu_);
  if (!started_) {
    return tensorflow::Status::OK();
  }
  LOG(INFO) << "Stopping npu stdout receiver of device " << device_id_;
  (void)stopping_.exchange(true);
  channel_->Destroy();
  thread_.join();
  started_ = false;
  DLOG() << "Npu stdout receiver of device " << device_id_ << " stopped";
  return tensorflow::Status::OK();
}
}  // namespace npu
