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

#include "tensorflow/core/graph/algorithm.h"

#include "op_executors/npu_kernel_registry.h"
#include "npu_global.h"
#include "npu_utils.h"

namespace npu {
static const auto kernel = [](TFE_Context *context, NpuDevice *dev, const tensorflow::NodeDef &ndef, int num_inputs,
                              TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs,
                              TF_Status *status) {
  (void)ndef;
  (void)num_outputs;
  (void)num_inputs;
  (void)outputs;
  (void)context;
  TFE_TensorHandle *input = inputs[0];
  const tensorflow::Tensor *tensor;
  NPU_CTX_REQUIRES_OK(status, GetTensorHandleTensor(input, &tensor));
  auto handle = tensor->scalar<tensorflow::ResourceHandle>()();
  if (dev->MirroredIterator(handle)) {
    std::string channel_name = npu::WrapResourceName(handle.name());
    DLOG() << "Start destroy handle: " << channel_name;
    std::vector<std::shared_ptr<npu::HdcChannel>> channels;
    global::GlobalHdcChannel::GetInstance().Get(channel_name, channels);
    if (!channels.empty()) {
      global::GlobalHdcChannel::GetInstance().Destroy(channel_name);
    }
  }
};

NPU_REGISTER_FALLBACK_HOOK("DeleteIterator", kernel);
}