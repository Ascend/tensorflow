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

#include "tensorflow/core/platform/logging.h"

#include "npu_device.h"

namespace {
TFE_TensorHandle *CopyTensorToNpuDevice(TFE_Context *context, TFE_TensorHandle *tensor, TF_Status *status,
                                        void *device_info) {
  auto *dev = reinterpret_cast<NpuDevice *>(device_info);
  tensorflow::Status tf_status;
  LOG(INFO) << "[CopyTensorToNpuDevice] Copy tensor from " << tensorflow::unwrap(tensor)->DeviceName(&tf_status)
            << " to " << dev->device_name;
  TFE_TensorHandle *npu_handle = dev->CopyTensorH2D(context, tensor, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return npu_handle;
}

TFE_TensorHandle *CopyTensorFromNpuDevice(TFE_Context *context, TFE_TensorHandle *tensor,
                                          const char *target_device_name, TF_Status *status, void *device_info) {
  auto *dev = reinterpret_cast<NpuDevice *>(device_info);
  DLOG() << "[CopyTensorFromNpuDevice] Copy tensor from " << dev->device_name << " to " << target_device_name;
  // 输入的TensorHandle是NPU的，应当先进行NPU->CPU的传输，再调用TFE_TensorHandleCopyToDevice防止可能的NPU->GPU传输
  // 一旦Copy动作发生，需要进行stream同步。如果是NPU->NPU的拷贝（理论上不应该发生），可以不同步。
  TFE_TensorHandle *local_tensor = dev->CopyTensorD2H(context, tensor, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_TensorHandle *target_tensor = TFE_TensorHandleCopyToDevice(local_tensor, context, target_device_name, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  TFE_DeleteTensorHandle(local_tensor);
  return target_tensor;
}

void NpuDeviceExecute(const TFE_Op *op, int *num_outputs, TFE_TensorHandle **outputs, TF_Status *s, void *device_info) {
  auto *dev = reinterpret_cast<NpuDevice *>(device_info);
  dev->Execute(op, *num_outputs, outputs, s);
}

void DeleteNpuDevice(void *device_info) { NpuDevice::DeleteDevice(device_info); }

void RegisterNpuDevice(TFE_Context *context, const char *name, void *device_info, TF_Status *status) {
  TFE_CustomDevice custom_device;
  custom_device.copy_tensor_to_device = &CopyTensorToNpuDevice;
  custom_device.copy_tensor_from_device = &CopyTensorFromNpuDevice;
  custom_device.delete_device = &DeleteNpuDevice;
  custom_device.execute = &NpuDeviceExecute;
  TFE_RegisterCustomDevice(context, custom_device, name, device_info, status);
}

std::vector<NpuDevice *> devices_instances;
}  // namespace

/**
 * @breif: create device
 * @param context: context
 * @param name: device name
 * @param device_index: device index
 * @param device_options: device options
 */
extern std::string CreateDevice(TFE_Context *context, const char *name, int device_index,
                                const std::map<std::string, std::string> &device_options) {
  const static std::string kSucceed;

  NpuDevice *device = nullptr;
  auto create_status = NpuDevice::CreateDevice(name, device_index, device_options, &device);
  if (create_status != kSucceed) {
    return create_status;
  }
  devices_instances.push_back(device);

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  RegisterNpuDevice(context, name, device, status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    return std::string("Register Npu device ") + name + " failed:" + TF_Message(status.get());
  }
  LOG(INFO) << "Npu device instance " << name << " created";

  return kSucceed;
}

/**
 * @breif: release device resource
 */
extern void ReleaseDeviceResource() {
  for (auto device : devices_instances) {
    device->ReleaseResource();
  }
  devices_instances.clear();
}