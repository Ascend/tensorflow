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

#ifndef TENSORFLOW_KERNELS_OM_EXECUTOR_H_
#define TENSORFLOW_KERNELS_OM_EXECUTOR_H_

#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "tensorflow/core/framework/op_kernel.h"
#include "acl/acl.h"

namespace tensorflow {
class ModelProcess {
enum DynamicGearType {
  DYNAMIC_UNDEFINED = -1,
  DYNAMIC_BATCH,
  DYNAMIC_HW,
  DYNAMIC_DIMS
};

public:
  explicit ModelProcess(const std::string &model_data);

  ~ModelProcess();

  void SendRequest(const std::vector<Tensor> &inputs);

  void WaitReply(std::vector<Tensor> &outputs);

  Status GetThreadRet();

private:
  void StartWorkThread();

  Status PrepareProcess();

  bool IsDynamic(const aclmdlIODims &dims) const;

  Status GetDynamicGearInfo();

  Status LoadModelFromFile();

  Status CreateInput();

  Status CreateOutput();

  Status ProcessInput(const std::vector<Tensor> &inputs) const;

  Status ProcessDynamicGearInput(const std::vector<Tensor> &inputs) const;

  Status Execute(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs);

  Status ProcessOutput(std::vector<Tensor> &outputs);

  Status ProcessStaticOutput(const size_t index, const tensorflow::DataType tf_type, const aclDataBuffer *data_buf,
    std::vector<Tensor> &outputs) const;

  Status ProcessDynamicOutput(const size_t index, const tensorflow::DataType tf_type, aclDataBuffer *data_buf,
    std::vector<Tensor> &outputs) const;

  void WorkThread();

  Status MappingAclDtToTf(const aclDataType &acl_type, tensorflow::DataType &tf_type) const;

  Status MappingTfDtToAcl(const tensorflow::DataType tf_type, aclDataType &acl_type) const;

  void UnloadModel();

  void DestroyInput();

  void DestroyOutput();

  void DestroyResource();

private:
  std::string model_data_;
  uint32_t model_id_ = UINT32_MAX;
  uint32_t device_id_ = 0;
  aclmdlDesc *model_desc_ = nullptr;
  aclmdlDataset *input_ = nullptr;
  aclmdlDataset *output_ = nullptr;
  bool load_flag_ = false;
  bool is_set_device_ = false;
  std::vector<bool> is_input_dynamic_;
  std::vector<bool> is_output_dynamic_;

  std::atomic_bool run_flag_ {false};
  std::thread work_thread_;
  std::mutex mu_request_;
  std::atomic_bool request_flag_ {false};
  std::condition_variable cond_request_;
  std::mutex mu_reply_;
  std::atomic_bool reply_flag_ {false};
  std::condition_variable cond_reply_;

  Status thread_ret_;
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
  std::vector<bool> outputs_feed_nullptr_vec_;

  DynamicGearType dymainc_gear_type_ = DYNAMIC_UNDEFINED;
  std::vector<std::vector<uint64_t>> dynamic_gear_info_;
  // which inut is dynamic gear
  size_t dynamic_gear_input_index_ = SIZE_MAX;
  // which dim is dynamic gear in shape
  std::vector<size_t> dynamic_gear_shape_index_;
};

class OmExecutor {
public:
  /// \param model_data file of the om file
  /// \param executor Created om executor
  /// \return Status::OK() or error status if any error occurs
  static Status Create(const std::string &model_data, std::unique_ptr<OmExecutor> &executor);

  /// \param inputs Tensorflow host input tensors
  /// \param outputs Empty output tensors to be filling
  /// \return Status::OK() or error status if any error occurs
  Status Execute(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs);

private:
  OmExecutor() = default;
  std::unique_ptr<ModelProcess> model_process_;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_OM_EXECUTOR_H_
