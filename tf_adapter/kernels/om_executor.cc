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
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/npu_attrs.h"

namespace tensorflow {
ModelProcess::ModelProcess(const std::string &model_data) {
  model_data_ = model_data;
  StartWorkThread();
}

ModelProcess::~ModelProcess() {
  DestroyResource();
  run_flag_ = false;
  ADP_LOG(INFO) << "send request to thread";
  {
    std::unique_lock<std::mutex> lk {mu_request_};
    request_flag_ = true;
    cond_request_.notify_one();
  }
  ADP_LOG(INFO) << "start join";
  if (work_thread_.joinable()) {
    work_thread_.join();
  }
  ADP_LOG(INFO) << "join success";
}

void ModelProcess::DestroyResource() {
  UnloadModel();
  DestroyInput();
  DestroyOutput();
  if (is_set_device_) {
    aclrtResetDevice(device_id_);
    is_set_device_ = false;
  }
}

Status ModelProcess::PrepareProcess() {
  TF_RETURN_IF_ERROR(LoadModelFromFile());
  TF_RETURN_IF_ERROR(CreateInput());
  TF_RETURN_IF_ERROR(CreateOutput());
  return Status::OK();
}

bool ModelProcess::IsDynamic(const aclmdlIODims &dims) const {
  for (size_t i = 0; i < dims.dimCount; ++i) {
    if ((dims.dims[i] == -1) || (dims.dims[i] == -2)) {
      return true;
    }
  }
  return false;
}

Status ModelProcess::GetDynamicGearInfo() {
  aclmdlBatch dynamic_batch = {};
  REQUIRES_ACL_STATUS_OK(aclmdlGetDynamicBatch(model_desc_, &dynamic_batch), aclmdlGetDynamicBatch);
  if (dynamic_batch.batchCount > 0U) {
    dymainc_gear_type_ = DYNAMIC_BATCH;
    for (size_t i = 0U; i < dynamic_batch.batchCount; ++i) {
      std::vector<uint64_t> current_batch;
      current_batch.emplace_back(dynamic_batch.batch[i]);
      ADP_LOG(INFO) << "this "<< i << " batch is " << dynamic_batch.batch[i];
      dynamic_gear_info_.emplace_back(current_batch);
    }
    for (size_t j = 0U; j < is_input_dynamic_.size(); ++j) {
      if (is_input_dynamic_[j]) {
        // the first dynamic inut is enough
        dynamic_gear_input_index_ = j;
        break;
      }
    }
    ADP_LOG(INFO) << "dynamic gear input index is "<< dynamic_gear_input_index_;
    aclmdlIODims dims = {};
    REQUIRES_ACL_STATUS_OK(aclmdlGetInputDims(model_desc_, dynamic_gear_input_index_, &dims), aclmdlGetInputDims);
    for (size_t k = 0U; k < dims.dimCount; ++k) {
      if (dims.dims[k] == -1) {
        ADP_LOG(INFO) << "dynamic gear shape index is "<< k;
        dynamic_gear_shape_index_.emplace_back(k);
        break;
      }
    }
    if (dynamic_gear_shape_index_.size() != 1U) {
      ADP_LOG(ERROR) << "dynamic_gear_shape_index_ size is invalid, its value is " << dynamic_gear_shape_index_.size();
      return tensorflow::errors::Internal("get dynamic gear shape index fail");
    }
    return Status::OK();
  }
  return Status::OK();
}

Status ModelProcess::LoadModelFromFile() {
  const aclError acl_ret =  aclInit(nullptr);
  if ((acl_ret != ACL_SUCCESS) && (acl_ret != ACL_ERROR_REPEAT_INITIALIZE)) {
    return tensorflow::errors::Internal("aclInit fail");
  }
  (void)GetEnvDeviceID(device_id_);
  REQUIRES_ACL_STATUS_OK(aclrtSetDevice(device_id_), aclrtSetDevice);
  is_set_device_ = true;
  REQUIRES_ACL_STATUS_OK(aclmdlLoadFromMem(model_data_.c_str(), model_data_.size(), &model_id_), aclmdlLoadFromMem);
  load_flag_ = true;
  model_desc_ = aclmdlCreateDesc();
  REQUIRES_NOT_NULL(model_desc_);
  REQUIRES_ACL_STATUS_OK(aclmdlGetDesc(model_desc_, model_id_), aclmdlGetDesc);
  aclmdlIODims dims = {};
  for (size_t i = 0U; i < aclmdlGetNumInputs(model_desc_); ++i) {
    REQUIRES_ACL_STATUS_OK(aclmdlGetInputDims(model_desc_, i, &dims), aclmdlGetInputDims);
    is_input_dynamic_.emplace_back(IsDynamic(dims));
    ADP_LOG(INFO) << "this "<< i << " input is " << is_input_dynamic_[i];
  }
  for (size_t j = 0U; j < aclmdlGetNumOutputs(model_desc_); ++j) {
    REQUIRES_ACL_STATUS_OK(aclmdlGetOutputDims(model_desc_, j, &dims), aclmdlGetOutputDims);
    is_output_dynamic_.emplace_back(IsDynamic(dims));
    ADP_LOG(INFO) << "this "<< j << " output is " << is_output_dynamic_[j];
  }
  TF_RETURN_IF_ERROR(GetDynamicGearInfo());
  return Status::OK();
}

Status ModelProcess::CreateInput() {
  input_ = aclmdlCreateDataset();
  REQUIRES_NOT_NULL(input_);
  size_t input_num = aclmdlGetNumInputs(model_desc_);
  for (size_t i = 0U; i < input_num; ++i) {
    size_t input_size = aclmdlGetInputSizeByIndex(model_desc_, i);
    ADP_LOG(INFO) << "this "<< i << " input size is " << input_size;
    if (input_size == 0U) {
      ADP_LOG(ERROR) << "current " << i << " input is 0, can not get its real size";
      return tensorflow::errors::Internal("get input size is 0");
    }
    void *dev_ptr = nullptr;
    REQUIRES_ACL_STATUS_OK(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), aclrtMalloc);
    REQUIRES_NOT_NULL(dev_ptr);
    aclDataBuffer *data_buf = aclCreateDataBuffer(dev_ptr, input_size);
    if (data_buf == nullptr) {
      (void)aclrtFree(dev_ptr);
      return tensorflow::errors::Internal("aclCreateDataBuffer fail");
    }
    if (aclmdlAddDatasetBuffer(input_, data_buf) != ACL_SUCCESS) {
      (void)aclrtFree(dev_ptr);
      (void)aclDestroyDataBuffer(data_buf);
      return tensorflow::errors::Internal("aclmdlAddDatasetBuffer fail");
    }
  }
  return Status::OK();
}

Status ModelProcess::CreateOutput() {
  output_ = aclmdlCreateDataset();
  REQUIRES_NOT_NULL(output_);
  size_t output_num = aclmdlGetNumOutputs(model_desc_);
  for (size_t i = 0U; i < output_num; ++i) {
    size_t output_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
    ADP_LOG(INFO) << "this "<< i << " output size is " << output_size;
    if (output_size == 0U) {
      ADP_LOG(ERROR) << "current " << i << " output is 0, can not get its real size";
      return tensorflow::errors::Internal("get output size is 0");
    }
    void *dev_ptr = nullptr;
    REQUIRES_ACL_STATUS_OK(aclrtMalloc(&dev_ptr, output_size, ACL_MEM_MALLOC_NORMAL_ONLY), aclrtMalloc);
    REQUIRES_NOT_NULL(dev_ptr);
    aclDataBuffer *data_buf = aclCreateDataBuffer(dev_ptr, output_size);
    if (data_buf == nullptr) {
      (void)aclrtFree(dev_ptr);
      return tensorflow::errors::Internal("aclCreateDataBuffer fail");
    }
    if (aclmdlAddDatasetBuffer(output_, data_buf) != ACL_SUCCESS) {
      (void)aclrtFree(dev_ptr);
      (void)aclDestroyDataBuffer(data_buf);
      return tensorflow::errors::Internal("aclmdlAddDatasetBuffer fail");
    }
  }
  return Status::OK();
}

Status ModelProcess::Execute(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  TF_RETURN_IF_ERROR(ProcessInput(inputs));
  REQUIRES_ACL_STATUS_OK(aclmdlExecute(model_id_, input_, output_), aclmdlExecute);
  TF_RETURN_IF_ERROR(ProcessOutput(outputs));
  return Status::OK();
}

Status ModelProcess::ProcessDynamicGearInput(const std::vector<Tensor> &inputs) const {
  if (dymainc_gear_type_ == DYNAMIC_UNDEFINED) {
    return Status::OK();
  }
  if (dymainc_gear_type_ == DYNAMIC_BATCH) {
    if (inputs.size() <= dynamic_gear_input_index_) {
      ADP_LOG(ERROR) << "input size " << inputs.size() << " is invalid, need be larger than "
          << dynamic_gear_input_index_;
      return tensorflow::errors::Internal("input size %zu is invalid, need be larger than %zu",
          inputs.size(), dynamic_gear_input_index_);
    }
    auto dims = inputs[dynamic_gear_input_index_].shape().dim_sizes();
    if (dims.size() <= dynamic_gear_shape_index_[0]) {
      ADP_LOG(ERROR) << "input dim size " << dims.size() << " is invalid, need be larger than "
          << dynamic_gear_shape_index_[0];
      return tensorflow::errors::Internal("input dim size %zu is invalid, need be larger than %zu",
          dims.size(), dynamic_gear_shape_index_[0]);
    }
    const auto current_batch = dims[dynamic_gear_shape_index_[0]];
    size_t dynamic_batch_index = 0U;
    REQUIRES_ACL_STATUS_OK(aclmdlGetInputIndexByName(model_desc_, ACL_DYNAMIC_TENSOR_NAME, &dynamic_batch_index),
        aclmdlGetInputIndexByName);
    ADP_LOG(INFO) << "current batch is " << current_batch << ", shape index is " << dynamic_gear_shape_index_[0] <<
        ", dynamic_batch_index is " << dynamic_batch_index;
    REQUIRES_ACL_STATUS_OK(aclmdlSetDynamicBatchSize(model_id_, input_, dynamic_batch_index, current_batch),
        aclmdlSetDynamicBatchSize);
    return Status::OK();
  }
  return Status::OK();
}

Status ModelProcess::ProcessInput(const std::vector<Tensor> &inputs) const {
  const size_t model_input_size = aclmdlGetNumInputs(model_desc_);
  if (dymainc_gear_type_ == DYNAMIC_UNDEFINED) {
    if (inputs.size() < model_input_size) {
      ADP_LOG(ERROR) << "input num " << inputs.size() << " is smaller than model input num " << model_input_size;
      return tensorflow::errors::Internal("input num is invalid");
    }
  }
  // dynamic gear data need to be feeded alone
  size_t need_feed_input_cnts = ((dymainc_gear_type_ == DYNAMIC_UNDEFINED) && (model_input_size >= 1U))
      ? model_input_size : (model_input_size - 1U);
  for (size_t i = 0U; i < need_feed_input_cnts; ++i) {
    auto tensor_data = inputs[i].tensor_data().data();
    auto tensor_size = inputs[i].tensor_data().size();
    aclDataBuffer *data_buf = aclmdlGetDatasetBuffer(input_, i);
    REQUIRES_NOT_NULL(data_buf);
    void *dev_ptr = aclGetDataBufferAddr(data_buf);
    REQUIRES_NOT_NULL(dev_ptr);
    size_t cur_size = aclGetDataBufferSizeV2(data_buf);
    ADP_LOG(INFO) << "current input tensor is " << inputs[i].DebugString() << " model cur size is " << cur_size;
    if (tensor_size > cur_size) {
      ADP_LOG(ERROR) << "input " << i << " size " << tensor_size << " is larger than model input size " << cur_size;
      return tensorflow::errors::Internal("input size is too long");
    }
    REQUIRES_ACL_STATUS_OK(aclrtMemcpy(dev_ptr, cur_size,
        tensor_data, tensor_size, ACL_MEMCPY_HOST_TO_DEVICE), aclrtMemcpy);
    // set shpae
    tensorflow::DataType tf_type = inputs[i].dtype();
    aclDataType acl_dt = ACL_DT_UNDEFINED;
    TF_RETURN_IF_ERROR(MappingTfDtToAcl(tf_type, acl_dt));
    auto dims = inputs[i].shape().dim_sizes();
    aclTensorDesc *tensor_desc = aclCreateTensorDesc(acl_dt, dims.size(),
        (dims.empty() ? nullptr : reinterpret_cast<const int64_t *>(dims.data())), ACL_FORMAT_UNDEFINED);
    REQUIRES_NOT_NULL(tensor_desc);
    const aclError ret = aclmdlSetDatasetTensorDesc(input_, tensor_desc, i);
    aclDestroyTensorDesc(tensor_desc);
    tensor_desc = nullptr;
    REQUIRES_ACL_STATUS_OK(ret, aclmdlSetDatasetTensorDesc);
  }
  TF_RETURN_IF_ERROR(ProcessDynamicGearInput(inputs));
  return Status::OK();
}

Status ModelProcess::ProcessStaticOutput(const size_t index, const tensorflow::DataType tf_type, const void* dev_ptr,
    const size_t cur_size, std::vector<Tensor> &outputs) const {
  ADP_LOG(INFO) << "this out " << index << " is static";
  aclmdlIODims acl_dims = {};
  if (dymainc_gear_type_ == DYNAMIC_UNDEFINED) {
    REQUIRES_ACL_STATUS_OK(aclmdlGetOutputDims(model_desc_, index, &acl_dims), aclmdlGetOutputDims);
  } else {
    REQUIRES_ACL_STATUS_OK(aclmdlGetCurOutputDims(model_desc_, index, &acl_dims), aclmdlGetCurOutputDims);
  }
  TensorShape tf_shape;
  for (size_t j = 0U; j < acl_dims.dimCount; ++j) {
    tf_shape.AddDim(acl_dims.dims[j]);
  }
  Tensor tensor = Tensor(tf_type, tf_shape);
  auto tensor_data = const_cast<char *>(tensor.tensor_data().data());
  auto tensor_size = tensor.tensor_data().size();
  ADP_LOG(INFO) << "current output " << index << ", tensor is " << tensor.DebugString();
  REQUIRES_ACL_STATUS_OK(
      aclrtMemcpy(tensor_data, tensor_size, dev_ptr, tensor_size, ACL_MEMCPY_DEVICE_TO_HOST), aclrtMemcpy);
  outputs.emplace_back(std::move(tensor));
  return Status::OK();
}

Status ModelProcess::ProcessDynamicOutput(const size_t index, const tensorflow::DataType tf_type, const void* dev_ptr,
    const size_t cur_size, std::vector<Tensor> &outputs) const {
  ADP_LOG(INFO) << "this out " << index << " is dynamic";
  (void)cur_size;
  auto *desc = aclmdlGetDatasetTensorDesc(output_, index);
  REQUIRES_NOT_NULL(desc);
  size_t real_size = aclGetTensorDescSize(desc);
  TensorShape tf_shape;
  size_t shape_size = aclGetTensorDescNumDims(desc);
  ADP_LOG(INFO) << "get model output size is " << real_size << ", shape size is "
      << shape_size << " dt is " << tf_type;
  int64_t cur_dim = 0;
  for (size_t j = 0U; j < shape_size; ++j) {
    REQUIRES_ACL_STATUS_OK(aclGetTensorDescDimV2(desc, j, &cur_dim), aclGetTensorDescDimV2);
    tf_shape.AddDim(cur_dim);
  }
  Tensor tensor = Tensor(tf_type, tf_shape);
  auto tensor_data = const_cast<char *>(tensor.tensor_data().data());
  auto tensor_size = tensor.tensor_data().size();
  REQUIRES_ACL_STATUS_OK(
      aclrtMemcpy(tensor_data, tensor_size, dev_ptr, tensor_size, ACL_MEMCPY_DEVICE_TO_HOST), aclrtMemcpy);
  ADP_LOG(INFO) << "current output " << index << " tensor is " << tensor.DebugString();
  outputs.emplace_back(std::move(tensor));
  return Status::OK();
}

Status ModelProcess::ProcessOutput(std::vector<Tensor> &outputs) {
  outputs.clear();
  for (size_t i = 0U; i < aclmdlGetNumOutputs(model_desc_); ++i) {
    aclDataBuffer *data_buf = aclmdlGetDatasetBuffer(output_, i);
    REQUIRES_NOT_NULL(data_buf);
    void *dev_ptr = aclGetDataBufferAddr(data_buf);
    REQUIRES_NOT_NULL(dev_ptr);
    size_t cur_size = aclGetDataBufferSizeV2(data_buf);
    aclDataType acl_dt = aclmdlGetOutputDataType(model_desc_, i);
    tensorflow::DataType tf_type = DT_FLOAT;
    TF_RETURN_IF_ERROR(MappingAclDtToTf(acl_dt, tf_type));
    ADP_LOG(INFO) << "model output size is " << cur_size << ", dt is " << tf_type;
    if (!is_output_dynamic_[i]) {
      TF_RETURN_IF_ERROR(ProcessStaticOutput(i, tf_type, dev_ptr, cur_size, outputs));
    } else {
      TF_RETURN_IF_ERROR(ProcessDynamicOutput(i, tf_type, dev_ptr, cur_size, outputs));
    }
  }
  return Status::OK();
}

void ModelProcess::WorkThread() {
  ADP_LOG(INFO) << "start work thread " << this;
  bool is_prepared = false;
  while (run_flag_) {
    {
      std::unique_lock<std::mutex> lk {mu_request_};
      ADP_LOG(INFO) << "start wait request";
      cond_request_.wait(lk, [this] { return (!this->run_flag_.load() || this->request_flag_.load()); });
      request_flag_ = false;
    }
    ADP_LOG(INFO) << "get request, start working";
    if (!run_flag_) {
      ADP_LOG(INFO) << "stop thread";
      return;
    }
    if (!is_prepared) {
      thread_ret_ = PrepareProcess();
      if (thread_ret_ == Status::OK()) {
        is_prepared = true;
      } else {
        DestroyResource();
        ADP_LOG(ERROR) << "prepare fail";
        {
          std::unique_lock<std::mutex> lk {mu_reply_};
          reply_flag_ = true;
          cond_reply_.notify_one();
        }
        continue;
      }
    }
    thread_ret_ = Execute(inputs_, outputs_);
    ADP_LOG(INFO) << "execute end " << thread_ret_.ToString();
    {
      std::unique_lock<std::mutex> lk {mu_reply_};
      reply_flag_ = true;
      cond_reply_.notify_one();
    }
  }
}

void ModelProcess::StartWorkThread() {
  run_flag_ = true;
  work_thread_ = std::thread(&ModelProcess::WorkThread, this);
}

Status ModelProcess::MappingTfDtToAcl(const tensorflow::DataType tf_type, aclDataType &acl_type) const {
  const static std::map<tensorflow::DataType, aclDataType> type_mapping = {
      {DT_FLOAT, ACL_FLOAT},  {DT_HALF, ACL_FLOAT16},  {DT_INT8, ACL_INT8},     {DT_INT32, ACL_INT32},
      {DT_UINT8, ACL_UINT8},  {DT_INT16, ACL_INT16},   {DT_UINT16, ACL_UINT16}, {DT_UINT32, ACL_UINT32},
      {DT_INT64, ACL_INT64},  {DT_UINT64, ACL_UINT64}, {DT_DOUBLE, ACL_DOUBLE}, {DT_BOOL, ACL_BOOL},
      {DT_STRING, ACL_STRING}};
  auto found = type_mapping.find(tf_type);
  if (found == type_mapping.end()) {
    return errors::Internal("Unsupported tf data type", DataTypeString(tf_type), " by acl.");
  }
  acl_type = found->second;
  return Status::OK();
}

Status ModelProcess::MappingAclDtToTf(const aclDataType &acl_type, tensorflow::DataType &tf_type) const {
  const static std::map<aclDataType, tensorflow::DataType> type_mapping = {
      {ACL_FLOAT, DT_FLOAT},  {ACL_FLOAT16, DT_HALF},  {ACL_INT8, DT_INT8},     {ACL_INT32, DT_INT32},
      {ACL_UINT8, DT_UINT8},  {ACL_INT16, DT_INT16},   {ACL_UINT16, DT_UINT16}, {ACL_UINT32, DT_UINT32},
      {ACL_INT64, DT_INT64},  {ACL_UINT64, DT_UINT64}, {ACL_DOUBLE, DT_DOUBLE}, {ACL_BOOL, DT_BOOL},
      {ACL_STRING, DT_STRING}};
  auto found = type_mapping.find(acl_type);
  if (found == type_mapping.end()) {
    return errors::Internal("Acl channel receive unsupported data type", acl_type);
  }
  tf_type = found->second;
  return Status::OK();
}

void ModelProcess::UnloadModel() {
  if (!load_flag_) {
    return;
  }
  (void)aclmdlUnload(model_id_);

  if (model_desc_ != nullptr) {
    (void)aclmdlDestroyDesc(model_desc_);
    model_desc_ = nullptr;
  }
  load_flag_ = false;
}

void ModelProcess::DestroyInput() {
  if (input_ == nullptr) {
    return;
  }

  for (size_t i = 0U; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
    void *data = aclGetDataBufferAddr(dataBuffer);
    (void)aclrtFree(data);
    (void)aclDestroyDataBuffer(dataBuffer);
  }
  (void)aclmdlDestroyDataset(input_);
  input_ = nullptr;
}

void ModelProcess::DestroyOutput() {
  if (output_ == nullptr) {
    return;
  }
  for (size_t i = 0U; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
    void* data = aclGetDataBufferAddr(dataBuffer);
    (void)aclrtFree(data);
    (void)aclDestroyDataBuffer(dataBuffer);
  }
  (void)aclmdlDestroyDataset(output_);
  output_ = nullptr;
}

void ModelProcess::SendRequest(const std::vector<Tensor> &inputs) {
  inputs_ = inputs;
  ADP_LOG(INFO) << "send request to thread";
  {
    std::unique_lock<std::mutex> lk {mu_request_};
    request_flag_ = true;
    cond_request_.notify_one();
  }
}

void ModelProcess::WaitReply(std::vector<Tensor> &outputs) {
  {
    std::unique_lock<std::mutex> lk {mu_reply_};
    cond_reply_.wait(lk, [this] { return this->reply_flag_.load(); });
    reply_flag_ = false;
  }
  outputs = outputs_;
}

Status ModelProcess::GetThreadRet() {
  return thread_ret_;
}

Status OmExecutor::Create(const std::string &model_data, std::unique_ptr<OmExecutor> &executor) {
  executor.reset(new (std::nothrow) OmExecutor());
  if (executor == nullptr) {
    return errors::Internal("Failed create executor for om");
  }
  executor->model_process_ = std::unique_ptr<ModelProcess> (new (std::nothrow) ModelProcess(model_data));
  REQUIRES_NOT_NULL(executor->model_process_);
  return Status::OK();
}

Status OmExecutor::Execute(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  ADP_LOG(INFO) << "send request start";
  model_process_->SendRequest(inputs);
  ADP_LOG(INFO) << "send request end, start wait thread reply";
  model_process_->WaitReply(outputs);
  auto ret = model_process_->GetThreadRet();
  ADP_LOG(INFO) << "execute end " << ret.ToString();
  return ret;
}
} // namespace tensorflow
