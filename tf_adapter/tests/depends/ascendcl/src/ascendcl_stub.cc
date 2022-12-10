/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "acl/acl_tdt.h"
#include "ascendcl_stub.h"
#include "acl/acl.h"
#include "securec.h"
#include <map>
#include <mutex>
#include "tf_adapter/common/adapter_logger.h"

namespace {
    constexpr uint32_t kDeviceSatModeLimit = 2U;
    std::mutex aclChannleMutex;
    std::map<std::string, acltdtChannelHandle *> aclChannleMap;
    std::map<std::string, aclDataType> aclDataTypeStrMap =
    {
        {"bool",     ACL_BOOL},
        {"int8",     ACL_INT8},
        {"uint8",    ACL_UINT8},
        {"half",     ACL_FLOAT16},
        {"int16",    ACL_INT16},
        {"uint16",   ACL_UINT16},
        {"float",    ACL_FLOAT},
        {"int32",    ACL_INT32},
        {"uint32",   ACL_UINT32},
        {"int64",    ACL_INT64},
        {"uint64",   ACL_UINT64},
        {"double",   ACL_DOUBLE},
        {"string",   ACL_STRING}
    };
}

namespace acl {
    void GetTensorDimsString(const int64_t *dims, size_t dimNum, std::string &dimsStr)
    {
        for (size_t i = 0; i < dimNum; ++i) {
            dimsStr += std::to_string(dims[i]);
            if (i + 1 == dimNum) {
                break;
            }
            dimsStr.push_back(',');
        }
        dimsStr += "]";
    }
}

aclError acltdtDestroyChannel(acltdtChannelHandle *handle) {
    if (handle == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

acltdtChannelHandle *acltdtCreateChannel(uint32_t deviceId, const char *name) {
    acltdtChannelHandle *handle = new(std::nothrow) acltdtChannelHandle(deviceId, name);
    {
        std::unique_lock<std::mutex> lk(aclChannleMutex);
        aclChannleMap[name] = handle;
    }
    return handle;
}

aclError aclrtSetDevice(int32_t deviceId){
    return ACL_SUCCESS;
}

aclError aclrtMallocHost(void **hostPtr, size_t size) {
    (*hostPtr) = std::malloc(size);
    return ACL_SUCCESS;
}

aclError aclrtMemcpy(void *dst, size_t destMax,
                     const void *src, size_t count,
                     aclrtMemcpyKind kind) {
    auto ret = memcpy_s(dst, destMax, src, count);
    if (ret != EOK) {
        return ACL_ERROR_BAD_ALLOC;
    }
    return ACL_SUCCESS;
}

aclError aclrtFreeHost(void *hostPtr) {
    free(hostPtr);
    return ACL_SUCCESS;
}

aclError aclrtResetDevice(int32_t deviceId) {
    return ACL_SUCCESS;
}

acltdtChannelHandle *acltdtCreateChannelWithCapacity(uint32_t deviceId,
                                                     const char *name,
                                                     size_t capacity) {
    acltdtChannelHandle *handle = new(std::nothrow) acltdtChannelHandle(deviceId, name);
    {
        std::unique_lock<std::mutex> lk(aclChannleMutex);
        aclChannleMap[name] = handle;
    }
    return handle;
}

acltdtDataItem *acltdtGetDataItem(const acltdtDataset *dataset, size_t index) {
  if ((dataset == nullptr) || (index >= dataset->blobs.size())) {
      return nullptr;
  }
  return dataset->blobs[index];
}

aclError acltdtDestroyDataItem(acltdtDataItem *dataItem) {
  if (dataItem == nullptr) {
      return ACL_ERROR_INVALID_PARAM;
  }
  delete dataItem;
  return ACL_SUCCESS;
}

size_t acltdtGetDatasetSize(const acltdtDataset *dataset) {
  if (dataset == nullptr) {
      return 0;
  }
  return dataset->blobs.size();
}

aclError acltdtDestroyDataset(acltdtDataset *dataset) {
  if (dataset == nullptr) {
      return ACL_ERROR_INVALID_PARAM;
  }
  delete dataset;
  return ACL_SUCCESS;
}

acltdtDataset *acltdtCreateDataset() {
    return new(std::nothrow) acltdtDataset();
}

aclError acltdtReceiveTensor(const acltdtChannelHandle *handle,
                             acltdtDataset *dataset,
                             int32_t timeout) {
    if (handle->recvName.empty() && handle->name.empty()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (handle->recvName == "train" || handle->name == "train") {
        acltdtDataItem *acl_data = acltdtCreateDataItem(ACL_TENSOR_DATA_END_OF_SEQUENCE, nullptr, 0, ACL_BOOL /* whatever */, nullptr, 0);
        if (acltdtAddDataItem(dataset, acl_data) != ACL_ERROR_NONE) {
            if (acltdtDestroyDataItem(acl_data) != ACL_ERROR_NONE) {
                return ACL_ERROR_FAILURE;
            }
        }
    } else {
        std::string vaue_str = "print message!!";
        std::string *value = &vaue_str;
        // for scalar type, *dims is nullptr and dim_num is 0
        acltdtDataItem *acl_data = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, nullptr, 0, ACL_STRING,
                                      const_cast<char *>(value->c_str()), value->size());
        if (acltdtAddDataItem(dataset, acl_data) != ACL_ERROR_NONE) {
            if (acltdtDestroyDataItem(acl_data) != ACL_ERROR_NONE) {
                return ACL_ERROR_FAILURE;
            }
        }
        int32_t value_int = 1;
        acltdtDataItem *acl_int_data = acltdtCreateDataItem(
            ACL_TENSOR_DATA_TENSOR, nullptr,
            0, ACL_INT32, &value_int, 4);
        if (acltdtAddDataItem(dataset, acl_int_data) != ACL_ERROR_NONE) {
            if (acltdtDestroyDataItem(acl_int_data) != ACL_ERROR_NONE) {
                return ACL_ERROR_FAILURE;
            }
        }
    }
    return ACL_SUCCESS;
}

acltdtDataItem *acltdtCreateDataItem(acltdtTensorType tdtType,
                                     const int64_t *dims,
                                     size_t dimNum,
                                     aclDataType dataType,
                                     void *data,
                                     size_t size) {
    if ((dims == nullptr && dimNum != 0) || (dims != nullptr && dimNum == 0)) {
        return nullptr;
    }
    std::string dimsStr = "[";
    acl::GetTensorDimsString(dims, dimNum, dimsStr);
    std::string typeStr;
    for (const auto &item: aclDataTypeStrMap) {
        if (item.second == dataType) {
            typeStr = item.first;
            break;
        }
    }
    if (typeStr.empty()) {
        return nullptr;
    }
    std::shared_ptr<void> dataPtr;
    dataPtr.reset(data, [](const void *p) {});
    return new(std::nothrow) acltdtDataItem(tdtType, dims, dimNum, dimsStr, dataType, typeStr, dataPtr, size);
}

aclError acltdtAddDataItem(acltdtDataset *dataset, acltdtDataItem *dataItem) {
    if (dataset == nullptr || dataItem == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (dataset->freeSelf) {
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }
    dataset->blobs.push_back(dataItem);
    return ACL_SUCCESS;
}

aclError acltdtSendTensor(const acltdtChannelHandle *handle,
                          const acltdtDataset *dataset,
                          int32_t timeout) {
    if (dataset == nullptr || handle == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

acltdtTensorType acltdtGetTensorTypeFromItem(const acltdtDataItem *dataItem) {
    if (dataItem == nullptr) {
        return ACL_TENSOR_DATA_UNDEFINED;
    }
    return dataItem->tdtType;
}

aclDataType acltdtGetDataTypeFromItem(const acltdtDataItem *dataItem) {
    if (dataItem == nullptr) {
        return ACL_DT_UNDEFINED;
    }
    return dataItem->dataType;
}

size_t acltdtGetDimNumFromItem(const acltdtDataItem *dataItem) {
    if (dataItem == nullptr) {
        return 0;
    }
    return dataItem->dims.size();
}

size_t acltdtGetDataSizeFromItem(const acltdtDataItem *dataItem) {
    if (dataItem == nullptr) {
        return 0;
    }
    return dataItem->dataLen;
}

void *acltdtGetDataAddrFromItem(const acltdtDataItem *dataItem) {
    if (dataItem == nullptr) {
        return nullptr;
    }
    return dataItem->dataPtr.get();
}

aclError acltdtGetDimsFromItem(const acltdtDataItem *dataItem, int64_t *dims, size_t dimNum) {
    if (dataItem == nullptr) {
        return ACL_TENSOR_DATA_UNDEFINED;
    }
    // check dims and dimNum
    if ((dims == nullptr && dimNum != 0) || (dims != nullptr && dimNum == 0)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    if (dimNum < dataItem->dims.size()) {
        return ACL_ERROR_INVALID_PARAM;
    }

    for (size_t i = 0; i < dataItem->dims.size(); ++i) {
        dims[i] = dataItem->dims[i];
    }

    return ACL_SUCCESS;
}

aclError acltdtStopChannel(acltdtChannelHandle *handle)
{
    if (handle == nullptr) {
        return ACL_TENSOR_DATA_UNDEFINED;
    }
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream *stream) {
  ADP_LOG(INFO) << "aclrtCreateStream stub enter";
  AclStreamStub *stream_ = new (std::nothrow)AclStreamStub();
  if (stream_ == nullptr) {
    ADP_LOG(INFO) << "new AclStreamStub failed";
    *stream = nullptr;
    return ACL_ERROR_INVALID_PARAM;
  }
  stream_->status = ACL_EVENT_STATUS_NOT_READY;
  *stream = stream_;
  ADP_LOG(INFO) << "aclrtCreateStream stub out, stream_ = " << stream_;
  return ACL_ERROR_NONE;
}

aclError aclrtDestroyStream(aclrtStream stream) {
  ADP_LOG(INFO) << "aclrtDestroyStream stub enter";
  delete (AclStreamStub*)stream;
  ADP_LOG(INFO) << "aclrtDestroyStream stub out";
  return ACL_ERROR_NONE;
}

aclError aclrtCreateEvent(aclrtEvent *event) {
  ADP_LOG(INFO) << "aclrtCreateEvent stub enter";
  *event = new (std::nothrow)AclEventStub();
  ADP_LOG(INFO) << "aclrtCreateEvent stub out";
  return ACL_ERROR_NONE;
}

aclError aclrtDestroyEvent(aclrtEvent event) {
  ADP_LOG(INFO) << "aclrtDestroyEvent stub enter";
  delete (AclEventStub*)(event);
  ADP_LOG(INFO) << "aclrtDestroyEvent stub out";
  return ACL_ERROR_NONE;
}

const char *aclrtGetSocName() {
  return "Ascend910B";
}

// for GE RunGraph api
#if 0
aclError aclrtSynchronizeStream(aclrtStream stream) {
  ADP_LOG(INFO) << "aclrtSynchronizeStream stub enter, stream = " << stream;
  AclStreamStub *stub = static_cast<AclStreamStub*>(stream);
  if (stub->hook != nullptr) {
      ADP_LOG(INFO) << "aclrtSynchronizeStream:: stream = " << stream << ", process hook = "
            << stub->hook.target<ge::Status(*)(const std::vector<ge::Tensor> &input_data, std::vector<ge::Tensor> &output_data)>();
      return stub->hook(stub->input_data, *stub->output_data);
  }
  ADP_LOG(INFO) << "aclrtSynchronizeStream stub out, stream = " << stream;
  return ACL_ERROR_NONE;
}

aclError aclrtSynchronizeEvent(aclrtEvent event) {
  ADP_LOG(INFO) << "aclrtSynchronizeEvent stub enter, event = " << event;
  AclEventStub *stub = static_cast<AclEventStub*>(event);
  if (stub->hook != nullptr) {
    ADP_LOG(INFO) << "aclrtSynchronizeEvent:: event = " << event << ", process hook = "
        << stub->hook.target<ge::Status(*)(const std::vector<ge::Tensor> &input_data, std::vector<ge::Tensor> &output_data)>();
    (void)stub->hook(stub->input_data, *stub->output_data);
  }
  ADP_LOG(INFO) << "aclrtSynchronizeEvent stub out, event = " << event;
  return ACL_ERROR_NONE;
}

aclError aclrtQueryEvent(aclrtEvent event, aclrtEventStatus *status) {
  ADP_LOG(INFO) << "aclrtQueryEvent stub enter, event = " << event;
  AclEventStub *stub = static_cast<AclEventStub*>(event);
  *status = stub->status;
  if (stub->status == ACL_EVENT_STATUS_COMPLETE) {
    ADP_LOG(INFO) << "aclrtQueryEvent:: event = " << event << ", process hook = "
        << stub->hook.target<ge::Status(*)(const std::vector<ge::Tensor> &input_data, std::vector<ge::Tensor> &output_data)>();
    (void)stub->hook(stub->input_data, *stub->output_data);
  }
  ADP_LOG(INFO) << "aclrtQueryEvent stub out, event = " << event;
  return ACL_ERROR_NONE;
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
  AclStreamStub *stubStream = static_cast<AclStreamStub*>(stream);
  AclEventStub *stubEvent = static_cast<AclEventStub*>(event);
  ADP_LOG(INFO) << "aclrtRecordEvent stub enter, stream = " << stream
      << ", event = " << event << ", process hook = "
      << stubStream->hook.target<ge::Status(*)(const std::vector<ge::Tensor> &input_data, std::vector<ge::Tensor> &output_data)>();
  stubEvent->input_data = stubStream->input_data;
  stubEvent->output_data = stubStream->output_data;
  stubEvent->hook = stubStream->hook;
  stubEvent->status = stubStream->status;
  ADP_LOG(INFO) << "aclrtRecordEvent stub out, event = " << event << ", process hook = "
        << stubEvent->hook.target<ge::Status(*)(const std::vector<ge::Tensor> &input_data, std::vector<ge::Tensor> &output_data)>();
  return ACL_ERROR_NONE;
}
#endif

size_t aclDataTypeSize(aclDataType dataType) {
  switch (dataType) {
    case ACL_STRING:
    case ACL_DT_UNDEFINED:
      return 0U;
    case ACL_FLOAT:
    case ACL_INT32:
    case ACL_UINT32:
      return sizeof(int32_t);
    case ACL_INT64:
    case ACL_UINT64:
    case ACL_DOUBLE:
    default:
      return sizeof(int64_t);
  }
}

aclError aclrtSynchronizeStream(aclrtStream stream) {
  ADP_LOG(INFO) << "aclrtSynchronizeStream stub enter, stream = " << stream;
  AclStreamStub *stub = static_cast<AclStreamStub*>(stream);
  if (stub->hook != nullptr) {
      ADP_LOG(INFO) << "aclrtSynchronizeStream:: stream = " << stream << ", process hook = "
            << stub->hook.target<aclError(*)(const aclmdlDataset *input_data, aclmdlDataset *output_data)>();
      return stub->hook(stub->input_data, stub->output_data);
  }
  ADP_LOG(INFO) << "aclrtSynchronizeStream stub out, stream = " << stream;
  return ACL_ERROR_NONE;
}

aclError aclrtSynchronizeEvent(aclrtEvent event) {
  ADP_LOG(INFO) << "aclrtSynchronizeEvent stub enter, event = " << event;
  AclEventStub *stub = static_cast<AclEventStub*>(event);
  if (stub->hook != nullptr) {
    ADP_LOG(INFO) << "aclrtSynchronizeEvent:: event = " << event << ", process hook = "
        << stub->hook.target<aclError(*)(const aclmdlDataset *input_data, aclmdlDataset *output_data)>();
    (void)stub->hook(stub->input_data, stub->output_data);
  }
  ADP_LOG(INFO) << "aclrtSynchronizeEvent stub out, event = " << event;
  return ACL_ERROR_NONE;
}

aclError aclrtQueryEvent(aclrtEvent event, aclrtEventStatus *status) {
  ADP_LOG(INFO) << "aclrtQueryEvent stub enter, event = " << event;
  AclEventStub *stub = static_cast<AclEventStub*>(event);
  *status = stub->status;
  if (stub->status == ACL_EVENT_STATUS_COMPLETE) {
    ADP_LOG(INFO) << "aclrtQueryEvent:: event = " << event << ", process hook = "
        << stub->hook.target<aclError(*)(const aclmdlDataset *input_data, aclmdlDataset *output_data)>();
    (void)stub->hook(stub->input_data, stub->output_data);
  }
  ADP_LOG(INFO) << "aclrtQueryEvent stub out, event = " << event;
  return ACL_ERROR_NONE;
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
  AclStreamStub *stubStream = static_cast<AclStreamStub*>(stream);
  AclEventStub *stubEvent = static_cast<AclEventStub*>(event);
  ADP_LOG(INFO) << "aclrtRecordEvent stub enter, stream = " << stream
      << ", event = " << event << ", process hook = "
      << stubStream->hook.target<aclError(*)(const aclmdlDataset *input_data, aclmdlDataset *output_data)>();
  stubEvent->input_data = stubStream->input_data;
  stubEvent->output_data = stubStream->output_data;
  stubEvent->hook = stubStream->hook;
  stubEvent->status = stubStream->status;
  ADP_LOG(INFO) << "aclrtRecordEvent stub out, event = " << event << ", process hook = "
        << stubEvent->hook.target<aclError(*)(const aclmdlDataset *input_data, aclmdlDataset *output_data)>();
  return ACL_ERROR_NONE;
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
  *devPtr = malloc(size);
  return ACL_ERROR_NONE;
}

aclError aclrtFree(void *devPtr) {
  free(devPtr);
  return ACL_ERROR_NONE;
}

aclError aclmdlLoadFromMem(const void *model, size_t modelSize, uint32_t *modelId) {
  return ACL_SUCCESS;
}

aclmdlDataset *aclmdlCreateDataset() {
  return new(std::nothrow) aclmdlDataset();
}

aclError aclmdlDestroyDataset(const aclmdlDataset *dataset) {
  delete dataset;
  return ACL_SUCCESS;
}

size_t aclGetTensorDescElementCount(const aclTensorDesc *desc) {
  return 1;
}

size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataset) {
  return dataset->blobs.size();
}

aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index) {
  return dataset->blobs[index].dataBuf;
}

aclDataBuffer *aclCreateDataBuffer(void *data, size_t size) {
  return new(std::nothrow) aclDataBuffer(data, size);
}

aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer) {
  const acl::AclModelTensor tensor = acl::AclModelTensor(dataBuffer, nullptr);
  dataset->blobs.push_back(tensor);
  return ACL_SUCCESS;
}

aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) {
  delete dataBuffer;
  return ACL_ERROR_NONE;
}

void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer) {
  return dataBuffer->data;
}

aclTensorDesc *aclCreateTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format) {
  return new(std::nothrow) aclTensorDesc;
}

size_t aclmdlGetNumOutputs(aclmdlDesc *modelDesc) {
  if (modelDesc == nullptr) {
      return 0U;
  }
  return modelDesc->outputDesc.size();
}

void aclDestroyTensorDesc(const aclTensorDesc *desc) {
  delete desc;
}

aclmdlDesc *aclmdlCreateDesc() {
  return new(std::nothrow) aclmdlDesc();
}

aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc) {
  delete modelDesc;
  modelDesc = nullptr;
  return ACL_SUCCESS;
}

ACLMdlGetDescStub g_mdlGetDescStub = nullptr;
void RegACLMdlGetDescStub(ACLMdlGetDescStub stub) {
  g_mdlGetDescStub = stub;
}
aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId) {
  if (g_mdlGetDescStub != nullptr) {
    return g_mdlGetDescStub(modelDesc);
  } else {
    modelDesc->inputDesc.emplace_back(aclmdlTensorDesc());
    modelDesc->inputDesc.emplace_back(aclmdlTensorDesc());
    modelDesc->outputDesc.emplace_back(aclmdlTensorDesc());
  }
  return ACL_SUCCESS;
}

aclTensorDesc *aclmdlGetDatasetTensorDesc(const aclmdlDataset *dataset, size_t index) {
  return dataset->blobs[index].tensorDesc;
}

size_t aclGetTensorDescSize(const aclTensorDesc *desc) {
  size_t size = 0U;
  const size_t descCount = aclGetTensorDescElementCount(desc);
  const size_t typeSize = aclDataTypeSize(desc->dataType);
  size = descCount * typeSize;
  return size;
}

size_t aclGetTensorDescNumDims(const aclTensorDesc *desc) {
  return desc->dims.size();
}

int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index) {
  size_t elementCount = 1U;
  return elementCount;
}

aclError aclmdlSetDatasetTensorDesc(aclmdlDataset *dataset, aclTensorDesc *tensorDesc, size_t index) {
  return ACL_SUCCESS;
}

AclRunGraphStub g_RunGraphStub = nullptr;
void RegAclRunGraphStub(AclRunGraphStub stub) {
  g_RunGraphStub = stub;
}

aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *inputs, aclmdlDataset *outputs) {
  if (g_RunGraphStub != nullptr) {
    return g_RunGraphStub(modelId, inputs, outputs);
  }
  return ACL_SUCCESS;
}

AclRunGraphWithStreamAsyncStub g_RunGraphWithStreamAsyncStub = nullptr;
void RegAclRunGraphWithStreamAsyncStub(AclRunGraphWithStreamAsyncStub stub) {
  g_RunGraphWithStreamAsyncStub = stub;
}

aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *inputs, aclmdlDataset *outputs, aclrtStream stream) {
  ADP_LOG(INFO) << "RunGraphWithStreamAsync enter, stream = " << stream;
  AclStreamStub *stub = static_cast<AclStreamStub*>(stream);
  stub->input_data = const_cast<aclmdlDataset*>(inputs);
  stub->output_data = outputs;
  stub->hook = nullptr;
  if (g_RunGraphWithStreamAsyncStub != nullptr) {
    (void)g_RunGraphWithStreamAsyncStub(modelId, inputs, outputs, stream);
    ADP_LOG(INFO) << "AclRunGraphWithStreamAsync proc hook, stream = " << stub << "hook = "
        << stub->hook.target<aclError(*)(const aclmdlDataset *input_data, aclmdlDataset *output_data)>();
  }
  return ACL_SUCCESS;
}

aclError aclmdlUnload(uint32_t modelId) {
  return ACL_SUCCESS;
}

size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr) {
        return 0U;
    }
    return static_cast<size_t>(dataBuffer->length);
}

aclDataType aclmdlGetOutputDataType(const aclmdlDesc *modelDesc, size_t index) {
  return ACL_FLOAT;
}

aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims) {
  dims->dimCount = 1;
  dims->dims[0] = 2;
  return ACL_SUCCESS;
}

aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize) {
  return ACL_SUCCESS;
}

size_t aclmdlGetNumInputs(aclmdlDesc *modelDesc) {
  if (modelDesc == nullptr) {
    return 0U;
  }
  return modelDesc->inputDesc.size();
}

aclError aclSetTensorShape(aclTensorDesc *desc, int numDims, const int64_t *dims) {
  return ACL_SUCCESS;
}

size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index) {
  return 8;
}

size_t aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index) {
  return 8;
}

aclError aclInit(const char *configPath) {
  return ACL_SUCCESS;
}

aclError aclmdlLoadFromFile(const char *modelPath, uint32_t *modelId) {
  return ACL_SUCCESS;
}

aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims) {
  dims->dimCount = 2;
  dims->dims[0] = 2;
  dims->dims[1] = -1;
  return ACL_SUCCESS;
}
