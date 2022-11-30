/* Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstring>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_op_compiler.h"
#include "acl/acl_rt.h"
#include "acl/acl_tdt.h"

namespace {
const uint32_t kDeviceSatModeLimit = 2U;
std::uint32_t deviceSatMode = 2U;
}

struct aclopAttr {};
struct aclDataBuffer {};
struct aclTensorDesc {};

struct acltdtDataItem {
  acltdtDataItem(acltdtTensorType tdtType, const int64_t *dims, size_t dimNum, aclDataType dataType, void *data,
                 size_t size) {
    this->tensor_type = tdtType;
    this->data_type = dataType;
    this->data.reset(data, [](const void *) {});
    this->size = size;
    for (size_t i = 0U; i < dimNum; i++) {
      this->dims.push_back(dims[i]);
    }
  }
  acltdtTensorType tensor_type;
  aclDataType data_type;
  std::vector<int64_t> dims;
  std::shared_ptr<void> data;
  size_t size;
};

struct acltdtDataset {
  std::vector<acltdtDataItem *> blobs;
};

struct acltdtChannelHandle {
  explicit acltdtChannelHandle(const char *name) : name_(name) {}

 private:
  std::string name_;
};

#ifdef __cplusplus
extern "C" {
#endif

aclError aclopCompileAndExecute(const char *opType, int numInputs, const aclTensorDesc *const inputDesc[],
                                const aclDataBuffer *const inputs[], int numOutputs,
                                const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
                                const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
                                const char *opPath, aclrtStream stream) {
  return ACL_ERROR_NONE;
}

aclopAttr *aclopCreateAttr() { return new aclopAttr; }
void aclopDestroyAttr(const aclopAttr *attr) { delete attr; }
aclDataBuffer *aclCreateDataBuffer(void *data, size_t size) { return new aclDataBuffer; }
aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) {
  delete dataBuffer;
  return ACL_ERROR_NONE;
}
aclTensorDesc *aclCreateTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format) {
  return new aclTensorDesc;
}
void aclDestroyTensorDesc(const aclTensorDesc *desc) { delete desc; }
aclError aclopSetAttrString(aclopAttr *attr, const char *attrName, const char *attrValue) { return ACL_ERROR_NONE; }
aclError aclopSetAttrInt(aclopAttr *attr, const char *attrName, int64_t attrValue) { return ACL_ERROR_NONE; }

aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId) {
  *context = malloc(1U);
  return ACL_ERROR_NONE;
}
aclError aclrtDestroyContext(aclrtContext context) {
  free(context);
  return ACL_ERROR_NONE;
}
aclError aclrtSetCurrentContext(aclrtContext context) { return ACL_ERROR_NONE; }

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
  *devPtr = malloc(size);
  return ACL_ERROR_NONE;
}
aclError aclrtFree(void *devPtr) {
  free(devPtr);
  return ACL_ERROR_NONE;
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  (void)std::memcpy(dst, src, count);
  return ACL_ERROR_NONE;
}

aclError aclrtCreateStream(aclrtStream *stream) {
  *stream = malloc(1U);
  return ACL_ERROR_NONE;
}

aclError aclrtDestroyStream(aclrtStream stream) {
  free(stream);
  return ACL_ERROR_NONE;
}

aclError aclrtSynchronizeStream(aclrtStream stream) { return ACL_ERROR_NONE; }

acltdtTensorType acltdtGetTensorTypeFromItem(const acltdtDataItem *dataItem) { return dataItem->tensor_type; }

aclDataType acltdtGetDataTypeFromItem(const acltdtDataItem *dataItem) { return dataItem->data_type; }

void *acltdtGetDataAddrFromItem(const acltdtDataItem *dataItem) { return dataItem->data.get(); }

size_t acltdtGetDataSizeFromItem(const acltdtDataItem *dataItem) { return dataItem->size; }

size_t acltdtGetDimNumFromItem(const acltdtDataItem *dataItem) { return dataItem->dims.size(); }

aclError acltdtGetDimsFromItem(const acltdtDataItem *dataItem, int64_t *dims, size_t dimNum) {
  if (dimNum < dataItem->dims.size()) {
    return ACL_ERROR_INVALID_PARAM;
  }
  for (size_t i = 0U; i < dataItem->dims.size(); i++) {
    dims[i] = dataItem->dims[i];
  }
  return ACL_ERROR_NONE;
}

acltdtDataItem *acltdtCreateDataItem(acltdtTensorType tdtType, const int64_t *dims, size_t dimNum, aclDataType dataType,
                                     void *data, size_t size) {
  return new acltdtDataItem(tdtType, dims, dimNum, dataType, data, size);
}

aclError acltdtDestroyDataItem(acltdtDataItem *dataItem) {
  delete dataItem;
  return ACL_ERROR_NONE;
}

acltdtDataset *acltdtCreateDataset() { return new acltdtDataset; }

aclError acltdtDestroyDataset(acltdtDataset *dataset) {
  delete dataset;
  return ACL_ERROR_NONE;
}

acltdtDataItem *acltdtGetDataItem(const acltdtDataset *dataset, size_t index) {
  if (index >= dataset->blobs.size()) {
    return nullptr;
  }
  return dataset->blobs[index];
}

aclError acltdtAddDataItem(acltdtDataset *dataset, acltdtDataItem *dataItem) {
  dataset->blobs.push_back(dataItem);
  return ACL_ERROR_NONE;
}

size_t acltdtGetDatasetSize(const acltdtDataset *dataset) { return dataset->blobs.size(); }

aclError acltdtStopChannel(acltdtChannelHandle *handle) { return ACL_ERROR_NONE; }

acltdtChannelHandle *acltdtCreateChannel(uint32_t deviceId, const char *name) { return new acltdtChannelHandle(name); }

acltdtChannelHandle *acltdtCreateChannelWithCapacity(uint32_t deviceId, const char *name, size_t capacity) {
  return new acltdtChannelHandle(name);
}

aclError acltdtDestroyChannel(acltdtChannelHandle *handle) {
  delete handle;
  return ACL_ERROR_NONE;
}

aclError acltdtSendTensor(const acltdtChannelHandle *handle, const acltdtDataset *dataset, int32_t timeout) {
  return ACL_ERROR_NONE;
}

aclError acltdtReceiveTensor(const acltdtChannelHandle *handle, acltdtDataset *dataset, int32_t timeout) {
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(100ms);
  std::vector<int64_t> dims;
  dims.resize(4, 1);
  float value = 0.0;
  acltdtAddDataItem(
    dataset, acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, dims.data(), dims.size(), ACL_FLOAT, &value, sizeof(float)));
  return ACL_ERROR_NONE;
}

aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode) {
  if (mode != ACL_RT_OVERFLOW_MODE_SATURATION && mode != ACL_RT_OVERFLOW_MODE_INFNAN) {
    deviceSatMode = 2U;
    return ACL_ERROR_INVALID_PARAM;
  }
  deviceSatMode = mode;
  return ACL_ERROR_NONE;
}

aclError aclrtGetDeviceSatMode(aclrtFloatOverflowMode *mode) {
  if (deviceSatMode >= kDeviceSatModeLimit) {
    return ACL_ERROR_FAILURE;
  }
  *mode = aclrtFloatOverflowMode(deviceSatMode);
  return ACL_ERROR_NONE;
}
#ifdef __cplusplus
}
#endif
