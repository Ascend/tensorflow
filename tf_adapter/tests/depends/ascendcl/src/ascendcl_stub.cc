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
#include "acl/acl_rt.h"
#include "securec.h"
#include <map>
#include <mutex>

namespace {
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