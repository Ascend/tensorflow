/**
* @file tensor_data_transfer.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef DEPENDS_ASCENDCL_STUB_H
#define DEPENDS_ASCENDCL_STUB_H
#include <string.h>
#include <string>
#include <vector>
#include <memory>

#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "graph/tensor.h"
#include "acl/acl_base.h"
#include "acl/acl_tdt.h"
#include "acl/acl_rt.h"
#include "graph/ascend_string.h"
#include "graph/ge_error_codes.h"
#include "graph/small_vector.h"

void aclrtSetSocNameStub(std::string SocVersion);
void aclrtSetDefaultSocNameStub();

struct acltdtDataItem {
    acltdtDataItem(acltdtTensorType tdtType,
        const int64_t *dims, size_t dimNum, const std::string &dimsStr,
        aclDataType type, const std::string &typeStr,
        std::shared_ptr<void> tensorData, size_t size)
    {
        this->tdtType = tdtType;
        for (size_t i = 0; i < dimNum; ++i) {
            this->dims.push_back(dims[i]);
        }
        this->dimsStr = dimsStr;
        this->dataType = type;
        this->dataTypeStr = typeStr;
        this->dataLen = size;
        this->dataPtr = tensorData;
    }
    acltdtDataItem() = default;
    ~acltdtDataItem() = default;
    acltdtTensorType tdtType;
    std::vector<int64_t> dims;
    std::string dimsStr;
    aclDataType dataType;
    std::string dataTypeStr;
    size_t dataLen;
    std::shared_ptr<void> dataPtr;
};

struct acltdtDataset {
    acltdtDataset()  : freeSelf(false) {};
    ~acltdtDataset()
    {
        if (freeSelf) {
            for (auto iter = blobs.begin(); iter != blobs.end(); ++iter) {
                (void)acltdtDestroyDataItem(*iter);
            }
        }
    }
    std::vector<acltdtDataItem *> blobs;
    bool freeSelf;
};

struct acltdtChannelHandle {
    acltdtChannelHandle(uint32_t deviceId, const char *channelName)
    {
        devId = deviceId;
        if (channelName != nullptr) {
            name = channelName;
            size_t prefixLen = sizeof("TF_RECEIVE_") - 1;
            if (0 == strncmp(channelName, "TF_RECEIVE_", prefixLen)) {
                recvName = channelName + prefixLen;
            }
        }
    }
    acltdtChannelHandle() = default;
    ~acltdtChannelHandle() = default;
    std::string name;
    std::string recvName;
    uint32_t devId;
};

#if 0
using AclStreamStubHook = std::function<ge::Status(const std::vector<ge::Tensor> &input_data, std::vector<ge::Tensor> &output_data)>;
struct AclStreamStub{
  std::vector<ge::Tensor> input_data;
  std::vector<ge::Tensor> *output_data;
  AclStreamStubHook hook;
  aclrtEventStatus status;
};

typedef AclStreamStub AclEventStub;
#endif

namespace acl {
  struct AclModelTensor{
    AclModelTensor(aclDataBuffer *const dataBufIn,
      aclTensorDesc *const tensorDescIn) : dataBuf(dataBufIn), tensorDesc(tensorDescIn) {}

    ~AclModelTensor() = default;
    aclDataBuffer *dataBuf;
    aclTensorDesc *tensorDesc;
  };
} //namespace acl

struct aclmdlDataset {
  aclmdlDataset()
    : seq(0U),
      modelId(0U) {}

  ~aclmdlDataset() = default;
  uint32_t seq;
  uint32_t modelId;
  std::vector<acl::AclModelTensor> blobs;
};

using AclStreamStubHook = std::function<aclError(const aclmdlDataset *input_data, aclmdlDataset *output_data)>;
struct AclStreamStub{
  aclmdlDataset *input_data;
  aclmdlDataset *output_data;
  AclStreamStubHook hook;
  aclrtEventStatus status;
};

typedef AclStreamStub AclEventStub;

struct aclmdlTensorDesc {
  aclmdlTensorDesc() : name(""), size(0U), format(ACL_FORMAT_UNDEFINED), dataType(ACL_DT_UNDEFINED) {}
  ~aclmdlTensorDesc()  = default;

  std::string name;
  size_t size;
  aclFormat format;
  aclDataType dataType;
  std::vector<int64_t> dims;
  std::vector<int64_t> dimsV2;
  std::vector<std::pair<int64_t, int64_t>> shapeRanges;
};

struct aclmdlDesc {
  uint32_t modelId = 0U;
  std::vector<aclmdlTensorDesc> inputDesc;
  std::vector<aclmdlTensorDesc> outputDesc;
};

struct aclDataBuffer {
  aclDataBuffer(void* const dataIn, const uint64_t len) : data(dataIn), length(len) {}
  ~aclDataBuffer() = default;

  void *data;
  uint64_t length;
};

struct aclTensorDesc {
  aclTensorDesc() = default;
  aclTensorDesc(aclDataType type) : dataType(type) {}
  std::vector<int64_t> dims;
  aclDataType dataType;
};

using ACLMdlGetDescStub = std::function<aclError(aclmdlDesc *)>;
void RegACLMdlGetDescStub(ACLMdlGetDescStub stub);

using AclRunGraphWithStreamAsyncStub = std::function<aclError(uint32_t, const aclmdlDataset*, aclmdlDataset*, void*)>;
void RegAclRunGraphWithStreamAsyncStub(AclRunGraphWithStreamAsyncStub stub);

using AclRunGraphStub = std::function<aclError(uint32_t, const aclmdlDataset*, aclmdlDataset*)>;
void RegAclRunGraphStub(AclRunGraphStub stub);

#endif //ACL_TENSOR_DATA_TRANSFER_H

