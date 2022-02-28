#include "tf_adapter/util/util.h"

#include <numeric>
#include <vector>
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "inc/metadef/inc/graph/def_types.h"
#include "securec.h"
namespace tensorflow {
Status GetDtStringTensorData(const Tensor &tensor, uint8_t *&data_ptr, uint64_t &data_size,
                             std::vector<int64_t> &dims, std::vector<std::unique_ptr<uint8_t[]>> &buff_list) {
  for (int i = 0; i < tensor.dims(); ++i) { dims.emplace_back(tensor.dim_size(i)); }
  int64_t total_nums = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
  uint64_t total_size = 0UL;
  for (int64_t i = 0; i < total_nums; ++i) { total_size += tensor.flat<tstring>()(i).size(); }
  uint64_t buff_size = sizeof(ge::StringHead) * total_nums + total_size;
  std::unique_ptr<uint8_t[]> buffer(new (std::nothrow) uint8_t[buff_size]);
  REQUIRES_NOT_NULL(buffer);
  buff_list.push_back(std::move(buffer));

  uint8_t *base_ptr = buff_list.back().get();
  uint64_t offset = sizeof(ge::StringHead) * total_nums;
  for (int64_t i = 0; i < total_nums; ++i) {
    ge::StringHead *head = reinterpret_cast<ge::StringHead *>(base_ptr + i * sizeof(ge::StringHead));
    head->addr = offset;
    head->len = tensor.flat<tstring>()(i).size();
    // can not use memcpy_s here, data size may over 2G
    // total_size is calculate by item info, could not overflow here
    memcpy(base_ptr + offset, tensor.flat<tstring>()(i).data(), head->len);
    offset += head->len;
  }
  data_ptr = buff_list.back().get();
  data_size = buff_size;
  return Status::OK();
}

Status MappingDTStringTensor2DataItem(const Tensor &tensor, tdt::DataItem &item,
                                      std::vector<std::unique_ptr<uint8_t[]>> &buff_list) {
  if (tensor.dims() == 0) {
#ifdef TF_VERSION_TF2
    std::string value = tensor.scalar<tstring>()();
    item.dataLen_ = tensor.scalar<tstring>()().size();
#else
    std::string value = tensor.scalar<string>()();
    item.dataLen_ = tensor.scalar<string>()().size();
#endif
    item.dataPtr_ = std::shared_ptr<void>(const_cast<char *>(value.data()), [](void *elem) {});
    return Status::OK();
  }

  uint8_t *data_ptr = nullptr;
  uint64_t data_size = 0UL;
  std::vector<int64_t> dims;
  TF_RETURN_IF_ERROR(GetDtStringTensorData(tensor, data_ptr, data_size, dims, buff_list));
  item.dataPtr_ = std::shared_ptr<void>(data_ptr, [](void *ptr){});
  item.dataLen_ = data_size;
  return Status::OK();
}

Status MappingDtStringTensor2AclDataItem(const Tensor &tensor, acltdtDataItem *&acl_data,
                                         std::vector<std::unique_ptr<uint8_t[]>> &buff_list) {
  if (tensor.dims() == 0) {
    auto value = reinterpret_cast<tensorflow::tstring *>(const_cast<char *>(tensor.tensor_data().data()));
    // for scalar type, *dims is nullptr and dim_num is 0
    acl_data = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, nullptr, 0, ACL_STRING,
                                    const_cast<char *>(value->c_str()), value->size());
    return Status::OK();
  }

  uint8_t *data_ptr = nullptr;
  uint64_t data_size = 0UL;
  std::vector<int64_t> dims;
  TF_RETURN_IF_ERROR(GetDtStringTensorData(tensor, data_ptr, data_size, dims, buff_list));
  acl_data = acltdtCreateDataItem(ACL_TENSOR_DATA_TENSOR, dims.data(), dims.size(),
                                  ACL_STRING, data_ptr, data_size);
  return Status::OK();
}
} // namespace tensorflow