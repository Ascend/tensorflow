/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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

#include "tf_adapter/util/util.h"

#include <numeric>
#include <vector>
#include <string>
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/common/compat_tf1_tf2.h"
#include "inc/metadef/inc/graph/def_types.h"
#include "securec.h"
#include "framework/common/string_util.h"
namespace tensorflow {
namespace {
const std::string ATTR_VALUE_SCOPE_NAME = "_without_npu_compile";
const size_t kMaxDynamicDimNum = 100;

std::vector<std::string> SplitInputShape(const std::string &input_shape) {
  std::vector<std::string> shape_pair_vec;
  size_t pos = input_shape.rfind(":");
  if (pos != std::string::npos) {
    shape_pair_vec.emplace_back(input_shape.substr(0, pos));
    shape_pair_vec.emplace_back(input_shape.substr(pos + 1, input_shape.size() - pos));
  }
  return shape_pair_vec;
}
}

Status GetDtStringTensorData(const Tensor &tensor, uint8_t *&data_ptr, uint64_t &data_size,
                             std::vector<int64_t> &dims, std::vector<std::unique_ptr<uint8_t[]>> &buff_list) {
  for (int i = 0; i < tensor.dims(); ++i) { dims.emplace_back(tensor.dim_size(i)); }
  int64_t total_nums = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
  uint64_t total_size = 0UL;
  for (int64_t i = 0; i < total_nums; ++i) { total_size += tensor.flat<tstring>()(i).size(); }
  uint64_t buff_size = sizeof(ge::StringHead) * static_cast<uint64_t>(total_nums) + total_size;
  std::unique_ptr<uint8_t[]> buffer(new (std::nothrow) uint8_t[buff_size]);
  REQUIRES_NOT_NULL(buffer);
  buff_list.push_back(std::move(buffer));

  uint8_t *base_ptr = buff_list.back().get();
  uint64_t offset = sizeof(ge::StringHead) * static_cast<uint64_t>(total_nums);
  for (int64_t i = 0L; i < total_nums; ++i) {
    ge::StringHead *head = ge::PtrToPtr<uint8_t, ge::StringHead>(base_ptr + i * sizeof(ge::StringHead));
    head->addr = offset;
    head->len = tensor.flat<tstring>()(i).size();
    // can not use memcpy_s here, data size may over 2G
    // total_size is calculate by item info, could not overflow here
    (void)memcpy(base_ptr + offset, tensor.flat<tstring>()(i).data(), head->len);
    offset += head->len;
  }
  data_ptr = buff_list.back().get();
  data_size = buff_size;
  return Status::OK();
}

Status MappingDTStringTensor2DataItem(const Tensor &tensor, tdt::DataItem &item,
                                      std::vector<std::unique_ptr<uint8_t[]>> &buff_list) {
  if (tensor.dims() == 0) {
    std::string value = tensor.scalar<npu::compat_tf1_tf2::string>()();
    item.dataLen_ = tensor.scalar<npu::compat_tf1_tf2::string>()().size();
    item.dataPtr_ = std::shared_ptr<void>(const_cast<char *>(value.data()), [](const void *elem) {
      (void)elem;
    });
    return Status::OK();
  }

  uint8_t *data_ptr = nullptr;
  uint64_t data_size = 0UL;
  std::vector<int64_t> dims;
  TF_RETURN_IF_ERROR(GetDtStringTensorData(tensor, data_ptr, data_size, dims, buff_list));
  item.dataPtr_ = std::shared_ptr<void>(data_ptr, [](const void *ptr) {
    (void)ptr;
  });
  item.dataLen_ = data_size;
  return Status::OK();
}

Status MappingDtStringTensor2AclDataItem(const Tensor &tensor, acltdtDataItem *&acl_data,
                                         std::vector<std::unique_ptr<uint8_t[]>> &buff_list) {
  if (tensor.dims() == 0) {
    auto value = ge::PtrToPtr<char, tensorflow::tstring>(const_cast<char *>(tensor.tensor_data().data()));
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

bool IsWithoutNpuScope(const NodeDef &node_def) {
  if (node_def.attr().count(ATTR_VALUE_SCOPE_NAME) > 0) { return node_def.attr().at(ATTR_VALUE_SCOPE_NAME).b(); }
  return false;
}

bool IsWithoutNpuScope(const Node *node) {
  return IsWithoutNpuScope(node->def());
}

Status BuildSubgraphMuliDimsInput(const std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map,
                                  const DimsVector &dynamic_dims_vec,
                                  std::vector<std::string> &subgraph_multi_dims_input_shape,
                                  std::vector<std::string> &subgraph_multi_dims_input_dims) {
  size_t nodes_num = user_shape_map.size();
  size_t count = 0U;
  size_t dynamic_count = dynamic_dims_vec.size();
  for (size_t i = 0U; i < nodes_num; ++i) {
    std::vector<std::string> tmp(dynamic_count);
    auto &nodes_shape = user_shape_map[i].second;
    for (auto &dim : nodes_shape) {
      if (dim != -1) {
        continue;
      }
      for (size_t j = 0U; j < dynamic_count; ++j) {
        tmp[j].append(dynamic_dims_vec[j][count]).append(",");
      }
      ++count;
    }
    std::string tmp_dims;
    for (size_t j = 0U; j < dynamic_count; ++j) {
      if (tmp[j].empty()) {
        return errors::Internal("build subgraph multi dims input dims failed");
      }
      tmp_dims.append(tmp[j].substr(0, tmp[j].size() - 1)).append(";");
    }
    std::string tmp_shape;
    for (size_t j = 0U; j < nodes_shape.size(); ++j) {
      tmp_shape.append(std::to_string(nodes_shape[j])).append(",");
    }
    subgraph_multi_dims_input_dims.push_back(tmp_dims.substr(0, tmp_dims.size() - 1));
    subgraph_multi_dims_input_shape.push_back(tmp_shape.substr(0, tmp_shape.size() - 1));
    ADP_LOG(INFO) << "index: " << i << " subgraph_multi_dims_input_dims is: " << subgraph_multi_dims_input_dims[i];
    ADP_LOG(INFO) << "index: " << i << " subgraph_multi_dims_input_shape is: " << subgraph_multi_dims_input_shape[i];
  }
  return Status::OK();
}

Status ParseDynamicShapesAndDims(const std::string &input_shapes, const std::string &dynamic_dims,
                                 std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map,
                                 DimsVector &dynamic_dims_vec,
                                 std::vector<std::pair<std::string, std::vector<int64_t>>> &max_shape_range_map) {
  ADP_LOG(INFO) << "input_shapes: " << input_shapes << " dynamic_dims: " << dynamic_dims;
  TF_RETURN_IF_ERROR(ParseDynamicShapes(input_shapes, user_shape_map));
  std::vector<std::vector<int64_t>> dynamic_dims_digit_vec;
  TF_RETURN_IF_ERROR(ParseDynamicDims(dynamic_dims, dynamic_dims_vec, dynamic_dims_digit_vec, user_shape_map));
  TF_RETURN_IF_ERROR(ParseMaxShapeRange(user_shape_map, dynamic_dims_digit_vec, max_shape_range_map));
  return Status::OK();
}

Status ParseMaxShapeRange(const std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map,
                          const std::vector<std::vector<int64_t>> &dynamic_dims_digit_vec,
                          std::vector<std::pair<std::string, std::vector<int64_t>>> &max_shape_range_map) {
  size_t num = dynamic_dims_digit_vec[0].size();
  std::vector<int64_t> tmp(num, 0);
  for (auto &digit_vec : dynamic_dims_digit_vec) {
    for (size_t i = 0U; i < num; ++i) {
      tmp[i] = std::max(tmp[i], digit_vec[i]);
    }
  }

  size_t count = 0U;
  max_shape_range_map = user_shape_map;
  for (auto &shape_range : max_shape_range_map) {
    std::vector<int64_t> &shapes = shape_range.second;
    for (size_t i = 0U; i < shapes.size(); ++i) {
      if (shapes[i] == -1)
        shapes[i] = tmp[count++];
    }
  }
  return Status::OK();
}

Status ParseDynamicDims(const std::string &dynamic_dims, DimsVector &dynamic_dims_vec,
                        std::vector<std::vector<int64_t>> &dynamic_dims_digit_vec,
                        const std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map) {
  int32_t dynamic_dim_num = 0;
  for (auto &info_shapes : user_shape_map) {
    auto &shapes = info_shapes.second;
    dynamic_dim_num += std::count(shapes.begin(), shapes.end(), -1);
  }
  ADP_LOG(INFO) << "dynamic dim num: " << dynamic_dim_num;
  if (dynamic_dims.empty()) {
    return errors::Internal("dynamic_dims can not be empty.");
  }
  // Different parameter sets are split by ';'
  std::vector<std::string> split_set = ge::StringUtils::Split(dynamic_dims, ';');
  if (split_set.size() > kMaxDynamicDimNum) {
    return errors::Internal("dynamic_dims's num of parameter set can not exceed:", kMaxDynamicDimNum);
  }
  for (auto split_dim : split_set) {
    std::vector<std::string> one_dim_set = ge::StringUtils::Split(split_dim, ',');
    if (one_dim_set.size() != static_cast<size_t>(dynamic_dim_num)) {
      return errors::Internal(
          "dynamic_dims: ", dynamic_dims.c_str(),
          " invalid. reason: Each gear setting needs to be consistent with the number of -1 in the inputshape.");
    }
    std::vector<int64_t> digit_vec;
    for (auto dim : one_dim_set) {
      for (auto c : dim) {
        if (!isdigit(c)) {
          return errors::Internal("dynamic_dims: ", dynamic_dims.c_str(), " parameter must be positive integer.");
        }
        constexpr int32_t decimal = 10;
        digit_vec.push_back(std::strtol(dim.c_str(), nullptr, decimal));
      }
    }
    dynamic_dims_vec.push_back(one_dim_set);
    dynamic_dims_digit_vec.push_back(digit_vec);
  }
  return Status::OK();
}

Status ParseDynamicShapes(const std::string &input_shapes,
                          std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map) {
  std::vector<std::string> shape_vec = ge::StringUtils::Split(input_shapes, ';');
  const int32_t kDefaultShapePairSize = 2;
  for (const auto &shape : shape_vec) {
    std::vector<std::string> shape_pair_vec = SplitInputShape(shape);
    if (shape_pair_vec.size() != kDefaultShapePairSize) {
      return errors::Internal("parse input_shape failed.");
    }

    if (shape_pair_vec[1].empty()) {
      return errors::Internal("parse input_shape failed.");
    }

    std::vector<std::string> shape_value_strs = ge::StringUtils::Split(shape_pair_vec[1], ',');
    std::vector<int64_t> shape_values;
    for (auto &shape_value_str : shape_value_strs) {
      // stoul: The method may throw an exception: invalid_argument/out_of_range
      if (shape_value_str.find('.') != std::string::npos) {
        return errors::Internal("unsupport float config value.");
      }
      int64 result = 0;
      if (!strings::safe_strto64(shape_value_str, &result)) {
        return errors::InvalidArgument("value ", shape_value_str.c_str(),
                                       "is invalid, should be int, such as 0, 1, 2.");
      }
      shape_values.push_back(static_cast<int64_t>(result));
    }

    user_shape_map.push_back(make_pair(ge::StringUtils::Trim(shape_pair_vec[0]), shape_values));
  }
  return Status::OK();
}
} // namespace tensorflow