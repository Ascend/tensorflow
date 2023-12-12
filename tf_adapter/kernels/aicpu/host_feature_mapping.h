/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TENSORFLOW_TF_ADAPTER_KERNELS_HOST_FEATURE_MAPPING_OP_H
#define TENSORFLOW_TF_ADAPTER_KERNELS_HOST_FEATURE_MAPPING_OP_H

#include <future>
#include <thread>
#include <vector>

#include "tf_adapter/common/adapter_logger.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace featuremapping {
using HashmapType = std::unordered_map<int32_t, std::pair<int32_t, int32_t>>;
struct FeatureMappingTable {
  explicit FeatureMappingTable(int32_t input_buckets_num, int32_t input_threshold)
      : buckets_num(input_buckets_num), threshold(input_threshold), offsets(input_buckets_num),
        feature_mappings_ptr(input_buckets_num) {
    for (int i = 0; i < this->buckets_num; ++i) {
      this->offsets[i] = 0;
      this->feature_mappings_ptr[i] = new (std::nothrow) HashmapType(init_hashmap_size / buckets_num);
      if (this->feature_mappings_ptr[i] == nullptr) {
        // ADP_LOG(ERROR) << "new Hash map maping failed";
      }
    }
  }
  static const uint32_t init_hashmap_size = 60 * 10000;
  int32_t buckets_num;
  int32_t threshold;
  std::vector<int> offsets;
  std::vector<HashmapType *> feature_mappings_ptr;  // buckets_num分桶
};
extern std::unordered_map<std::string, FeatureMappingTable *> feature_mapping_table;
}  // namespace featuremapping
}  // namespace tensorflow

#endif // TENSORFLOW_TF_ADAPTER_KERNELS_HOST_FEATURE_MAPPING_OP_H