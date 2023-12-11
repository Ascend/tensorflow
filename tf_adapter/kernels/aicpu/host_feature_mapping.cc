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
#include "host_feature_mapping.h"

namespace tensorflow {
namespace featuremapping {
using HashmapType = std::unordered_map<int32_t, std::pair<int32_t, int32_t>>;
class SimpleThreadPool {
 public:
  void SyncRun(const std::vector<std::function<int()>> &tasks) {
    std::vector<std::future<int>> futs;
    for (auto &task : tasks) {
      futs.push_back(std::async(task));
    }
    for (auto &fut : futs) {
      fut.wait();
    }
  }
};

std::unordered_map<std::string, FeatureMappingTable *> feature_mapping_table;
class FeatureMappingOp : public OpKernel {
 public:
  explicit FeatureMappingOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &threshold));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name", &table_name));
    ADP_LOG(INFO) << "Host FeatureMapping built" << table_name;
  }
  ~FeatureMappingOp() override {
    ADP_LOG(INFO) << "Host FeatureMapping has been destructed";
  }

  FeatureMappingTable *get_or_init_tables(std::string table_name, int32_t buckets_num, int32_t threshold) {
    auto it = feature_mapping_table.find(table_name);
    if (it != feature_mapping_table.end()) {
      return it->second;
    } else {
      FeatureMappingTable *table = new (std::nothrow) FeatureMappingTable(buckets_num, threshold);
      if (table != nullptr) {
        feature_mapping_table[table_name] = table;
        return table;
      }
      return nullptr;
    }
  }

  int32_t get_and_increase_offset(FeatureMappingTable *table, int32_t bucket_index, int32_t buckets_num) {
    int32_t offset = table->offsets[bucket_index] * buckets_num + bucket_index;
    table->offsets[bucket_index]++;
    return offset;
  }

  void find_hash_table(FeatureMappingTable *table, int32_t bucket_index, int32_t feature_id_len,
                       const int32_t *feature_id_data, int32_t *offset_id_data) {
    const int32_t buckets_num = table->buckets_num;
    auto table_mappings = table->feature_mappings_ptr[bucket_index];
    int32_t last_index = table_mappings->size();
    ADP_LOG(INFO) << "last_index value " << last_index;
    for (int i = last_index; i < feature_id_len + last_index; ++i) {
      int32_t feature_id = feature_id_data[i - last_index];
      ADP_LOG(INFO) << "feature id " << feature_id;
      if (feature_id < 0) {
        offset_id_data[i - last_index] = -1;
        continue;
      }
      auto it = table_mappings->find(feature_id);
      if (it == table_mappings->end()) {
        int32_t offset = get_and_increase_offset(table, bucket_index, buckets_num);
        std::pair<int32_t, int32_t> count_and_offset = std::make_pair(1, offset);
        table_mappings->insert(std::make_pair(feature_id, count_and_offset));
        offset_id_data[i - last_index] = offset;
        ADP_LOG(INFO) << "new insert offset id " << offset;
      } else {
        std::pair<int32_t, int32_t> &count_and_offset = it->second;
        count_and_offset.first++;
        offset_id_data[i - last_index] = count_and_offset.second;
        ADP_LOG(INFO) << "orginal offset id " << count_and_offset.second;
      }
    }
  }

  void Compute(OpKernelContext *ctx) override {
    ADP_LOG(INFO) << "Host FeatureMapping compute begin";
    const Tensor &featureIdTensor = ctx->input(0);
    auto input = featureIdTensor.flat<int32_t>();
    auto feature_id_data = (const int32_t *)featureIdTensor.tensor_data().data();
    const int32_t feature_id_len = input.size();

    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, featureIdTensor.shape(), &output_tensor));
    auto offset_id = (int32_t *)output_tensor->tensor_data().data();

    // device FeatureMapping uses Usafe only support Single Core
    SimpleThreadPool pool;
    uint32_t thread_num = 1;
    std::vector<std::function<int()>> tasks;
    FeatureMappingTable *table = get_or_init_tables(table_name, thread_num, threshold);
    if (table == nullptr) {
        ADP_LOG(ERROR) << "get_or_init_tables failed ";
        return;
    }
    for (uint32_t i = 0; i < thread_num; ++i) {
      tasks.push_back([this, table, i, feature_id_len, feature_id_data, offset_id]() -> int {
        find_hash_table(table, i, feature_id_len, feature_id_data, offset_id);
        return int{};
      });
    }

    if (!tasks.empty()) {
      pool.SyncRun(tasks);
    }
    ADP_LOG(INFO) << "Host FeatureMapping compute end";
  }

 private:
  int threshold{};
  std::string table_name{};
};

REGISTER_KERNEL_BUILDER(Name("FeatureMapping").Device(DEVICE_CPU), FeatureMappingOp);
}  // namespace featuremapping
}  // namespace tensorflow
