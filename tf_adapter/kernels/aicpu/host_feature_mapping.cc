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
class SimpleThreadPool {
 public:
  void SyncRun(const std::vector<std::function<int()>> &tasks) const {
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
class HostFeatureMappingOp : public OpKernel {
 public:
  explicit HostFeatureMappingOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &threshold_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name", &table_name_));
    ADP_LOG(DEBUG) << "Host HostFeatureMapping built table_name " << table_name_;
  }
  ~HostFeatureMappingOp() override {
    ADP_LOG(DEBUG) << table_name_ << " has been destructed";
  }

  FeatureMappingTable *get_or_init_tables(std::string table_name, int32_t buckets_num, int32_t threshold) const {
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

  int64_t get_and_increase_offset(FeatureMappingTable *table, int32_t bucket_index,
                                  int32_t buckets_num, int64_t last_index) const {
    // compatible inference training exectuion and continuation training scenarios
    if (table->offsets[bucket_index] == 0) {
        table->offsets[bucket_index] = last_index;
    }
    int64_t offset = table->offsets[bucket_index] * buckets_num + bucket_index;
    table->offsets[bucket_index]++;
    return offset;
  }

  void find_hash_table(FeatureMappingTable *table, int32_t bucket_index, int64_t feature_id_len,
                       const int64_t *feature_id_data, int64_t *offset_id_data) {
    const int32_t buckets_num = table->buckets_num;
    auto table_mappings = table->feature_mappings_ptr[bucket_index];
    int64_t last_index = table_mappings->size();
    ADP_LOG(DEBUG) << "last_index value " << last_index;
    for (int64_t i = 0; i < feature_id_len; ++i) {
      int64_t feature_id = feature_id_data[i];
      auto it = table_mappings->find(feature_id);
      if (it == table_mappings->end()) {
        int64_t offset = get_and_increase_offset(table, bucket_index, buckets_num, last_index);
        std::pair<int64_t, int64_t> count_and_offset = std::make_pair(1LL, offset);
        table_mappings->insert(std::make_pair(feature_id, count_and_offset));
        offset_id_data[i] = offset;
        ADP_LOG(DEBUG) << "table_name " << table_name_ << " feature id " << feature_id <<
                          " new insert offset id " << offset;
      } else {
        std::pair<int64_t, int64_t> &count_and_offset = it->second;
        count_and_offset.first++;
        offset_id_data[i] = count_and_offset.second;
        ADP_LOG(DEBUG) << "table_name " << table_name_ << " feature id " << feature_id <<
                          " orginal offset id " << count_and_offset.second;
      }
    }
  }

  void Compute(OpKernelContext *ctx) override {
    ADP_LOG(INFO) << "table_name " << table_name_ << " compute begin";
    const Tensor &featureIdTensor = ctx->input(0);
    auto feature_id_data = (const int64_t *)(featureIdTensor.tensor_data().data());
    const int64_t feature_id_len = featureIdTensor.NumElements();

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, featureIdTensor.shape(), &output_tensor));
    auto offset_id = (int64_t *)(output_tensor->tensor_data().data());

    // device FeatureMapping uses Usafe only support Single Core
    SimpleThreadPool pool;
    int32_t thread_num = 1;
    std::vector<std::function<int()>> tasks;
    FeatureMappingTable *table = get_or_init_tables(table_name_, thread_num, threshold_);
    if (table == nullptr) {
        ADP_LOG(ERROR) << "get or init table failed table is nullptr";
        return;
    }
    for (int32_t i = 0; i < thread_num; ++i) {
      tasks.push_back([this, table, i, feature_id_len, feature_id_data, offset_id]() -> int {
        find_hash_table(table, i, feature_id_len, feature_id_data, offset_id);
        return int{};
      });
    }

    if (!tasks.empty()) {
      pool.SyncRun(tasks);
    }
    ADP_LOG(INFO) << "table_name " << table_name_ << " compute end";
  }

 private:
  int threshold_{};
  std::string table_name_{};
};

REGISTER_KERNEL_BUILDER(Name("HostFeatureMapping").Device(DEVICE_CPU), HostFeatureMappingOp);
}  // namespace featuremapping
}  // namespace tensorflow
