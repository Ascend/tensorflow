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
#include <fstream>
#include <iostream>
#include "dirent.h"

#include "host_feature_mapping.h"

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace featuremapping {
const uint32_t kSpaceAndSymbolLength = 2;
const uint32_t kIncludeCountsLength = 8;

class FeatureMappingImportOp : public OpKernel {
 public:
  explicit FeatureMappingImportOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    ADP_LOG(DEBUG) << "Host FeatureMappingImport built";
  }
  ~FeatureMappingImportOp() override {
    ADP_LOG(DEBUG) << "Host FeatureMappingImport has been destructed";
  }

  void ResotreLineToMapping(std::string &line, std::string &table_name) const {
    /* format :: feature_id: 3 | counts: 1 | offset_id: 7 */
    ADP_LOG(DEBUG) << "table name: " << table_name << " line " << line;
    size_t fid_pos = line.find(":") + kSpaceAndSymbolLength;
    size_t bar_pos = line.find("|");
    std::string feature_id_str = line.substr(fid_pos, bar_pos - fid_pos - 1);
    ADP_LOG(DEBUG) << "feature id str: " << feature_id_str;
    int64_t feature_id = 0;
    try {
      feature_id = stoll(feature_id_str);
    } catch(std::exception &e) {
      ADP_LOG(ERROR) << "stoll failed feature id str: " << feature_id_str << " reason: " << e.what();
      return;
    }

    size_t counts_index = line.find("counts") + kIncludeCountsLength;
    size_t last_sep_pos = line.find_last_of("|");
    std::string counts_str = line.substr(counts_index, last_sep_pos - 1 - counts_index);
    ADP_LOG(DEBUG) << "counts str: " << counts_str;
    int64_t counts = 0;
    try {
      counts = stoll(counts_str);
    } catch(std::exception &e) {
      ADP_LOG(ERROR) << "stoll failed counts str: " << counts_str << " reason: " << e.what();
      return;
    }

    size_t off_pos = line.find_last_of(":") + kSpaceAndSymbolLength;
    std::string offset_id_str = line.substr(off_pos, line.length());
    ADP_LOG(DEBUG) << "offset id str: " << offset_id_str;
    int64_t offset_id = 0;
    try {
      offset_id = stoll(offset_id_str);
    } catch(std::exception &e) {
      ADP_LOG(ERROR) << "stoll failed offset id str: " << offset_id_str << " reason: " << e.what();
      return;
    }
    ADP_LOG(DEBUG) << "feature_id: " << feature_id << " counts: " << counts << " offset_id: " << offset_id;

    // import data to hash map
    FeatureMappingTable *table = nullptr;
    auto it = feature_mapping_table.find(table_name);
    if (it != feature_mapping_table.end()) {
      ADP_LOG(DEBUG) << "have the map, insert directly";
      table = it->second;
    } else {
      uint32_t buckets_num = 1;
      uint32_t threshold = 1;
      table = new (std::nothrow) FeatureMappingTable(buckets_num, threshold);
    }

    if (table != nullptr) {
      feature_mapping_table[table_name] = table;
      // current use only one bucket refer to host feature mapping op
      int32_t bucket_index = 0;
      auto it_key = table->feature_mappings_ptr[bucket_index]->find(feature_id);
      if (it_key == table->feature_mappings_ptr[bucket_index]->end()) {
        std::pair<int64_t, int64_t> count_and_offset = std::make_pair(counts, offset_id);
        table->feature_mappings_ptr[bucket_index]->insert(std::make_pair(feature_id, count_and_offset));
        ADP_LOG(DEBUG) << "one item insert feature_id: " << feature_id << " counts: " << counts << " offset_id " << offset_id;
      } else {
        ADP_LOG(ERROR) << "do not here anymore";
      }
      ADP_LOG(DEBUG) << "map size: " << table->feature_mappings_ptr[bucket_index]->size();
    } else {
      ADP_LOG(ERROR) << "table new nothrow failed";
    }
    return;
  }

  void FindTableDoImport(std::string &dst_path_way, std::string &file_name) const {
    std::string src_file_name = dst_path_way + file_name;
    try {
      std::ifstream in_stream(src_file_name);
      if (!in_stream.is_open()) {
        ADP_LOG(ERROR) << "src_file_name: " << src_file_name << " can not open";
        return;
      }

      // read line by line
      std::string line = "";
      while (std::getline(in_stream, line)) {
        std::string table_name = "";
        size_t pos_period = file_name.find_last_of(".");
        if (pos_period != std::string::npos) {
          table_name = file_name.substr(0, pos_period);
        } else {
          ADP_LOG(ERROR) << "parse file " << file_name << " error";
          return;
        }
        ResotreLineToMapping(line, table_name);
      }
      in_stream.close();
    } catch (std::exception &e) {
      ADP_LOG(ERROR) << "write to file " << file_name << " failed, err: " << e.what();
      return;
    }
  }

  void TraverseAndParse(const std::string &src_path) {
    std::ifstream is_path(src_path);
    if (!is_path) {
      ADP_LOG(ERROR) << "import file path " << src_path << " is not exits";
      return;
    }

    const size_t path_length = src_path.size();
    std::string dst_path_way = src_path;
    if (dst_path_way[path_length - 1] != '/') {
      (void)dst_path_way.append("/");
    }

    DIR *dir;
    struct dirent *ent;
    dir = opendir(src_path.c_str());
    if (dir != nullptr) {
      while ((ent = readdir(dir)) != nullptr) {
        std::string file_name = ent->d_name;
        if (file_name == ".." || file_name == ".") {
          continue;
        }
        ADP_LOG(DEBUG) << "file_name: " << ent->d_name;
        FindTableDoImport(dst_path_way, file_name);
      }
      closedir(dir);
    } else {
       ADP_LOG(ERROR) << "open directory failed " << src_path;
    }
  }

  void Compute(OpKernelContext *ctx) override {
    ADP_LOG(INFO) << "Host FeatureMappingImport compute begin";
    const Tensor &restore_path_tensor = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(restore_path_tensor.shape()),
                errors::InvalidArgument("path expects a scalar."));
    OP_REQUIRES(ctx, (restore_path_tensor.dtype() == DT_STRING),
                errors::InvalidArgument("path should be string but got ",
                DataTypeString(restore_path_tensor.dtype())));
    const StringPiece restore_path = restore_path_tensor.scalar<tstring>()();
    OP_REQUIRES(ctx, !restore_path.empty(),
                errors::InvalidArgument("path should be a valid string."));
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, restore_path_tensor.shape(), &output_tensor));
    TraverseAndParse(std::string(restore_path));
    ADP_LOG(INFO) << "Host FeatureMappingImport compute end";
  }
};

REGISTER_KERNEL_BUILDER(Name("FeatureMappingImport").Device(DEVICE_CPU), FeatureMappingImportOp);
}  // namespace featuremapping
}  // namespace tensorflow
