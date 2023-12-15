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
#include <cerrno>
#include <fstream>
#include <iostream>

#include "host_feature_mapping.h"

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace featuremapping {
const std::string kBinFileSuffix = ".bin";

class FeatureMappingExportOp : public OpKernel {
 public:
  explicit FeatureMappingExportOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    ADP_LOG(DEBUG) << "FeatureMappingExport built";
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name_list", &table_name_list_));
  }
  ~FeatureMappingExportOp() override {
    ADP_LOG(DEBUG) << "FeatureMappingExport has been destructed";
  }

  void WriteMappingContens2File(std::string &table_name, std::string &dst_path) const {
    auto it = feature_mapping_table.find(table_name);
    if (it == feature_mapping_table.end()) {
      ADP_LOG(WARNING) << "this table " << table_name << " is not in mapping, just skip";
      return;
    }

    FeatureMappingTable *table = it->second;
    if (table == nullptr) {
      ADP_LOG(ERROR) << "table map find but table is nullptr";
      return;
    }

    try {
      std::ofstream out_stream(dst_path);
      // current use only one bucket refer to host feature mapping op
      int32_t bucket_index = 0;
      const auto mapping_map = table->feature_mappings_ptr[bucket_index];
      std::unordered_map<int64_t, std::pair<int64_t, int64_t>>::iterator map_iter;
      for (map_iter = mapping_map->begin(); map_iter != mapping_map->end(); ++map_iter) {
        const int64_t feature_id = map_iter->first;
        std::pair<int64_t, int64_t> &count_and_offset = map_iter->second;
        const int64_t counts = count_and_offset.first;
        const int64_t offset_id = count_and_offset.second;
        // feature_id: 3 | counts: 1 | offset_id: 7
        std::string content = "feature_id: " + std::to_string(feature_id) + " | "
                              + "counts: " + std::to_string(counts) + " | "
                              + "offset_id: " + std::to_string(offset_id);
        ADP_LOG(DEBUG) << "content: " << content;
        out_stream << content << std::endl;
      }
      out_stream.close();
    } catch (std::exception &e) {
      ADP_LOG(ERROR) << "write to file " << dst_path << " failed, err: " << e.what();
      return;
    }
  }

  void SaveFeatureMapping2File(const std::string &path) {
    const size_t path_length = path.size();
    std::string dst_path_way = path;
    if (path[path_length - 1] != '/') {
      (void)dst_path_way.append("/");
    }

    std::ifstream is_path(dst_path_way);
    if (!is_path) {
      ADP_LOG(DEBUG) << "export file path " << dst_path_way << " is not exits, make it";
      if (mkdir(dst_path_way.c_str(), S_IRWXO | S_IRWXG | S_IRWXU) != 0) {
        if (errno != EEXIST) {
          ADP_LOG(ERROR) << "Create file directory " << dst_path_way << " failed, errmsg " << strerror(errno);
          return;
        }
      }
    }

    const size_t name_size = table_name_list_.size();
    ADP_LOG(DEBUG) << "dst_path_way " << dst_path_way << " name_size " << name_size;
    if (name_size == 0) {
      ADP_LOG(DEBUG) << "default export all feature mapping";
      for (const auto &map_pair : feature_mapping_table) {
        std::string table_name = map_pair.first;
        std::string dst_path_file = dst_path_way + table_name + kBinFileSuffix;
        ADP_LOG(DEBUG) << "table_name " << table_name << " dst_path_file " << dst_path_file;
        WriteMappingContens2File(table_name, dst_path_file);
      }
    } else {
      ADP_LOG(DEBUG) << "export attr name of user specified";
      for (size_t index = 0; index < name_size; ++index) {
        std::string attr_table_name = std::string(table_name_list_[index]);
        std::string dst_file_path = dst_path_way + attr_table_name + kBinFileSuffix;
        ADP_LOG(DEBUG) << "attr_table_name " << attr_table_name << " dst_file_path " << dst_file_path;
        WriteMappingContens2File(attr_table_name, dst_file_path);
      }
    }
    return;
  }

  void Compute(OpKernelContext *ctx) override {
    ADP_LOG(INFO) << "Host FeatureMappingExport compute begin";
    const Tensor &save_path_tensor = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(save_path_tensor.shape()),
                errors::InvalidArgument("path expects a scalar."));
    OP_REQUIRES(ctx, (save_path_tensor.dtype() == DT_STRING),
                errors::InvalidArgument("path should be string but got ",
                DataTypeString(save_path_tensor.dtype())));
    const StringPiece save_path = save_path_tensor.scalar<tstring>()();
    OP_REQUIRES(ctx, !save_path.empty(),
                errors::InvalidArgument("path should be a valid string."));
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, save_path_tensor.shape(), &output_tensor));
    SaveFeatureMapping2File(std::string(save_path));
    ADP_LOG(INFO) << "Host FeatureMappingExport compute end";
  }

 private:
  std::vector<std::string> table_name_list_{};
};

REGISTER_KERNEL_BUILDER(Name("FeatureMappingExport").Device(DEVICE_CPU), FeatureMappingExportOp);
}  // namespace featuremapping
}  // namespace tensorflow
