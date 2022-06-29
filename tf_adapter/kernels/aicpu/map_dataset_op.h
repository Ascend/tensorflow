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

#ifndef TENSORFLOW_TF_ADAPTER_KERNELS_DATA_MAP_DATASET_OP_H_
#define TENSORFLOW_TF_ADAPTER_KERNELS_DATA_MAP_DATASET_OP_H_

#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tf_adapter/common/adp_logger.h"

namespace tensorflow {
namespace data {
class NpuMapDatasetOp : public UnaryDatasetOpKernel {
public:
  static constexpr const char* const kDatasetType = "NpuMap";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kOtherArguments = "other_arguments";
  static constexpr const char* const kNumParallelCalls = "num_parallel_calls";
  static constexpr const char* const kFunc = "f";
  static constexpr const char* const kTarguments = "Targuments";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kOutputDevice = "output_device";
  static constexpr const char* const kDeterministic = "deterministic";
  static constexpr const char* const kSloppy = "sloppy";
  static constexpr const char* const kPreserveCardinality =
      "preserve_cardinality";

  explicit NpuMapDatasetOp(OpKernelConstruction* ctx);

  ~NpuMapDatasetOp() {
      ADP_LOG(INFO) << "~NpuMapDatasetOp";
  };

protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

private:
  Status CheckOutputType();
  class Dataset;
  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  bool sloppy_;
  bool preserve_cardinality_;
  bool deterministic_;
  const std::map<std::string, std::string> sess_options_;
  std::map<std::string, std::string> init_options_;
  std::string output_device_;
  std::vector<std::pair<StringPiece, AttrValue>> attrs_;
}; // class NpuMapDatasetOp
}  // namespace data
}  // namespace tensorflow

#endif