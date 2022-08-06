/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#ifndef TENSORFLOW_CORE_KERNELS_DATASET_FUNCTION_H_
#define TENSORFLOW_CORE_KERNELS_DATASET_FUNCTION_H_

#include <unordered_map>
#include <atomic>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include "graph/tensor.h"
#include "graph/utils/graph_utils.h"
#include "graph/compute_graph.h"
#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "framework/omg/parser/model_parser.h"
#include "tf_adapter/common/adp_logger.h"

#define DATASET_REQUIRES(EXP, STATUS)    \
  do {                                   \
    if (!(EXP)) {                        \
      ADP_LOG(ERROR) << (STATUS);        \
      return (STATUS);                   \
    }                                    \
  } while (0)

namespace tensorflow {
namespace data {

class DatasetFunction {
  public:
    using Instance = uint32_t;
    DatasetFunction(const std::map<std::string, std::string> &init_options, const std::string &funcName,
        const DataTypeVector &input_types, const DataTypeVector &output_types,
        const std::vector<PartialTensorShape> &input_shape, const std::vector<PartialTensorShape> &output_shape)
      : init_options_(init_options),
        funcName_(funcName),
        input_types_(input_types),
        output_types_(output_types),
        input_shape_(input_shape),
        output_shape_(output_shape) {
      ADP_LOG(EVENT) << "[DatasetFunction] init success";
    };
    ~DatasetFunction();

    Status Initialize(const std::map<std::string, std::string> &session_options, FunctionLibraryDefinition &flib_def);
    Status Instantialte(Instance &instance);
    Status Run(Instance instance, std::vector<ge::Tensor> &in_tensors, std::vector<ge::Tensor> &out_tensors);
    Status Run(Instance instance, std::vector<Tensor> &in_tensors, std::vector<ge::Tensor> &out_tensors);

    Status RunWithStreamAsyn(Instance instance, void *stream, std::vector<ge::Tensor> &in_tensors,
        std::vector<ge::Tensor> &out_tensors);
    Status RunWithStreamAsyn(Instance instance, void *stream, std::vector<Tensor> &in_tensors,
        std::vector<ge::Tensor> &out_tensors);
    static Status RegisterNpuCancellation(std::function<void()> callback, std::function<void()>* deregister_fn);
    inline const std::vector<ge::DataType>& GetGeDataTypes() const { return ge_output_types_; }
    static bool HaveUnknowShape(const std::vector<PartialTensorShape> tf_shapes);
    static bool IsUnknowShape(const PartialTensorShape &tf_shape);
    static std::vector<int64_t> GetTfShapeDims(const PartialTensorShape &tf_shape);
    static std::vector<int64_t> GetTfShapeDims(const TensorShape &tf_shape);
    static Status ConvertDTStringTensor(ge::Tensor &input, uint64_t count, std::string &str_vec);
    static inline bool CheckMultiplyOverflow(int64_t a, int64_t b) {
      const static int64_t max_int64 = INT64_MAX;
      return (a != 0) && (b != 0) && (a > (max_int64 / b));
    }

    static inline bool CheckMultiplyOverflow(uint64_t a, uint64_t b) {
      const static uint64_t max_uint64 = UINT64_MAX;
      return (a != 0) && (b != 0) && (a > (max_uint64 / b));
    }

    static inline bool CheckAddOverflow(int64_t a, int64_t b) {
      const static int64_t max_int64 = INT64_MAX;
      return (a > (max_int64 - b));
    }

    static inline bool CheckAddOverflow(uint64_t a, uint64_t b) {
      const static uint64_t max_uint64 = UINT64_MAX;
      return (a > (max_uint64 - b));
    }

    static int64_t GetShapeDims(const std::vector<int64_t> &shape) {
      int64_t dims = 1;
      for (auto it : shape) {
        if (it < 0) {
          return -1;
        }

        if (CheckMultiplyOverflow(dims, it)) {
          return -1;
        }

        dims *= it;
      }
      return dims;
    }

    static bool EqualShape(const ge::Tensor &tensor, const std::vector<int64_t> &shape) {
      return std::equal(shape.cbegin(), shape.cend(), tensor.GetTensorDesc().GetShape().GetDims().cbegin());
    }

    static Status EqualShapes(const std::vector<ge::Tensor> &tensors,
        const std::vector<std::vector<int64_t>> &shapes) {
      if (tensors.size() != shapes.size()) {
        return errors::InvalidArgument("tensor num is match, out tensor num: ", tensors.size(),
            ", require num: ", shapes.size());
      }

      if (!std::equal(tensors.cbegin(), tensors.cend(), shapes.cbegin(), EqualShape)) {
        return errors::InvalidArgument("tensor shape is match");
      }
      return Status::OK();
    }

  private:
    void DumpTfGraph(const std::string &procPrifex, const std::string &func_name, const GraphDef &graph) const;
    void DumpGeComputeGraph(const std::string &procPrifex, const std::string &func_name,
        const ge::ComputeGraphPtr &graph) const;
    Status GeError(std::string errorDesc, ge::Status status) const;
    Status AddOpDef(Node &node) const;
    Status RefreshNodeDesc(Node &node) const;
    std::string BuildSubGraph(FunctionLibraryDefinition &flib_def, const std::string &func_name) const;
    Status CreateGeGraph(const std::shared_ptr<domi::ModelParser> &model_parser, FunctionLibraryDefinition &flib_def);
    ge::InputTensorInfo BuildTensorInfo(const std::shared_ptr<domi::ModelParser> &model_parser,
        DataType type, const PartialTensorShape &shape) const;
    std::vector<ge::InputTensorInfo> BuildInputTensorInfos(
        const std::shared_ptr<domi::ModelParser> &model_parser) const;
    Status BuildGeGraph(const Instance &instance, const std::shared_ptr<domi::ModelParser> &model_parser);
    Status InitGeGraph(FunctionLibraryDefinition &flib_def);
    static void LogOptions(const std::map<std::string, std::string> &options);

    static uint64_t NewGraphId() {
      static std::atomic<uint64_t> graphId{0};
      return graphId.fetch_add(1);
    }

    const std::string& GetPrefix() const { return prefix_; }

    const std::string prefix_ = "DatasetFunc";

    std::map<std::string, std::string> init_options_;
    const std::string funcName_;
    const DataTypeVector input_types_;
    const DataTypeVector output_types_;
    std::vector<ge::DataType> ge_output_types_;
    const std::vector<PartialTensorShape> &input_shape_;
    const std::vector<PartialTensorShape> &output_shape_;

    // std::string tf_session_;
    std::map<std::string, std::string> session_options_;

    std::shared_ptr<ge::Session> ge_session_ = nullptr;
    ge::Graph ge_graph_;
};
}  // namespace data
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_DATASET_FUNCTION_H_
