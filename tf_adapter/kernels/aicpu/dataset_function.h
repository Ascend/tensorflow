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
#include <nlohmann/json.hpp>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"

#include "graph/tensor.h"
#include "graph/utils/graph_utils.h"
#include "graph/compute_graph.h"
#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_factory.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/util/infershape_util.h"
#include "tf_adapter/common/common.h"

#include "acl/acl_base.h"
#include "acl/acl_mdl.h"
#include "ge/ge_ir_build.h"

#define DATASET_REQUIRES(EXP, STATUS)    \
  do {                                   \
    if (!(EXP)) {                        \
      ADP_LOG(ERROR) << (STATUS);        \
      return (STATUS);                   \
    }                                    \
  } while (0)

#define DATASET_REQUIRES_WARNING(EXP, STATUS)    \
  do {                                           \
    if (!(EXP)) {                                \
      ADP_LOG(WARNING) << (STATUS);              \
      return (STATUS);                           \
    }                                            \
  } while (0)

#define DATASET_REQUIRES_RT_NULL(EXP, STATUS)    \
  do {                                           \
    if (!(EXP)) {                                \
      ADP_LOG(ERROR) << (STATUS);                \
      return nullptr;                            \
    }                                            \
  } while (0)

#define DATASET_REQUIRES_RT_VOID(EXP, CALLBACK)    \
  do {                                             \
    if (!(EXP)) {                                  \
      CALLBACK;                                    \
      return;                                      \
    }                                              \
  } while (0)

namespace tensorflow {
namespace data {
class DatasetFunction {
  public:
    using ModelId = uint32_t;
    using TensorPartialShapes = tensorflow::gtl::InlinedVector<tensorflow::PartialTensorShape, 4>;
    using TensorDataTypes = tensorflow::gtl::InlinedVector<tensorflow::DataType, 4>;
    DatasetFunction(const std::map<std::string, std::string> &init_options, const std::string &funcName,
        const DataTypeVector &input_types, const DataTypeVector &output_types,
        const std::vector<PartialTensorShape> &input_shape, const std::vector<PartialTensorShape> &output_shape)
      : init_options_(init_options),
        funcName_(funcName),
        input_types_(input_types),
        output_types_(output_types),
        input_shape_(input_shape),
        output_shape_(output_shape) {
      ADP_LOG(EVENT) << "[DatasetFunction] init success.";
    };
    ~DatasetFunction();

    Status Instantialte();
    Status Initialize(const std::map<std::string, std::string> &session_options, FunctionLibraryDefinition &flib_def);
    Status Run(ModelId model_id, aclmdlDataset* in_dataset, aclmdlDataset* out_dataset) const;
    Status RunWithStreamAsyn(ModelId model_id, aclmdlDataset* in_dataset, aclmdlDataset* out_dataset,
        aclrtStream stream) const;
    Status LoadGeModelFromMem(ModelId &model_id);

    static Status RegisterNpuCancellation(std::function<void()> callback, std::function<void()>* deregister_fn);
    inline const std::vector<ge::DataType>& GetGeDataTypes() const { return ge_output_types_; }
    static bool HaveUnknowShape(const std::vector<PartialTensorShape> tf_shapes);
    static bool IsUnknowShape(const PartialTensorShape &tf_shape);
    static std::vector<int64_t> GetTfShapeDims(const PartialTensorShape &tf_shape);
    static std::vector<int64_t> GetTfShapeDims(const TensorShape &tf_shape);

    static aclmdlDataset *CreateAclInputDatasetWithTFTensors(std::vector<Tensor> &tf_tensors);
    static Status TransTfTensorToDataBuffer(aclmdlDataset *input_dataset, Tensor &tf_tensor);
    static void *ConvertDTStringTensor(const Tensor &tf_tensor, uint64_t &tensor_size);
    static void DestroyAclInputDataset(aclmdlDataset *input);
    static aclmdlDataset *CreateAclOutputDataset(const ModelId model_id);
    static void DestroyAclOutputDataset(aclmdlDataset *output, const bool isFree);
    static aclmdlDesc *CreateAclModelDesc(const ModelId model_id);
    static Status GetAclTenorDescDims(aclTensorDesc *desc, std::vector<int64_t> &ret_dims);
    static void *ReAllocDeviceMem(void *addr, size_t len);
    bool IsSplitGraph() const;

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
    void MarkDvppGraphNodes(Graph &sub_graph_tf, std::vector<Node*> &dvpp_graph_nodes,
        const std::vector<std::string> acc_while_list) const;
    void MarkConstNodes(const Graph &sub_graph_tf, std::vector<Node*> &dvpp_graph_nodes) const;
    bool CheckCorrectness(const tensorflow::Graph &sub_graph_tf, const std::vector<Node*> dvpp_graph_nodes,
        const std::vector<Node*> host_graph_nodes) const;
    void MarkHostGraphNodes(const tensorflow::Graph &sub_graph_tf, const std::vector<Node*> dvpp_graph_nodes,
        std::vector<Node*> &host_graph_nodes) const;
    void UpdateAttrsForArgOp(tensorflow::Node *arg, const tensorflow::Edge *edge) const;
    void CreateHostGraph(tensorflow::Graph &sub_graph_host, const std::vector<Node*> host_graph_nodes,
        std::vector<Node*> &dvpp_graph_nodes, std::map<tensorflow::Node*, int64> &dvpp_arg_indexs) const;
    void CreateDvppGraph(tensorflow::Graph &sub_graph_dvpp, const std::vector<Node*> dvpp_graph_nodes,
        const std::map<tensorflow::Node*, int64> dvpp_arg_indexs) const;
    std::string SplitSubGraph(FunctionLibraryDefinition &flib_def, const std::vector<std::string> acc_while_list);
    Status InitAccelateOpList(std::vector<std::string> &acc_while_list) const;
    Status ReadJsonFile(const string &json_file_path, nlohmann::json &json_read) const;
    tensorflow::DataType EdgeDataType(const tensorflow::Edge &edge) const;
    std::string GetSocVersion() const;

    void AssembleParserAddons(const tensorflow::FunctionLibraryDefinition &lib_def, tensorflow::Graph &graph) const;
    void AssembleOpDef(tensorflow::Node &n) const;
    void AssembleInputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node &n) const;
    void AssembleOutputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node &n) const;
    PartialTensorShape MakeCompatShape(const PartialTensorShape &a, const PartialTensorShape &b) const;
    void UpdateShapeForArgOp(Graph &graph) const;

    void DumpTfGraph(const std::string &procPrifex, const std::string &func_name, const GraphDef &graph) const;
    void DumpGeComputeGraph(const std::string &procPrifex, const std::string &func_name,
        const ge::ComputeGraphPtr &graph) const;
    Status GeError(std::string errorDesc, ge::Status status) const;
    std::string SerializeGraph(tensorflow::Graph &input_graph) const;
    Status BuildSubGraph(FunctionLibraryDefinition &flib_def, tensorflow::Graph &sub_graph,
        const std::string &func_name) const;
    Status CreateGeGraph(const std::shared_ptr<domi::ModelParser> &model_parser, FunctionLibraryDefinition &flib_def);
    ge::InputTensorInfo BuildTensorInfo(const std::shared_ptr<domi::ModelParser> &model_parser,
        DataType type, const PartialTensorShape &shape) const;
    std::vector<ge::InputTensorInfo> BuildInputTensorInfos(
        const std::shared_ptr<domi::ModelParser> &model_parser) const;
    Status BuildGeGraph(const ModelId &model_id, const std::shared_ptr<domi::ModelParser> &model_parser);
    Status InitGeGraph(FunctionLibraryDefinition &flib_def);
    static void LogOptions(const std::map<std::string, std::string> &options);

    static uint64_t NewGraphId() {
      static std::atomic<uint64_t> graphId{0};
      return graphId.fetch_add(1);
    }

    const std::string& GetPrefix() const { return prefix_; }
    const std::string prefix_ = "DatasetFunc";

    template <typename T>
    tensorflow::AttrValue BuildDescAttr(T shapes, TensorDataTypes types) const;

    std::map<std::string, std::string> init_options_;
    const std::string funcName_;
    const DataTypeVector input_types_;
    const DataTypeVector output_types_;
    std::vector<ge::DataType> ge_output_types_;
    const std::vector<PartialTensorShape> &input_shape_;
    const std::vector<PartialTensorShape> &output_shape_;

    std::map<std::string, std::string> session_options_;

    std::shared_ptr<ge::Session> ge_session_ = nullptr;
    ge::Graph ge_graph_;
    ge::ModelBufferData ge_model_;
    bool run_split_graph_ = false;
}; // class DatasetFunction

struct Items {
public:
  void Update() {
    int64_t interval_time = end_time - start_time;
    if ((start_time <= 0LL) || (end_time <= 0LL) || (interval_time <= 0LL)) {
      start_time = INT64_MIN;
      return;
    }
    // if data overflow, stop record
    if (DatasetFunction::CheckAddOverflow(total_time, interval_time)) {
      return;
    }
    total_time += interval_time;
    total_records++;
    min_time = std::min(min_time, interval_time);
    max_time = std::max(max_time, interval_time);
    start_time = INT64_MIN;
  };

  int64_t thread_id = INT64_MIN;
  int64_t start_time = INT64_MIN;
  int64_t end_time = INT64_MIN;
  int64_t min_time = INT64_MAX;
  int64_t max_time = INT64_MIN;
  int64_t avg_time = 0LL;
  int64_t total_time = 0LL;
  int64_t total_records = 0LL;
};

class TimeStatistic {
public:
  explicit TimeStatistic(int64_t total_threads);
  ~TimeStatistic() {};

  void RecordStartTime(Items &it) const;
  void RecordEndTime(Items &it) const;
  void UpdateWithTimeTag(Items &it, std::shared_ptr<Items> &tag) const;
  void ShowTimeStatistic();

  // record time statistics for GetNextInter API
  Items statis;
  std::vector<Items> statis_threads;
  std::vector<Items> statis_threads_ge;
  int64_t max_threads = 0LL;
  bool stop_record = true;
};
}  // namespace data
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_DATASET_FUNCTION_H_
