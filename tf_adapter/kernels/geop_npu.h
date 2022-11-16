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

#ifndef TENSORFLOW_KERNELS_GEOP_NPU_H_
#define TENSORFLOW_KERNELS_GEOP_NPU_H_

#include <unordered_map>
#include <atomic>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "graph/tensor.h"
#include "graph/utils/graph_utils.h"
#include "aoe_tuning_api.h"

namespace tensorflow {
using SessionId = uint64_t;
// aoe mode
using AoeInitializeFunc = Aoe::AoeStatus (*)(const std::map<Aoe::AscendString, Aoe::AscendString> &);
using AoeFinalizeFunc = Aoe::AoeStatus (*)();
using AoeCreateSessionFunc = Aoe::AoeStatus (*)(const std::map<Aoe::AscendString, Aoe::AscendString> &, SessionId &);
using AoeDestroySessionFunc = Aoe::AoeStatus (*)(SessionId);
using AoeSetGeSessionFunc = Aoe::AoeStatus (*)(SessionId, ge::Session*);
using AoeSetDependGraphFunc = Aoe::AoeStatus (*)(SessionId, std::vector<ge::Graph>&);
using AoeSetDependGraphsInputsFunc = Aoe::AoeStatus (*)(SessionId, std::vector<std::vector<ge::Tensor>> &);
using AoeSetTuningGraphInputFunc = Aoe::AoeStatus (*)(SessionId, std::vector<ge::Tensor> &);
using AoeSetTuningGraphFunc = Aoe::AoeStatus (*)(SessionId, ge::Graph &);
using AoeTuningGraphFunc = Aoe::AoeStatus (*)(SessionId, const std::map<Aoe::AscendString, Aoe::AscendString> &);

class GeOp : public AsyncOpKernel {
public:
  explicit GeOp(OpKernelConstruction *ctx);
  ~GeOp() override;
  void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override;

private:
  void Initialize(OpKernelConstruction *ctx);
  void Finalize();

  // global environment Initialize/Finalize, only invoke once for each process
  Status GlobalInitialize(OpKernelConstruction *ctx);
  void GlobalFinalize();

  // Build GraphDef from FunctionDef.
  Status BuildGraphDef(FunctionLibraryDefinition &flib_def, const std::vector<Tensor> &input_vec,
                       GraphDef &graph_def, bool &is_initialize, bool &is_allreduce);

  // Analyze sting input data
  Status AnalyzeStringInput(ge::Tensor &input, uint64_t count, const std::string *string_vector) const;

  // prepare input tensor
  Status BuildInputTensorInfo(OpKernelContext *const ctx,
                              std::vector<Tensor> &input_vec,
                              std::vector<std::string> &input_shapes,
                              std::vector<ge::Tensor> &inputs);
  // prepare output tensor
  Status BuildOutTensorInfo(OpKernelContext *ctx);

  // create input and output desc for NodeDef
  Status GenerateDesc(Node *&node);

  // parse onnx model in tensorflow node
  Status ParseOnnxGraphOpAttr(Node *&node) const;

  Status DomiFormatFromString(std::string format, int32_t &domi_format) const;

  Status GraphInputConvertToConst(OpKernelContext *ctx);

  Status GraphCheckInputEqualConstOp(Tensor &tensor, int32_t index, bool &is_equal);

  void AddNodeAttrs(Node *node, bool &is_initialize);

  int InitRebuildFlag(uint32_t cache_graph_id);

  bool IncrementGraphIdCount(uint32_t &graph_id);

  bool DecrementGraphIdCount(const std::string &tf_session, uint32_t &graph_id);

  void ClearGraphIdCount();

  void GetExecGraphId(uint32_t &cache_graph_id,
                      std::vector<std::string> input_shapes);

  void GetMsTuneConfig(std::map<std::string, std::string> init_options);

  void SetShapesToOutputDesc(const std::vector<std::string> &input_shapes,
                             const int &index, AttrValue &attr_shape_value) const;

  void BuildShapeNodeAndCacheArgNodes(Graph &graph);

  Status ChangeInputsShapeDesc();

  void AnalyzeInputDesc(void *tensor_ptr, ge::Tensor &input, ge::DataType type,
                        std::vector<std::string> &input_shapes) const;

  int RunTuning(std::vector<Tensor> &input_vec, std::vector<ge::Tensor> &inputs, const OpKernelContext *const ctx);

  std::string BuildSubGraph(FunctionLibraryDefinition *flib_def, const std::string &graph);

  void SetDynamicInput();

  void ProcessDpOpFuncDef(const Node &node) const;

  void BuildQueueDataAndGetNextFromQueue(Graph &graph, const Node &getnext_node,
                                         const std::string &channel_name) const;

  void HandleDpOpAndGetNextNodes(Graph &graph);

  void ChangeChannelNameAttr(NodeDef &node_def) const;

  bool IsDynamicConfig();

  PartialTensorShape MakeCompatShape(const PartialTensorShape &a, const PartialTensorShape &b);

  bool MaybeUpdateShape(OpKernelContext *const ctx);

  void UpdateInputsShapeDesc(Graph &graph);

  static const std::string INPUT_DESC;
  static const std::string OUTPUT_DESC;
  static const std::string SERIALIZE_FORMAT;
  static const std::string SERIALIZE_DATATYPE;
  static const std::string SERIALIZE_SHAPE;
  static const std::string SubGraph;

  static mutex mu_;
  static bool tuned_initialize_flag_;

  bool init_flag_;
  bool build_flag_;
  bool add_graph_flag_;
  bool sess_init_flag_;
  bool compute_graph_empty_;
  bool is_input_convert_;

  std::string input_shapes_;
  NameAttrList function_;
  std::string data_format_;
  uint32_t graph_id_;
  bool is_initialized_graph_;
  bool need_iteration_;
  std::string tf_session_;
  ge::Session *ge_session_;
  std::string job_type_;
  std::map<std::vector<std::string>, uint32_t> cache_graphs_;
  std::vector<std::pair<std::vector<std::string>, uint32_t>> graph_counts_;
  std::map<std::string, std::string> sess_options_;
  std::map<std::string, std::string> init_options_;
  static std::unordered_map<std::string, uint32_t> session_and_graph_id_map_;
  uint32_t iteration_per_loop_;
  bool is_host_graph_;
  std::map<std::string, std::string> graph_options_;
  std::map<int, TensorShape> outputs_shape_;
  std::string is_train_graph_;
  void *handle_;
  std::vector<Node*> dynamic_shape_nodes_;
  std::string dynamic_input_;
  std::string dynamic_graph_execute_mode_;
  std::string data_inputs_shape_range_;
  std::string getnext_inputs_shape_range_;
  bool need_compile_graph_first_;
  std::map<string, string> tune_options_;
  std::string is_dynamic_getnext_;
  std::string placeholder_index_;
  std::atomic_flag tuned_flag_;
  std::vector<std::pair<Tensor, int32_t>> remove_index_;
  std::string is_var_init_graph_;
  std::string recompute_mode_;
  std::string enable_graph_parallel_;
  std::string graph_parallel_option_path_;
  std::vector<absl::optional<PartialTensorShape>> input_shapes_vec_;
  bool jit_compile_;
  SessionId session_id_;
  AoeInitializeFunc aoe_initialize_;
  AoeFinalizeFunc aoe_finalize_;
  AoeCreateSessionFunc aoe_create_session_;
  AoeDestroySessionFunc aoe_destroy_session_;
  AoeSetGeSessionFunc aoe_set_gesession_;
  AoeSetDependGraphFunc aoe_set_dependgraphs_;
  AoeSetTuningGraphFunc aoe_set_tuninggraph_;
  AoeTuningGraphFunc aoe_tuning_graph_;
  AoeSetDependGraphsInputsFunc aoe_set_depend_graphs_inputs_;
  AoeSetTuningGraphInputFunc aoe_set_tuning_graph_input_;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_GEOP_NPU_H_
