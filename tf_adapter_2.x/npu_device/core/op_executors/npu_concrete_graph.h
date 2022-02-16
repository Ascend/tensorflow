/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef NPU_DEVICE_CORE_OP_EXECUTORS_NPU_CONCRETE_GRAPH_H
#define NPU_DEVICE_CORE_OP_EXECUTORS_NPU_CONCRETE_GRAPH_H

#include "npu_op_executor.h"

namespace npu {
using PruneInputsFunc = std::function<void(int, TFE_TensorHandle **, std::vector<TFE_TensorHandle *> &)>;

class NpuConcreteGraph : public OpExecutor {
 public:
  NpuConcreteGraph(const std::string &name, TensorDataTypes input_dtypes, TensorDataTypes output_dtypes,
                   uint64_t ge_graph_id, std::unique_ptr<tensorflow::Graph> graph)
      : OpExecutor(name, input_dtypes, output_dtypes),
        ge_graph_id_(ge_graph_id),
        graph_(std::move(graph)),
        prune_func_(nullptr) {
    graph_->ToGraphDef(&graph_def_);
    SetCacheStrategy(CacheStrategy::BY_OP_NAME);
  }

  const std::string &Type() const override {
    const static std::string kType = "NpuFunctionOp";
    return kType;
  }

  std::string AttachedDebugString() const override;

  void RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs, int num_outputs,
               TFE_TensorHandle **outputs, TF_Status *status) const override;

  uint64_t GeGraphId() const { return ge_graph_id_; }

  const std::map<int, std::shared_ptr<IteratorResourceProvider>> &DependentHostResources() const {
    return dependent_host_resources_;
  }

  const tensorflow::GraphDef &GraphDef() const {
    if (!graph_def_serialized_) {
      graph_->ToGraphDef(&graph_def_);
    }
    return graph_def_;
  }

  const tensorflow::Graph *Graph() const { return graph_.get(); }

  void RunOneShot(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs, int num_outputs,
                  TFE_TensorHandle **outputs, TF_Status *status) const;

  void Load(TFE_Context *context, NpuDevice *device, TF_Status *status) const;
  void UnLoad(TFE_Context *context, NpuDevice *device, TF_Status *status) const;

  void PruneInputs(int num_inputs, TFE_TensorHandle **inputs, std::vector<TFE_TensorHandle *> &pruned) const {
    prune_func_(num_inputs, inputs, pruned);
  }

  bool NeedLoop() const { return need_loop_; }
  bool BuiltinLoop() const { return builtin_loop_; }

  const std::map<int32_t, tensorflow::ResourceHandle> &GetNpuResources() { return npu_resources_; }

  const std::map<int32_t, tensorflow::ResourceHandle> &GetMirroredResources() { return mirrored_resources_; }

 protected:
  void SetBuilt(bool built) const { built_ = built; }
  bool Built() const { return built_; }
  tensorflow::NodeDef AsNodeDef(const std::string name) {
    tensorflow::NodeDef ndef;
    ndef.set_op(name);
    return ndef;
  }
  uint64_t ge_graph_id_;
  std::unique_ptr<tensorflow::Graph> graph_;
  mutable tensorflow::GraphDef graph_def_;
  PruneInputsFunc prune_func_;
  bool mutable built_{false};
  bool mutable need_loop_{false};
  bool mutable builtin_loop_{false};
  bool mutable is_cpu_graph_{false};
  std::map<int32_t, tensorflow::ResourceHandle> npu_resources_;
  std::map<int32_t, tensorflow::ResourceHandle> mirrored_resources_;
  std::map<int, std::shared_ptr<IteratorResourceProvider>> dependent_host_resources_;

 private:
  bool mutable graph_def_serialized_{false};
  bool mutable empty_ge_graph_{false};
};

class NpuMutableConcreteGraph : public NpuConcreteGraph {
 public:
  NpuMutableConcreteGraph(const std::string &name, TensorDataTypes input_dtypes, TensorDataTypes output_dtypes,
                          uint64_t ge_graph_id, std::unique_ptr<tensorflow::Graph> graph)
      : NpuConcreteGraph(name, input_dtypes, output_dtypes, ge_graph_id, std::move(graph)) {}
  void SetGraph(std::unique_ptr<tensorflow::Graph> graph) { graph_.swap(graph); }
  tensorflow::Graph *MutableGraph() { return graph_.get(); }

  void SetPruneInputsFunc(PruneInputsFunc func) { prune_func_ = func; }

  void SetNeedLoop(bool loop) { need_loop_ = loop; }

  void SetBuiltinLoop(bool loop) { builtin_loop_ = loop; }

  void SetIsCpuGraph(bool v) { is_cpu_graph_ = v; }

  void SetNpuResources(const std::map<int32_t, tensorflow::ResourceHandle> &resources) { npu_resources_ = resources; }

  void SetMirroredResources(const std::map<int32_t, tensorflow::ResourceHandle> &resources) {
    mirrored_resources_ = resources;
  }

  void SetDependentHostResources(const std::map<int, std::shared_ptr<IteratorResourceProvider>> &resources) {
    dependent_host_resources_ = resources;
  }
};

}  // namespace npu

#endif  // NPU_DEVICE_CORE_OP_EXECUTORS_NPU_CONCRETE_GRAPH_H
