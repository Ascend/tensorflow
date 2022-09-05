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
class NpuConcreteGraph : public OpExecutor {
 public:
  enum class ExecutionType { NPU, MIX };
  enum class LoopType { NPU_LOOP, BUILTIN_LOOP, HOST_LOOP, NO_LOOP };
  NpuConcreteGraph(const std::string &name, TensorDataTypes input_dtypes, TensorDataTypes output_dtypes,
                   uint64_t ge_graph_id, std::unique_ptr<tensorflow::Graph> graph)
      : OpExecutor(name, input_dtypes, output_dtypes), ge_graph_id_(ge_graph_id), graph_(std::move(graph)) {
    graph_->ToGraphDef(&graph_def_);
    SetCacheStrategy(CacheStrategy::BY_OP_NAME);
    // Cache vector for performance
    input_handles_.resize(InputTypes().size(), nullptr);
    output_handles_.resize(OutputTypes().size(), nullptr);
    for (size_t i = 0; i < InputTypes().size(); i++) {
      consumed_inputs_.push_back(i);
    }
    for (size_t i = 0; i < OutputTypes().size(); i++) {
      produced_outputs_.push_back(i);
    }
  }
  ~NpuConcreteGraph() = default;

  const std::string &Type() const override {
    const static std::string kType = "NpuFunctionOp";
    return kType;
  }

  void RunImpl(TFE_Context *context, NpuDevice *device, int tf_num_inputs, TFE_TensorHandle **tf_inputs,
               int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const override;

  uint64_t GeGraphId() const { return ge_graph_id_; }

  const std::map<int, std::shared_ptr<IteratorResourceProvider>> &ConsumedIteratos() const {
    return consumed_iterators_;
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

  void Load(TFE_Context *context, NpuDevice *device, bool &loaded, TF_Status *status) const;
  void UnLoad(TFE_Context *context, NpuDevice *device, TF_Status *status) const;

  const std::string &GraphLoopTypeString() const;

  bool NeedFuzzCompile() const;

  void RunAoeTuning(TFE_Context *context, NpuDevice *device, std::vector<TFE_TensorHandle *> inputs, bool loaded,
                    TF_Status *status) const;

  void SetFunctionOpFlag(bool is_function_op) {
    function_op_ = is_function_op;
  }

 protected:
  std::string AttachedDebugString() const override;
  void SetBuilt(bool built) const { built_ = built; }
  bool Built() const { return built_; }
  ExecutionType execution_type_{ExecutionType::NPU};
  uint64_t ge_graph_id_;
  std::unique_ptr<tensorflow::Graph> graph_;
  mutable tensorflow::GraphDef graph_def_;
  mutable absl::optional<bool> fuzz_compile_;
  tensorflow::NodeDef mixed_ndef_;
  bool mutable built_{false};
  LoopType loop_type_{LoopType::NO_LOOP};
  std::map<int, std::shared_ptr<IteratorResourceProvider>> consumed_iterators_;
  std::vector<int32_t> consumed_inputs_;
  std::vector<int32_t> produced_outputs_;
  std::map<int32_t, int64_t> bypass_outputs_;
  std::vector<TFE_TensorHandle *> mutable input_handles_;
  std::vector<TFE_TensorHandle *> mutable output_handles_;

 private:
  bool mutable graph_def_serialized_{false};
  bool mutable empty_ge_graph_{false};
  bool mutable function_op_{false};
};

class NpuMutableConcreteGraph : public NpuConcreteGraph {
 public:
  NpuMutableConcreteGraph(const std::string &name, TensorDataTypes input_dtypes, TensorDataTypes output_dtypes,
                          uint64_t ge_graph_id, std::unique_ptr<tensorflow::Graph> graph)
      : NpuConcreteGraph(name, input_dtypes, output_dtypes, ge_graph_id, std::move(graph)),
        consumed_types_(InputTypes()),
        produced_types_(OutputTypes()) {}
  ~NpuMutableConcreteGraph() = default;
  void SetGraph(std::unique_ptr<tensorflow::Graph> graph) { graph_.swap(graph); }
  tensorflow::Graph *MutableGraph() const { return graph_.get(); }

  void SetConsumedInputs(const std::set<int32_t> &v) {
    consumed_types_.clear();
    consumed_inputs_.clear();
    input_handles_.resize(v.size());
    for (auto index : v) {
      if (index < 0) {
        return;
      }
      (void)consumed_types_.emplace_back(InputTypes()[static_cast<size_t>(index)]);
      consumed_inputs_.emplace_back(index);
    }
  }

  void SetProducedOutputs(const std::set<int32_t> &v) {
    produced_types_.clear();
    produced_outputs_.clear();
    output_handles_.resize(v.size());
    for (auto index : v) {
      if (index < 0) {
        return;
      }
      (void)produced_types_.emplace_back(OutputTypes()[static_cast<size_t>(index)]);
      produced_outputs_.emplace_back(index);
    }
  }

  const TensorDataTypes &ConsumedTypes() const { return consumed_types_; }

  const TensorDataTypes &ProducedTypes() const { return produced_types_; }

  void SetBypassOutputs(const std::map<int32_t, int64_t> &v) { bypass_outputs_ = v; }

  void SetLoopType(LoopType type) { loop_type_ = type; }

  void SetExecutionType(ExecutionType type) { execution_type_ = type; }

  void SetMixedFunctionName(const std::string &fn) { mixed_ndef_.set_op(fn); }

  void SetConsumedIterators(const std::map<int, std::shared_ptr<IteratorResourceProvider>> &resources) {
    consumed_iterators_ = resources;
  }

  void SetNpuResources(const std::map<int32_t, tensorflow::ResourceHandle> &resources) { npu_resources_ = resources; }
  void SetCpuResources(const std::map<int32_t, tensorflow::ResourceHandle> &resources) { cpu_resources_ = resources; }

  const std::map<int32_t, tensorflow::ResourceHandle> &NpuResources() const { return npu_resources_; }
  const std::map<int32_t, tensorflow::ResourceHandle> &CpuResources() const { return cpu_resources_; }

  tensorflow::Status TryTransToNpuLoopGraph(TFE_Context *context);

 private:
  std::map<int32_t, tensorflow::ResourceHandle> npu_resources_;
  std::map<int32_t, tensorflow::ResourceHandle> cpu_resources_;
  TensorDataTypes consumed_types_;
  TensorDataTypes produced_types_;
};

}  // namespace npu

#endif  // NPU_DEVICE_CORE_OP_EXECUTORS_NPU_CONCRETE_GRAPH_H
