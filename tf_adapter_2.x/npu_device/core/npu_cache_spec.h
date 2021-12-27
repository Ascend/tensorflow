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

#ifndef TENSORFLOW_NPU_CACHE_SPEC_H
#define TENSORFLOW_NPU_CACHE_SPEC_H

#include "tensorflow/c/eager/c_api.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"

#include "npu_dp.h"
#include "npu_logger.h"
#include "npu_parser.h"
#include "npu_types.h"

namespace npu {

class TaskSpec {
 public:
  virtual bool IsFunctionOp() const = 0;
  bool ShouldFallback() const { return !fallback_reason_.empty(); };
  std::string FallbackReason() const { return fallback_reason_; };
  std::string Op() const { return ndef_.op(); }
  virtual std::string DebugString() const = 0;
  tensorflow::NodeDef NodeDef() const { return ndef_; }
  const TensorDataTypes &InputTypes() const { return input_dtypes_; }
  const TensorShapes &InputShapes() const { return input_shapes_; }
  const TensorDataTypes &OutputTypes() const { return output_dtypes_; }
  virtual const tensorflow::OpRegistrationData *OpRegistrationData() const { return op_spec_; }

 protected:
  TaskSpec() : op_spec_(nullptr){};
  ~TaskSpec() = default;
  const tensorflow::OpRegistrationData *op_spec_;  // 算子IR注册的信息，非实例
  tensorflow::NodeDef ndef_;                       // 节点的NodeDef，主要存储实例化属性信息
  TensorDataTypes input_dtypes_;
  TensorShapes input_shapes_;
  TensorDataTypes output_dtypes_;
  std::string fallback_reason_;
};

class OpSpec : public TaskSpec {
 public:
  OpSpec(const tensorflow::OpRegistrationData *op_spec, tensorflow::NodeDef ndef, TensorShapes input_shapes,
         TensorPartialShapes output_shapes, std::string reason)
      : always_infer_shape_(false), partial_output_shapes_(output_shapes) {
    TensorDataTypes input_dtypes;
    TensorDataTypes output_dtypes;
    tensorflow::InOutTypesForNode(ndef, op_spec->op_def, &input_dtypes, &output_dtypes);
    op_spec_ = op_spec;
    ndef_ = std::move(ndef);
    input_dtypes_ = std::move(input_dtypes);
    input_shapes_ = std::move(input_shapes);
    output_dtypes_ = std::move(output_dtypes);

    fallback_reason_ = std::move(reason);
    if (ShouldFallback()) {
      return;
    }
    TensorShapes shapes;
    shapes.resize(output_shapes.size());
    for (size_t i = 0; i < output_shapes.size(); i++) {
      // 如果不是函数算子，那么必须要求inferShape输出确定的结果
      if (!output_shapes[i].AsTensorShape(&shapes[i])) {
        fallback_reason_ = tensorflow::strings::StrCat("output", i, " unknown shape ", output_shapes[i].DebugString());
        break;
      }
    }

    if (!ShouldFallback()) {
      output_shapes_ = shapes;
      AssembleInputDesc(input_shapes_, input_dtypes_, &attached_attrs_);
      AssembleOutputDesc(output_shapes_, output_dtypes_, &attached_attrs_);
    }
  }

  OpSpec(const tensorflow::OpRegistrationData *op_spec, tensorflow::NodeDef ndef, TensorShapes input_shapes,
         std::string reason)
      : always_infer_shape_(true) {
    TensorDataTypes input_dtypes;
    TensorDataTypes output_dtypes;
    tensorflow::InOutTypesForNode(ndef, op_spec->op_def, &input_dtypes, &output_dtypes);

    op_spec_ = op_spec;
    ndef_ = std::move(ndef);
    input_dtypes_ = std::move(input_dtypes);
    input_shapes_ = std::move(input_shapes);
    output_dtypes_ = std::move(output_dtypes);
    fallback_reason_ = std::move(reason);

    if (!ShouldFallback()) {
      AssembleInputDesc(input_shapes_, input_dtypes_, &attached_attrs_);
    }
  }

  ~OpSpec() = default;
  bool IsFunctionOp() const override { return false; }
  bool ShouldInferShape() const { return always_infer_shape_; }
  const TensorShapes &OutputShapes() const { return output_shapes_; }
  const TensorPartialShapes &OutputPartialShapes() const { return partial_output_shapes_; }
  tensorflow::NodeDef ParserNodeDef() const {
    tensorflow::NodeDef ndef;
    ndef.MergeFrom(ndef_);
    ndef.MergeFrom(attached_attrs_);
    return ndef;
  }
  std::string DebugString() const override {
    std::stringstream ss;
    ss << NodeDef().DebugString() << std::endl;
    ss << attached_attrs_.DebugString() << std::endl;
    ss << OpRegistrationData()->op_def.DebugString() << std::endl;
    for (size_t i = 0; i < output_dtypes_.size(); i++) {
      if (always_infer_shape_ || ShouldFallback()) {
        ss << "output " << i << " " << tensorflow::DataTypeString(output_dtypes_[i]) << " <need re-infer>" << std::endl;
      } else {
        ss << "output " << i << " " << tensorflow::DataTypeString(output_dtypes_[i]) << " "
           << partial_output_shapes_[i].DebugString() << std::endl;
      }
    }
    if (ShouldFallback()) {
      ss << "Fallback reason " << fallback_reason_;
    }
    return ss.str();
  }

 private:
  bool always_infer_shape_;
  TensorShapes output_shapes_;
  TensorPartialShapes partial_output_shapes_;
  tensorflow::NodeDef attached_attrs_;
};

class FuncSpec : public TaskSpec {
  using TensorDataTypes = tensorflow::gtl::InlinedVector<tensorflow::DataType, 4>;

 public:
  using PruneInputsFunc =
    std::function<void(int num_inputs, TFE_TensorHandle **inputs, std::vector<TFE_TensorHandle *> &)>;
  FuncSpec(const tensorflow::OpRegistrationData *op_spec, tensorflow::NodeDef ndef, uint64_t ge_graph_id,
           std::unique_ptr<const tensorflow::GraphDef> graph, PruneInputsFunc prune_func,
           std::map<int, std::shared_ptr<npu::IteratorResourceProvider>> dependent_host_resources, std::string reason = "")
      : ge_graph_id_(ge_graph_id),
        graph_(std::move(graph)),
        prune_func_(std::move(prune_func)),
        dependent_host_resources_(std::move(dependent_host_resources)) {
    TensorDataTypes input_dtypes;
    TensorDataTypes output_dtypes;
    tensorflow::InOutTypesForNode(ndef, op_spec->op_def, &input_dtypes, &output_dtypes);

    op_spec_ = op_spec;
    ndef_ = std::move(ndef);
    input_dtypes_ = std::move(input_dtypes);
    output_dtypes_ = std::move(output_dtypes);
    fallback_reason_ = std::move(reason);
  }
  ~FuncSpec() = default;
  bool IsFunctionOp() const override { return true; }

  uint64_t GeGraphId() const { return ge_graph_id_; }

  const std::map<int, std::shared_ptr<npu::IteratorResourceProvider>> &DependentHostResources() const {
    return dependent_host_resources_;
  }

  const tensorflow::GraphDef *GraphDef() const { return graph_.get(); }

  void SetBuilt() const { built_.store(true); }
  bool Built() const { return built_; }

  void SetNeedLoop(bool loop) const { need_loop_.store(loop); }
  void SetBuiltinLoop(bool loop) const { builtin_loop_.store(loop); }
  bool NeedLoop() const { return need_loop_; }
  bool BuiltinLoop() const { return builtin_loop_; }

  void PruneInputs(int num_inputs, TFE_TensorHandle **inputs, std::vector<TFE_TensorHandle *> &pruned) const {
    prune_func_(num_inputs, inputs, pruned);
  }
  std::string DebugString() const override {
    std::stringstream ss;
    ss << NodeDef().DebugString() << std::endl;
    ss << OpRegistrationData()->op_def.DebugString() << std::endl;
    ss << "Ge graph id " << ge_graph_id_ << std::endl;
    for (size_t i = 0; i < output_dtypes_.size(); i++) {
      ss << "output " << i << " " << tensorflow::DataTypeString(output_dtypes_[i]) << std::endl;
    }
    if (ShouldFallback()) {
      ss << "Fallback reason " << fallback_reason_;
    }
    return ss.str();
  }

 private:
  uint64_t ge_graph_id_;
  std::unique_ptr<const tensorflow::GraphDef> graph_;
  PruneInputsFunc prune_func_;
  const std::map<int, std::shared_ptr<IteratorResourceProvider>> dependent_host_resources_;
  std::atomic_bool mutable built_{false};
  std::atomic_bool mutable need_loop_{false};
  std::atomic_bool mutable builtin_loop_{false};
};
}  // namespace npu

#endif  // TENSORFLOW_NPU_CACHE_SPEC_H
