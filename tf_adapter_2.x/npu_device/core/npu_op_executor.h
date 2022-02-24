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

#ifndef NPU_DEVICE_CORE_NPU_TASK_SPEC_H
#define NPU_DEVICE_CORE_NPU_TASK_SPEC_H

#include "tensorflow/c/eager/c_api.h"

#include <algorithm>
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
class NpuConcreteGraph;
class NpuDevice;

class OpExecutor {
 public:
  static std::shared_ptr<OpExecutor> Create(TFE_Context *context, NpuDevice *device, const tensorflow::NodeDef &ndef,
                                            int num_inputs, TFE_TensorHandle **inputs, TF_Status *s);
  const std::string &Op() const { return ndef_.op(); }
  const tensorflow::NodeDef &NodeDef() const { return ndef_; }
  const TensorDataTypes &InputTypes() const { return input_dtypes_; }
  const TensorShapes &InputShapes() const { return input_shapes_; }
  const TensorDataTypes &OutputTypes() const { return output_dtypes_; }
  const tensorflow::OpRegistrationData *OpRegistrationData() const { return op_spec_; }

  tensorflow::NodeDef ParserNodeDef() const {
    tensorflow::NodeDef ndef;
    ndef.MergeFrom(ndef_);
    ndef.MergeFrom(attached_attrs_);
    return ndef;
  }

  CacheStrategy GetCacheStrategy() const { return cache_strategy_; }
  void SetCacheStrategy(CacheStrategy strategy) { cache_strategy_ = strategy; }

  std::string DebugString() const {
    std::stringstream ss;
    ss << Type() << std::endl;
    ss << NodeDef().DebugString() << std::endl;
    ss << attached_attrs_.DebugString() << std::endl;
    if (OpRegistrationData() != nullptr) {
      ss << OpRegistrationData()->op_def.DebugString() << std::endl;
    }
    ss << AttachedDebugString() << std::endl;
    return ss.str();
  }

  virtual const std::string &Type() const = 0;

  void Run(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs, int num_outputs,
           TFE_TensorHandle **outputs, TF_Status *status) const {
    DLOG() << "Start executing op " << Op() << " by " << Type() << " executor";
    RunImpl(context, device, num_inputs, inputs, num_outputs, outputs, status);
  }

  virtual void RunImpl(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs,
                       int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const = 0;

 protected:
  virtual std::string AttachedDebugString() const = 0;
  OpExecutor(const tensorflow::OpRegistrationData *op_spec, const tensorflow::NodeDef &ndef, TensorShapes input_shapes)
      : op_spec_(op_spec), cache_strategy_(CacheStrategy::DEFAULT) {
    TensorDataTypes input_dtypes;
    TensorDataTypes output_dtypes;
    tensorflow::InOutTypesForNode(ndef, op_spec->op_def, &input_dtypes, &output_dtypes);
    ndef_ = ndef;
    input_dtypes_ = std::move(input_dtypes);
    input_shapes_ = std::move(input_shapes);
    output_dtypes_ = std::move(output_dtypes);
  };
  OpExecutor(const std::string &function_name, TensorDataTypes input_dtypes, TensorDataTypes output_dtypes)
      : op_spec_(nullptr), cache_strategy_(CacheStrategy::DEFAULT) {
    ndef_.set_op(function_name);
    input_dtypes_ = std::move(input_dtypes);
    output_dtypes_ = std::move(output_dtypes);
  };
  ~OpExecutor() = default;
  const tensorflow::OpRegistrationData *op_spec_;  // 算子IR注册的信息，非实例
  tensorflow::NodeDef ndef_;                       // 节点的NodeDef，主要存储实例化属性信息
  tensorflow::NodeDef attached_attrs_;             // NPU 附件的属性
  TensorDataTypes input_dtypes_;
  TensorShapes input_shapes_;
  TensorDataTypes output_dtypes_;
  CacheStrategy cache_strategy_;
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_TASK_SPEC_H
