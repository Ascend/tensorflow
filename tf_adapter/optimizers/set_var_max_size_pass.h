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

#ifndef TENSORFLOW_SET_VAR_MAX_SIZE_PASS_H_
#define TENSORFLOW_SET_VAR_MAX_SIZE_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class SetVarMaxSizePass : public GraphOptimizationPass {
 public:
  SetVarMaxSizePass() = default;
  ~SetVarMaxSizePass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;
  Status AssignMaxSizeToVarOutNodes(const Node *node) const;
  Status SetMaxSizeListNodes(Node *node) const;
  Status AssignConstToVarOutNodes(const Node *node, std::vector<std::string> &input_names) const;
  Status SetConstListNodes(Node *node, std::vector<std::string> &input_names) const;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_SET_VAR_MAX_SIZE_PASS_H_
