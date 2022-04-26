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

#include "tf_adapter/optimizers/om_set_var_max_size_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/public/session_options.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/npu_attrs.h"

namespace tensorflow {
const std::string ATTR_NAME_OP_MAX_SIZE = "_op_max_size";
const std::string ATTR_NAME_CONST_INPUT_NAME = "_const_input";

Status SetVarMaxSizePass::SetMaxSizeListNodes(Node *node) const {
  std::vector<int64> attr_values;
  for (const Edge *in : node->in_edges()) {
    REQUIRES_NOT_NULL(in);
    Node *src_node = in->src();
    REQUIRES_NOT_NULL(src_node);
    int64 attr_value = 0;
    Status s = GetNodeAttr(src_node->attrs(), ATTR_NAME_OP_MAX_SIZE, &attr_value);
    if ((s.ok()) && (attr_value > 0)) {
      attr_values.push_back(attr_value);
    } else {
      attr_values.push_back(0);
    }
  }
  node->AddAttr(ATTR_NAME_OP_MAX_SIZE, attr_values);
  return Status::OK();
}
Status SetVarMaxSizePass::AssignMaxSizeToVarOutNodes(const Node *node) const {
  for (const Edge *out : node->out_edges()) {
    REQUIRES_NOT_NULL(out);
    Node *dst_node = out->dst();
    REQUIRES_NOT_NULL(dst_node);
    int64 attr_value = 0;
    Status s = GetNodeAttr(node->attrs(), ATTR_NAME_OP_MAX_SIZE, &attr_value);
    if ((s.ok()) && (attr_value > 0)) {
      std::vector<int64> attr_vector;
      Status result = GetNodeAttr(dst_node->attrs(), ATTR_NAME_OP_MAX_SIZE, &attr_vector);
      if ((result.ok()) && (!attr_vector.empty())) {
        ADP_LOG(DEBUG) << "The node : " << dst_node->name().c_str() << " had set max size value!";
        continue;
      }
      SetMaxSizeListNodes(dst_node);
    }
  }
  return Status::OK();
}

Status SetVarMaxSizePass::SetConstListNodes(Node *node, std::vector<std::string> &input_names) const {
  std::vector<std::string> const_inputs;
  for (const Edge *in : node->in_edges()) {
    REQUIRES_NOT_NULL(in);
    Node *src_node = in->src();
    REQUIRES_NOT_NULL(src_node);
    bool check_const_input = false;
    for (auto it : input_names) {
      if (it == src_node->name()) {
        check_const_input = true;
      }
    }
    if (check_const_input) {
      const_inputs.push_back(src_node->name());
    } else {
      const_inputs.push_back("");
    }
  }
  node->AddAttr(ATTR_NAME_CONST_INPUT_NAME, const_inputs);
  return Status::OK();
}

Status SetVarMaxSizePass::AssignConstToVarOutNodes(const Node *node, std::vector<std::string> &input_names) const {
  // check input have const flag
  for (auto it : input_names) {
    ADP_LOG(DEBUG) << "Const input attr name:" << it << " begin to check.";
    if (it != node->name()) {
      continue;
    }
    for (const Edge *out : node->out_edges()) {
      REQUIRES_NOT_NULL(out);
      Node *dst_node = out->dst();
      REQUIRES_NOT_NULL(dst_node);
      std::string const_input;
      Status ret = GetNodeAttr(dst_node->attrs(), ATTR_NAME_CONST_INPUT_NAME, &const_input);
      if ((ret.ok()) || (!const_input.empty())) {
        ADP_LOG(INFO) << "The node : " << dst_node->name().c_str() << " had set const input.";
        continue;
      }
      SetConstListNodes(dst_node, input_names);
    }
  }
  return Status::OK();
}

Status SetVarMaxSizePass::Run(const GraphOptimizationPassOptions &options) {
  Graph *graph_in = (options.graph)->get();
  if (graph_in == nullptr || options.session_options == nullptr) {
    return Status::OK();
  }

  std::map<std::string, std::string> pass_options = NpuAttrs::GetPassOptions(options);
  std::string placeholder_name = pass_options["const_input"];
  std::vector<std::string> const_input_names;
  Split(placeholder_name, const_input_names, ",");

  for (Node *node : graph_in->op_nodes()) {
    if ((node != nullptr) && (node->type_string() == "Placeholder")) {
      AssignMaxSizeToVarOutNodes(node);
      AssignConstToVarOutNodes(node, const_input_names);
    }
  }
  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 2, SetVarMaxSizePass);
}  // namespace tensorflow
