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

#include "npu_device.h"
#include "optimizers/npu_optimizer_manager.h"

namespace {
const static std::string kHcomBroadcast = "HcomBroadcast";
const static std::string kWeightUpdateGroupingAttr = "_weight_update_grouping";
const static std::string kReadVariableOp = "ReadVariableOp";
const static std::string kAssignOp = "AssignVariableOp";

/**
 * @brief: weight update grouping optimize
 * @param context: tfe context
 * @param graph: tensorflow graph
 * @param changed: if changed or not
 */
tensorflow::Status WeightUpdateGroupingOptimizeInner(tensorflow::FunctionLibraryDefinition *lib_def,
                                                     tensorflow::Graph *graph, bool &changed) {
  for (tensorflow::Node *node : graph->op_nodes()) {
    for (auto &attr : node->attrs()) {
      if (attr.second.has_func()) {
        std::string func_name = attr.second.func().name();
        const tensorflow::FunctionDef *fdef = lib_def->Find(func_name);
        std::unique_ptr<tensorflow::FunctionBody> fbody;
        NPU_REQUIRES_OK(FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice{}, lib_def, &fbody));
        bool optimized = false;
        NPU_REQUIRES_OK(WeightUpdateGroupingOptimizeInner(lib_def, fbody->graph, optimized));
        if (optimized) {
          tensorflow::FunctionDef optimized_fdef;
          auto lookup = [&fdef](const tensorflow::Node *node) -> absl::optional<std::string> {
            for (const auto &control_ret : fdef->control_ret()) {
              if (control_ret.second == node->name()) {
                return absl::make_optional(node->name());
              }
            }
            return absl::nullopt;
          };
          NPU_REQUIRES_OK(tensorflow::GraphToFunctionDef(*fbody->graph, func_name, lookup, &optimized_fdef));
          NPU_REQUIRES_OK(lib_def->RemoveFunction(func_name));
          NPU_REQUIRES_OK(lib_def->AddFunctionDef(optimized_fdef));
        }
      }
    }

    if (node->type_string() == kHcomBroadcast && node->attrs().Find(kWeightUpdateGroupingAttr) != nullptr) {
      std::unordered_set<const tensorflow::Edge *> edges_to_remove;
      tensorflow::Node *read_var_node = nullptr;
      for (auto in_edge : node->in_edges()) {
        if (in_edge->IsControlEdge()) {
          (void)edges_to_remove.insert(in_edge);
        } else {
          if (node->num_inputs() == 1 && in_edge->src()->type_string() == kReadVariableOp &&
              in_edge->src()->attrs().Find(kWeightUpdateGroupingAttr) != nullptr) {
            read_var_node = in_edge->src();
            (void)edges_to_remove.insert(in_edge);
          }
        }
      }
      if (read_var_node == nullptr) {
        continue;
      }

      NPU_REQUIRES((node->num_outputs() == 1),
                   tensorflow::errors::Internal("When the weight_update_grouping switch is on, there can only be one "
                                                "data edge after broadcast in the function grouping_gradients_apply"));
      tensorflow::Node *assign_node = nullptr;
      for (auto out_edge : node->out_edges()) {
        if (!out_edge->IsControlEdge()) {
          NPU_REQUIRES(
            (out_edge->dst()->type_string() == kAssignOp &&
             out_edge->dst()->attrs().Find(kWeightUpdateGroupingAttr) != nullptr),
            tensorflow::errors::Internal("When the weight_update_grouping switch is on, the operator following "
                                         "broadcast in function grouping_gradients_apply must be assign"));
          assign_node = out_edge->dst();
          break;
        }
      }

      tensorflow::Node *var_node = nullptr;
      NPU_REQUIRES_OK(assign_node->input_node(0, &var_node));
      NPU_REQUIRES((var_node != nullptr), tensorflow::errors::Internal("Get the 0 th input of assign is nullptr"));

      for (auto in_edge : assign_node->in_edges()) {
        if (in_edge->IsControlEdge()) {
          (void)edges_to_remove.insert(in_edge);
        }
      }

      tensorflow::Node *new_read_var_node = graph->CopyNode(read_var_node);
      if (new_read_var_node == nullptr) {
        return tensorflow::errors::Internal("Failed copy node from ", read_var_node->name());
      }
      new_read_var_node->set_name(read_var_node->name() + std::string("_copied"));

      for (auto edge : edges_to_remove) {
        graph->RemoveEdge(edge);
      }

      (void)graph->AddEdge(var_node, 0, new_read_var_node, 0);
      (void)graph->AddEdge(new_read_var_node, 0, node, 0);
      for (auto var_edge : var_node->out_edges()) {
        if (var_edge->dst() != new_read_var_node && var_edge->dst() != assign_node) {
          (void)graph->AddControlEdge(assign_node, var_edge->dst());
        }
      }
      changed = true;
    }
  }

  return tensorflow::Status::OK();
}
}  // namespace

namespace npu {
tensorflow::Status WeightUpdateGroupingOptimize(TFE_Context *context, tensorflow::Graph *graph,
                                                std::map<std::string, std::string> options) {
  (void)options;
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  bool unused = false;
  return WeightUpdateGroupingOptimizeInner(lib_def, graph, unused);
}

NPU_REGISTER_META_OPTIMIZER(2, "WeightUpdateGroupingOptimizer", WeightUpdateGroupingOptimize);
}  // namespace npu