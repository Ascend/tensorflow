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

#include "tensorflow/core/graph/algorithm.h"
#include "npu_device.h"
#include "optimizers/npu_optimizer_manager.h"

const static std::string kHcomAllReduce = "HcomAllReduce";
const static std::string kDropOutGenMaskV3 = "DropOutGenMaskV3";
const static std::string kDropOutDoMaskV3 = "DropOutDoMaskV3";
const static std::string kNpuLossScaleAttr = "_npu_loss_scale";
const static std::string kNpuGetFloatStatusOp = "NpuGetFloatStatus";

namespace {
bool IsFirstDropoutNode(tensorflow::Node *node) {
  if (node->type_string() != kDropOutGenMaskV3) { return false; }
  for (const auto edge : node->in_edges()) {
    if (edge->IsControlEdge() && edge->src()->type_string() == kDropOutDoMaskV3) {
      return false;
    }
  }
  return true;
}

void FineTuneDropoutControlEdge(tensorflow::Graph *graph, tensorflow::Node *first_gen_mask) {
  std::vector<tensorflow::Node *> dropout_gen_masks;
  std::vector<tensorflow::Node *> dropout_do_masks;
  std::set<tensorflow::Node *> seen_masks;

  const std::function<void(tensorflow::Node *)> &enter = [&dropout_gen_masks, &dropout_do_masks,
                                                          &seen_masks](tensorflow::Node *node) {
    if (node->type_string() == kDropOutGenMaskV3) {
      for (auto edge : node->in_edges()) {
        if (edge->IsControlEdge() && edge->src()->type_string() == kDropOutDoMaskV3) {
          if (seen_masks.insert(edge->src()).second) {
            dropout_do_masks.push_back(edge->src());
          }
        }
      }
      if (seen_masks.insert(node).second) {
        dropout_gen_masks.push_back(node);
      }
    }
  };

  tensorflow::EdgeFilter filter = [](const tensorflow::Edge &edge) {
    return edge.dst()->type_string() == kDropOutDoMaskV3 || edge.dst()->type_string() == kDropOutGenMaskV3;
  };

  tensorflow::DFSFrom(*graph, {first_gen_mask}, enter, {}, {}, filter);

  size_t total_size = dropout_gen_masks.size();
  if (dropout_do_masks.size() < total_size) {
    total_size = dropout_do_masks.size();
  }
  DLOG() << "Total dropout gen mask " << dropout_gen_masks.size() << " do mask " << dropout_do_masks.size();

  const static size_t kDropoutCtrlDistance = 3;
  size_t start = 0;
  size_t end = kDropoutCtrlDistance;
  while (end < total_size) {
    auto &do_mask_node = dropout_do_masks[start++];
    auto &gen_mask_node = dropout_gen_masks[end++];
    auto edge = graph->AddControlEdge(do_mask_node, gen_mask_node);
    if (edge != nullptr) {
      DLOG() << "Add control edge [D->G] " << edge->DebugString();
    } else {
      DLOG() << "Existed control edge [D->G] " << do_mask_node->name() << " -> " << gen_mask_node->name();
    }
  }
}

bool IsEdgeRedundant(const tensorflow::Edge *edge) {
  if (!edge->IsControlEdge()) { return false; }
  const std::string &src = edge->src()->type_string();
  const std::string &dst = edge->dst()->type_string();
  if ((dst == kHcomAllReduce && src != kNpuGetFloatStatusOp) ||
      (src == kHcomAllReduce && edge->src()->attrs().Find(kNpuLossScaleAttr) == nullptr)) {
    return true;
  } else if (src == kDropOutDoMaskV3 && dst == kDropOutGenMaskV3) {
    return true;
  } else if (edge->src()->IsArg() && edge->dst()->IsOp()) {
    return true;
  }
  return false;
}

tensorflow::Status ControlEdgeOptimizeInner(TFE_Context *context, tensorflow::Graph *graph, bool &optimized) {
  std::vector<tensorflow::Edge *> edges_to_remove;
  for (auto edge : graph->edges()) {
    if (IsEdgeRedundant(edge)) {
      edges_to_remove.push_back(edge);
    }
  }

  bool any_subgraph_optimized = false;
  bool fine_tune_dropout = kGraphEngineGreedyMemory;
  for (auto node : graph->op_nodes()) {
    if (fine_tune_dropout && IsFirstDropoutNode(node)) {
      DLOG() << "Tune control edges for dropout in graph engine greedy memory mode for saving device memory, "
             << "start with first dropout gen mask node " << node->name();
      FineTuneDropoutControlEdge(graph, node);
      fine_tune_dropout = false;
    }

    for (auto &attr : node->attrs()) {
      if (attr.second.has_func()) {
        tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
        std::string func_name = attr.second.func().name();
        const tensorflow::FunctionDef *fdef = lib_def->Find(func_name);
        std::unique_ptr<tensorflow::FunctionBody> fbody;
        NPU_REQUIRES_OK(FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice{}, lib_def, &fbody));

        bool function_optimized = false;
        NPU_REQUIRES_OK(ControlEdgeOptimizeInner(context, fbody->graph, function_optimized));
        if (!function_optimized) { continue; }

        any_subgraph_optimized = true;
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
        DLOG() << "Control edge optimize for function " << func_name << " of node " << node->name();
        NPU_REQUIRES_OK(lib_def->RemoveFunction(func_name));
        NPU_REQUIRES_OK(lib_def->AddFunctionDef(optimized_fdef));
      }
    }
  }

  for (auto edge : edges_to_remove) {
    DLOG() << "Remove redundant control edge " << edge->DebugString();
    graph->RemoveEdge(edge);
  }
  optimized = (any_subgraph_optimized || (!edges_to_remove.empty()));
  return tensorflow::Status::OK();
}
}  // namespace

namespace npu {
tensorflow::Status ControlEdgeOptimize(TFE_Context *context, tensorflow::Graph *graph,
                                       std::map<std::string, std::string> options) {
  TF_UNUSED_VARIABLE(options);
  bool unused = false;
  return ControlEdgeOptimizeInner(context, graph, unused);
}

NPU_REGISTER_META_OPTIMIZER(999, "ControlEdgeOptimizer", ControlEdgeOptimize);
}  // namespace npu
