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
#include "npu_optimizer_manager.h"

const static std::string kHcomAllReduce = "HcomAllReduce";
const static std::string kDropOutGenMaskV3 = "DropOutGenMaskV3";
const static std::string kDropOutDoMaskV3 = "DropOutDoMaskV3";
const static std::string kNpuLossScaleAttr = "_npu_loss_scale";
const static std::string kNpuGetFloatStatusOp = "NpuGetFloatStatus";

namespace {
size_t RemoveRedundantControlEdges(tensorflow::Graph *graph) {
  std::vector<tensorflow::Edge *> edges_to_remove;
  for (auto edge : graph->edges()) {
    if (edge->IsControlEdge()) {
      if ((edge->dst()->type_string() == kHcomAllReduce && edge->src()->type_string() != kNpuGetFloatStatusOp) ||
          (edge->src()->type_string() == kHcomAllReduce && edge->src()->attrs().Find(kNpuLossScaleAttr) == nullptr)) {
        edges_to_remove.push_back(edge);
      } else if (edge->src()->type_string() == kDropOutDoMaskV3 && edge->dst()->type_string() == kDropOutGenMaskV3) {
        edges_to_remove.push_back(edge);
      }
    }
  }

  if (kGraphEngineGreedyMemory) {
    for (auto node : graph->op_nodes()) {
      if (node->type_string() == kDropOutGenMaskV3) {
        bool is_first_dropout_mask = true;
        for (const auto edge : node->in_edges()) {
          if (edge->IsControlEdge() && edge->src()->type_string() == kDropOutDoMaskV3) {
            is_first_dropout_mask = false;
            break;
          }
        }
        if (is_first_dropout_mask) {
          DLOG() << "Tune control edges for dropout in graph engine greedy memory mode for saving device memory, "
                 << "start with first dropout gen mask node " << node->name();

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

          tensorflow::DFSFrom(*graph, {node}, enter, {}, {}, filter);

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
          break;
        }
      }
    }
  }

  for (auto edge : edges_to_remove) {
    DLOG() << "Remove redundant control edge " << edge->DebugString();
    graph->RemoveEdge(edge);
  }
  return edges_to_remove.size();
}
}  // namespace

namespace npu {
tensorflow::Status TransResourceInput2GraphNodeInner(TFE_Context *context, tensorflow::Graph *graph,
                                                     std::map<std::string, std::string> options, NpuDevice *device,
                                                     int num_inputs, TFE_TensorHandle **inputs) {
  (void)RemoveRedundantControlEdges(graph);

  std::map<tensorflow::Node *, tensorflow::Node *> arg_substitutes;
  for (auto node : graph->op_nodes()) {
    if (node->IsArg()) {
      auto index = node->attrs().Find("index")->i();
      if (inputs[index] == nullptr) continue;
      const tensorflow::Tensor *tensor;
      NPU_REQUIRES_OK(npu::GetTensorHandleTensor(inputs[index], &tensor));
      if (tensor->dtype() == tensorflow::DT_RESOURCE) {
        auto handle = tensor->flat<tensorflow::ResourceHandle>()(0);
        std::shared_ptr<ResourceGenerator> generator = nullptr;
        device->GetResourceGeneratorDef(handle, &generator);
        NPU_REQUIRES(generator != nullptr, tensorflow::errors::Internal("Unknown npu resource ", handle.DebugString()));
        tensorflow::Status status;
        tensorflow::Node *substitute = graph->AddNode(*generator->NodeDef(), &status);
        NPU_REQUIRES_OK(status);
        substitute->AddAttr("_arg_name", node->name());
        substitute->AddAttr("_arg_index", int(index));
        arg_substitutes[node] = substitute;
      }
    }
  }

  // 这里需要把涉及的function的resource输入也一并替换了
  std::vector<tensorflow::Node *> nodes_to_remove;
  std::vector<tensorflow::Node *> control_flow_nodes;
  for (auto node : graph->op_nodes()) {
    if (node->IsRetval() && node->input_type(0) == tensorflow::DT_RESOURCE) {
      if (kDumpExecutionDetail) {
        const tensorflow::Edge *edge;
        NPU_REQUIRES_OK(node->input_edge(0, &edge));
        LOG(INFO) << "Retval " << node->def().DebugString() << " from " << edge->src()->name() << ":"
                  << edge->src_output() << " will be removed";
      }

      nodes_to_remove.push_back(node);
      continue;
    }
    if (node->IsIfNode() || node->IsCaseNode() || node->IsWhileNode() || node->IsPartitionedCall()) {
      DLOG() << "Start pruning control flow op " << node->def().DebugString();
      std::string func_input_name = node->IsPartitionedCall() ? "args" : "input";
      bool need_trans_resource = false;
      for (auto edge : node->in_edges()) {
        if (edge->src()->IsArg() && arg_substitutes.find(edge->src()) != arg_substitutes.end()) {
          DLOG() << node->name() << " input " << edge->src()->attrs().Find("index")->i() << " is resource arg";
          need_trans_resource = true;
        }
      }
      if (!need_trans_resource) continue;

      control_flow_nodes.push_back(node);

      tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
      const tensorflow::OpRegistrationData *op_reg_data;
      NPU_REQUIRES_OK(lib_def->LookUp(node->type_string(), &op_reg_data));
      int func_input_start = 0;
      int func_input_end = 0;
      for (const auto &in_arg : op_reg_data->op_def.input_arg()) {
        func_input_start = func_input_end;
        if (in_arg.type_list_attr().empty()) {
          func_input_end++;
        } else {
          func_input_end += node->attrs().Find(in_arg.type_list_attr())->list().type_size();
        }
        DLOG() << node->name() << " input arg " << in_arg.name() << " range [" << func_input_start << ", "
               << func_input_end << ")";
        if (in_arg.name() == func_input_name) {
          break;
        }
      }

      std::vector<TFE_TensorHandle *> func_inputs;
      for (int i = func_input_start; i < func_input_end; i++) {
        const tensorflow::Edge *edge;
        NPU_REQUIRES_OK(node->input_edge(i, &edge));
        if (edge->src()->IsArg() && arg_substitutes.find(edge->src()) != arg_substitutes.end()) {
          func_inputs.push_back(inputs[edge->src()->attrs().Find("index")->i()]);
        } else {
          func_inputs.push_back(nullptr);
        }
      }

      for (auto &attr : node->attrs()) {
        if (attr.second.has_func()) {
          std::string func_name =
            node->type_string() + "_" + attr.first + "_" + attr.second.func().name() + "_" + std::to_string(node->id());
          const tensorflow::FunctionDef *fdef = lib_def->Find(attr.second.func().name());
          std::unique_ptr<tensorflow::FunctionBody> fbody;
          NPU_REQUIRES_OK(FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice{}, lib_def, &fbody));
          NPU_REQUIRES_OK(TransResourceInput2GraphNodeInner(context, fbody->graph, options, device, func_inputs.size(),
                                                            func_inputs.data()));

          // Arg节点可能会被优化掉，因而需要重新排列index
          std::vector<int> remain_indexes;
          for (auto n : fbody->graph->nodes()) {
            if (n->IsArg()) {
              remain_indexes.push_back(n->attrs().Find("index")->i());
            }
          }
          FixGraphArgRetvalIndex(fbody->graph);
          DLOG() << func_name << " remained input index [0-" << func_inputs.size() << ") -> "
                 << VecToString(remain_indexes);

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
          NPU_REQUIRES_OK(lib_def->AddFunctionDef(optimized_fdef));
          DLOG() << "Change " << node->name() << " attr " << attr.first << " func name " << attr.second.func().name()
                 << " to " << func_name;
          const_cast<tensorflow::AttrValue *>(node->attrs().Find(attr.first))->mutable_func()->set_name(func_name);
        }
      }
    }

    std::vector<const tensorflow::Edge *> edges;
    for (auto edge : node->in_edges()) {
      edges.emplace_back(edge);
    }  // You can never modify and iterator an EdgeSet
    for (auto edge : edges) {
      if (edge->src()->IsArg()) {
        auto iter = arg_substitutes.find(edge->src());
        if (iter != arg_substitutes.end()) {
          int index = edge->src()->attrs().Find("index")->i();
          graph->AddEdge(iter->second, 0, node, edge->dst_input());
          graph->RemoveEdge(edge);
        }
      }
    }
  }

  for (auto node : control_flow_nodes) {
    if (node->IsWhileNode() || node->IsIfNode() || node->IsCaseNode() || node->IsPartitionedCall()) {
      tensorflow::NodeDef ndef = node->def();
      if (node->IsWhileNode()) {
        int removed_nums = 0;
        for (int i = 0; i < node->num_inputs(); i++) {
          if (node->input_type(i) == tensorflow::DT_RESOURCE) {
            int index = i - removed_nums;
            removed_nums++;

            ndef.mutable_input()->erase(ndef.mutable_input()->begin() + index);

            auto type = ndef.mutable_attr()->at("T").mutable_list()->mutable_type();
            type->erase(type->begin() + index);

            auto shape = ndef.mutable_attr()->at("output_shapes").mutable_list()->mutable_shape();
            shape->erase(shape->begin() + index);
          }
        }
      } else if (node->IsIfNode() || node->IsCaseNode() || node->IsPartitionedCall()) {
        int removed_nums = 0;
        int arg_start_index = node->IsPartitionedCall() ? 0 : 1;
        for (int i = arg_start_index; i < node->num_inputs(); i++) {
          if (node->input_type(i) == tensorflow::DT_RESOURCE) {
            int index = i - removed_nums;
            removed_nums++;

            ndef.mutable_input()->erase(ndef.mutable_input()->begin() + index);

            auto type = ndef.mutable_attr()->at("Tin").mutable_list()->mutable_type();
            type->erase(type->begin() + index - arg_start_index);
          }
        }
      }
      DLOG() << "Pruned control flow op " << ndef.DebugString();
      tensorflow::Status status;
      auto pruned_node = graph->AddNode(ndef, &status);
      NPU_REQUIRES_OK(status);
      int pruned_input_index = 0;
      for (auto edge : node->in_edges()) {
        if (edge->IsControlEdge()) {
          graph->AddControlEdge(edge->src(), pruned_node);
          DLOG() << "Add ctrl edge from " << edge->src()->name() << " to " << pruned_node->name();
        }
      }
      for (int i = 0; i < node->num_inputs(); i++) {
        const tensorflow::Edge *edge;
        NPU_REQUIRES_OK(node->input_edge(i, &edge));
        if (node->input_type(i) != tensorflow::DT_RESOURCE) {
          graph->AddEdge(edge->src(), edge->src_output(), pruned_node, pruned_input_index++);
          DLOG() << "Add edge from " << edge->src()->name() << ":" << edge->src_output() << " to "
                 << pruned_node->name() << ":" << (pruned_input_index - 1);
        }
      }
      for (auto edge : node->out_edges()) {
        graph->AddEdge(pruned_node, edge->src_output(), edge->dst(), edge->dst_input());
        DLOG() << "Add edge from " << pruned_node->name() << ":" << edge->src_output() << " to " << edge->dst()->name()
               << ":" << edge->dst_input();
      }
      graph->RemoveNode(node);
    }
  }
  for (auto node : nodes_to_remove) {
    graph->RemoveNode(node);
  }
  for (auto arg_substitute : arg_substitutes) {
    graph->RemoveNode(arg_substitute.first);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status TransResourceInput2GraphNode(TFE_Context *context, NpuMutableConcreteGraph *graph,
                                                std::map<std::string, std::string> options, NpuDevice *device,
                                                int num_inputs, TFE_TensorHandle **inputs) {
  auto mutable_graph = graph->MutableGraph();

  std::map<int, std::shared_ptr<npu::IteratorResourceProvider>> dependent_resources;
  for (auto node : mutable_graph->op_nodes()) {
    if (!node->IsArg()) continue;
    auto index = node->attrs().Find("index")->i();
    if (inputs[index] == nullptr) continue;
    const tensorflow::Tensor *tensor;
    NPU_REQUIRES_OK(npu::GetTensorHandleTensor(inputs[index], &tensor));
    if (tensor->dtype() != tensorflow::DT_RESOURCE) continue;
    auto &handle = tensor->flat<tensorflow::ResourceHandle>()(0);
    if (!device->MirroredIterator(handle)) continue;
    for (auto edge : node->out_edges()) {
      if (edge->IsControlEdge()) continue;
      if ((!edge->dst()->IsOp()) || edge->dst()->IsRetval()) continue;
      auto provider = device->GetIteratorProvider(context, handle);
      NPU_REQUIRES(provider != nullptr,
                   tensorflow::errors::Internal("Resource provider for ", handle.name(), " not found"));
      DLOG() << "Collect iterator provider " << handle.name();
      dependent_resources.emplace(index, provider);
      break;
    }
  }

  graph->SetDependentHostResources(dependent_resources);

  NPU_REQUIRES_OK(TransResourceInput2GraphNodeInner(context, mutable_graph, options, device, num_inputs, inputs));

  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  const tensorflow::FunctionDef *fdef = lib_def->Find(graph->Op());

  PruneGraphByFunctionSignature(*fdef, mutable_graph);

  std::vector<int> remain_indexes;
  for (auto node : graph->Graph()->nodes()) {
    if (!node->IsArg()) continue;
    remain_indexes.push_back(node->attrs().Find("index")->i());
  }
  DLOG() << graph->Op() << " remained input index [0-" << (num_inputs - 1) << "] -> " << VecToString(remain_indexes);

  FixGraphArgRetvalIndex(mutable_graph);
  graph->SetPruneInputsFunc(
    [remain_indexes](int num_inputs, TFE_TensorHandle **inputs, std::vector<TFE_TensorHandle *> &pruned) {
      TF_UNUSED_VARIABLE(num_inputs);
      for (auto index : remain_indexes) {
        pruned.push_back(inputs[index]);
      }
    });
  return tensorflow::Status::OK();
}

NPU_REGISTER_RT_OPTIMIZER(1, "TransResourceInput2GraphNodeOptimizer", TransResourceInput2GraphNode);
}  // namespace npu
