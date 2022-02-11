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

namespace {
bool IsControlFlowNode(tensorflow::Node *node) {
  return (node->IsIfNode() || node->IsCaseNode() || node->IsWhileNode() || node->IsPartitionedCall());
}

bool IsSubstituteNode(tensorflow::Node *node) {
  auto attr = node->attrs().Find("_is_substitute");
  return (attr != nullptr) && attr->b();
}

bool IsNodeHasSubstituteInput(tensorflow::Node *node) {
  for (auto in_node : node->in_nodes()) {
    if (IsSubstituteNode(in_node)) {
      return true;
    }
  }
  return false;
}

tensorflow::Status ConvertAndReplaceWhileNode(tensorflow::Graph *graph, tensorflow::Node *node) {
  tensorflow::NodeDef ndef = node->def();
  std::vector<int32_t> removed_inputs_num(node->num_inputs());
  std::vector<int32_t> removed_outputs_num(node->num_outputs());

  int removed_nums = 0;
  for (int i = 0; i < node->num_inputs(); i++) {
    const tensorflow::Edge *edge;
    NPU_REQUIRES_OK(node->input_edge(i, &edge));
    if (IsSubstituteNode(edge->src())) {
      int index = i - removed_nums;
      removed_nums++;

      ndef.mutable_input()->erase(ndef.mutable_input()->begin() + index);

      auto type = ndef.mutable_attr()->at("T").mutable_list()->mutable_type();
      type->erase(type->begin() + index);

      auto shape = ndef.mutable_attr()->at("output_shapes").mutable_list()->mutable_shape();
      shape->erase(shape->begin() + index);
    }
    removed_inputs_num[i] = removed_nums;
    removed_outputs_num[i] = removed_nums;
  }
  DLOG() << "Pruned control flow op " << ndef.DebugString();
  tensorflow::Status status;
  auto pruned_node = graph->AddNode(ndef, &status);
  NPU_REQUIRES_OK(status);

  for (auto edge : node->in_edges()) {
    if (IsSubstituteNode(edge->src())) continue;
    if (edge->IsControlEdge()) {
      graph->AddControlEdge(edge->src(), pruned_node);
      DLOG() << "Add ctrl edge from " << edge->src()->name() << " to " << pruned_node->name();
    } else {
      auto dst_idx = edge->dst_input() - removed_inputs_num[edge->dst_input()];
      graph->AddEdge(edge->src(), edge->src_output(), pruned_node, dst_idx);
      DLOG() << "Add edge from " << edge->src()->name() << ":" << edge->src_output() << " to " << pruned_node->name()
             << ":" << dst_idx;
    }
  }
  for (auto edge : node->out_edges()) {
    if (edge->IsControlEdge()) {
      graph->AddControlEdge(edge->src(), pruned_node);
      DLOG() << "Add ctrl edge from " << edge->src()->name() << " to " << pruned_node->name();
    } else {
      const tensorflow::Edge *in_edge;
      NPU_REQUIRES_OK(node->input_edge(edge->src_output(), &in_edge));
      if (!IsSubstituteNode(in_edge->src())) {
        auto src_idx = edge->src_output() - removed_outputs_num[edge->src_output()];
        graph->AddEdge(pruned_node, src_idx, edge->dst(), edge->dst_input());
        DLOG() << "Add edge from " << pruned_node->name() << ":" << src_idx << " to " << edge->dst()->name() << ":"
               << edge->dst_input();
      } else {
        graph->AddEdge(in_edge->src(), in_edge->src_output(), edge->dst(), edge->dst_input());
        graph->AddControlEdge(pruned_node, edge->dst());
      }
    }
  }
  graph->RemoveNode(node);
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertAndReplaceControlFlowNode(tensorflow::Graph *graph, tensorflow::Node *node) {
  if (node->IsWhileNode()) {
    return ConvertAndReplaceWhileNode(graph, node);
  } else {
    tensorflow::NodeDef ndef = node->def();
    std::vector<int32_t> removed_inputs_num(node->num_inputs());
    std::vector<int32_t> removed_outputs_num(node->num_outputs());
    int removed_nums = 0;
    int arg_start_index = node->IsPartitionedCall() ? 0 : 1;
    for (int i = arg_start_index; i < node->num_inputs(); i++) {
      const tensorflow::Edge *edge;
      NPU_REQUIRES_OK(node->input_edge(i, &edge));
      if (IsSubstituteNode(edge->src())) {
        int index = i - removed_nums;
        removed_nums++;

        ndef.mutable_input()->erase(ndef.mutable_input()->begin() + index);

        auto type = ndef.mutable_attr()->at("Tin").mutable_list()->mutable_type();
        type->erase(type->begin() + index - arg_start_index);
      }
      removed_inputs_num[i] = removed_nums;
      removed_outputs_num[i] = 0;
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
        DLOG() << "Add edge from " << edge->src()->name() << ":" << edge->src_output() << " to " << pruned_node->name()
               << ":" << (pruned_input_index - 1);
      }
    }
    for (auto edge : node->out_edges()) {
      graph->AddEdge(pruned_node, edge->src_output(), edge->dst(), edge->dst_input());
      DLOG() << "Add edge from " << pruned_node->name() << ":" << edge->src_output() << " to " << edge->dst()->name()
             << ":" << edge->dst_input();
    }
    graph->RemoveNode(node);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status TransResourceInput2Node(TFE_Context *context, tensorflow::Graph *graph,
                                           std::map<int, tensorflow::Node *> arg_substitutes);

tensorflow::Status ConvertControlFlowNodeSubgraph(TFE_Context *context, tensorflow::Node *node) {
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  const tensorflow::OpRegistrationData *op_reg_data;
  NPU_REQUIRES_OK(lib_def->LookUp(node->type_string(), &op_reg_data));
  std::string func_input_name = node->IsPartitionedCall() ? "args" : "input";
  int func_arg_start = 0;
  int func_arg_end = 0;
  for (const auto &in_arg : op_reg_data->op_def.input_arg()) {
    func_arg_start = func_arg_end;
    if (in_arg.type_list_attr().empty()) {
      func_arg_end++;
    } else {
      func_arg_end += node->attrs().Find(in_arg.type_list_attr())->list().type_size();
    }
    DLOG() << node->name() << " input " << in_arg.name() << " range [" << func_arg_start << ", " << func_arg_end << ")";
    if (in_arg.name() == func_input_name) {
      break;
    }
  }

  std::map<int, tensorflow::Node *> arg_substitutes;
  for (int i = func_arg_start; i < func_arg_end; i++) {
    const tensorflow::Edge *edge;
    NPU_REQUIRES_OK(node->input_edge(i, &edge));
    arg_substitutes[i - func_arg_start] = nullptr;
    if (IsSubstituteNode(edge->src())) {
      arg_substitutes[i - func_arg_start] = edge->src();
    }
  }

  for (auto &attr : node->attrs()) {
    if (!attr.second.has_func()) continue;
    const std::string &func_name = attr.second.func().name();

    npu::OptimizeStageGraphDumper dumper(tensorflow::strings::StrCat(node->name(), ".", attr.first));

    DLOG() << "Start prune " << node->name() << " attr " << attr.first << " subgraph " << func_name;
    const tensorflow::FunctionDef *fdef = lib_def->Find(func_name);
    std::unique_ptr<tensorflow::FunctionBody> fbody;
    NPU_REQUIRES_OK(FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice{}, lib_def, &fbody));

    dumper.DumpWithSubGraphs("before_prune", fbody->graph->ToGraphDefDebug(), lib_def);

    NPU_REQUIRES_OK(TransResourceInput2Node(context, fbody->graph, arg_substitutes));
    npu::FixGraphArgRetvalIndex(fbody->graph);

    dumper.DumpWithSubGraphs("after_prune", fbody->graph->ToGraphDefDebug(), lib_def);

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
    DLOG() << "Pruned " << node->name() << " attr " << attr.first << " subgraph " << func_name;
  }
  return tensorflow::Status::OK();
}

tensorflow::Status TransResourceInput2Node(TFE_Context *context, tensorflow::Graph *graph,
                                           std::map<int, tensorflow::Node *> arg_substitutes) {
  std::set<tensorflow::Node *> control_flow_nodes;
  std::set<tensorflow::Node *> nodes_to_remove;
  for (auto node : graph->op_nodes()) {
    if (IsControlFlowNode(node) && IsNodeHasSubstituteInput(node)) {
      DLOG() << "Control flow node " << node->name() << " will be pruned";
      control_flow_nodes.insert(node);
      continue;
    };
    if (!node->IsArg()) continue;
    auto index = node->attrs().Find("index")->i();
    auto &substitute = arg_substitutes[index];
    if (substitute == nullptr) continue;
    for (auto edge : node->out_edges()) {
      graph->AddEdge(substitute, edge->src_output(), edge->dst(), edge->dst_input());
      if ((!edge->IsControlEdge()) && edge->dst()->IsRetval()) {
        nodes_to_remove.insert(edge->dst());
      }
    }
    nodes_to_remove.insert(node);
  }
  for (auto node : nodes_to_remove) {
    graph->RemoveNode(node);
  }

  for (auto node : control_flow_nodes) {
    NPU_REQUIRES_OK(ConvertControlFlowNodeSubgraph(context, node));
    NPU_REQUIRES_OK(ConvertAndReplaceControlFlowNode(graph, node));
  }

  (void)tensorflow::FixupSourceAndSinkEdges(graph);

  return tensorflow::Status::OK();
}
}  // namespace

namespace npu {
tensorflow::Status TransResourceInput2NodeOptimize(TFE_Context *context, NpuMutableConcreteGraph *graph,
                                                   std::map<std::string, std::string> options, NpuDevice *device,
                                                   int num_inputs, TFE_TensorHandle **inputs) {
  auto mutable_graph = graph->MutableGraph();

  std::map<int, std::shared_ptr<npu::IteratorResourceProvider>> dependent_resources;
  std::map<int, std::pair<tensorflow::Node *, std::shared_ptr<ResourceGenerator>>> arg_generators;
  for (auto node : mutable_graph->op_nodes()) {
    if (!node->IsArg()) continue;
    auto index = node->attrs().Find("index")->i();
    NPU_REQUIRES(inputs[index] != nullptr, tensorflow::errors::Internal("Input ", index, " is nullptr"));
    const tensorflow::Tensor *tensor;
    NPU_REQUIRES_OK(npu::GetTensorHandleTensor(inputs[index], &tensor));

    if (tensor->dtype() == tensorflow::DT_RESOURCE) {
      auto handle = tensor->flat<tensorflow::ResourceHandle>()(0);
      device->GetResourceGeneratorDef(handle, &arg_generators[index].second);
      NPU_REQUIRES(arg_generators[index].second != nullptr,
                   tensorflow::errors::Internal("Unknown npu resource ", handle.DebugString()));

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
    } else {
      arg_generators[index].second = nullptr;
    }
  }

  std::map<int, tensorflow::Node *> arg_substitutes;
  for (auto &item : arg_generators) {
    auto &index = item.first;
    auto &arg = item.second.first;
    auto &generator = item.second.second;
    tensorflow::Status status;
    tensorflow::Node *substitute = mutable_graph->AddNode(*generator->NodeDef(), &status);
    NPU_REQUIRES_OK(status);
    substitute->AddAttr("_arg_name", arg->name());
    substitute->AddAttr("_arg_index", int(index));
    substitute->AddAttr("_is_substitute", true);
    arg_substitutes[index] = substitute;
  }

  graph->SetDependentHostResources(dependent_resources);

  NPU_REQUIRES_OK(TransResourceInput2Node(context, mutable_graph, arg_substitutes));

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

NPU_REGISTER_RT_OPTIMIZER(1, "TransResourceInput2GraphNodeOptimizer", TransResourceInput2NodeOptimize);
}  // namespace npu
