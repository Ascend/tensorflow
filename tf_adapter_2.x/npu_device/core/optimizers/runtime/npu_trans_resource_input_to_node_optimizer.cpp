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
#include "npu_utils.h"
#include "optimizers/npu_optimizer_manager.h"

namespace {
tensorflow::Status TransResourceInput2Node(TFE_Context *context, tensorflow::Graph *graph,
                                           std::map<int, tensorflow::Node *> arg_substitutes,
                                           bool is_while_body_graph = false);

tensorflow::Status TransFunctionDef(TFE_Context *context, const std::string &func_name,
                                    const std::string &new_func_name,
                                    std::map<int, tensorflow::Node *> &node_substitutes,
                                    bool is_while_body_graph = false) {
  npu::OptimizeStageGraphDumper dumper("Function." + func_name);
  DLOG() << "Start trans function " << func_name;
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  const tensorflow::FunctionDef *fdef = lib_def->Find(func_name);
  std::unique_ptr<tensorflow::FunctionBody> fbody;
  NPU_REQUIRES_OK(FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice{}, lib_def, &fbody));

  dumper.Dump("before_trans_resource", fbody->graph->ToGraphDefDebug());
  std::map<int, tensorflow::Node *> subgraph_substitutes;
  for (auto &item : node_substitutes) {
    tensorflow::Status status;
    DLOG() << "Copy arg substitute " << item.second->name() << " for function " << func_name << " input " << item.first;
    tensorflow::Node *substitute = fbody->graph->AddNode(item.second->def(), &status);
    NPU_REQUIRES_OK(status);
    (void)subgraph_substitutes.emplace(item.first, substitute);
  }

  NPU_REQUIRES_OK(TransResourceInput2Node(context, fbody->graph, subgraph_substitutes, is_while_body_graph));
  npu::FixGraphArgRetvalIndex(fbody->graph);

  tensorflow::FunctionDef optimized_fdef;
  auto lookup = [&fdef](const tensorflow::Node *node) -> absl::optional<std::string> {
    for (const auto &control_ret : fdef->control_ret()) {
      if (control_ret.second == node->name()) {
        return absl::make_optional(node->name());
      }
    }
    return absl::nullopt;
  };

  dumper.Dump("after_trans_resource", fbody->graph->ToGraphDefDebug());
  NPU_REQUIRES_OK(tensorflow::GraphToFunctionDef(*fbody->graph, new_func_name, lookup, &optimized_fdef));
  NPU_REQUIRES_OK(lib_def->RemoveFunction(new_func_name));
  NPU_REQUIRES_OK(lib_def->AddFunctionDef(optimized_fdef));
  DLOG() << "Finish trans function " << func_name << " to " << new_func_name;
  return tensorflow::Status::OK();
}

tensorflow::Status TransWhileNode(TFE_Context *context, tensorflow::Graph *graph, tensorflow::Node *node) {
  DLOG() << "Start trans node " << node->name() << std::endl << node->DebugString();
  std::map<int, tensorflow::Node *> substitutes;
  std::map<int32_t, int32_t> pruned_index;
  for (int i = 0; i < node->num_inputs(); i++) {
    const tensorflow::Edge *edge;
    NPU_REQUIRES_OK(node->input_edge(i, &edge));
    if (npu::IsSubstituteNode(edge->src())) {
      DLOG() << "Node input " << i << " from substitute " << edge->src()->name();
      substitutes[i] = edge->src();
    } else {
      DLOG() << "Node input " << i << " shift to " << pruned_index.size();
      (void)pruned_index.emplace(edge->dst_input(), pruned_index.size());
    }
  }

  std::string cond = node->attrs().Find("cond")->func().name();
  std::string body = node->attrs().Find("body")->func().name();

  DLOG() << "Trans cond function " << cond << " of node " << node->name();
  (void)TransFunctionDef(context, cond, cond, substitutes);
  DLOG() << "Trans body function " << body << " of node " << node->name();
  (void)TransFunctionDef(context, body, body, substitutes, true);

  tensorflow::NodeDef ndef = node->def();
  auto copied_type_attr = ndef.attr().at("T");                           // Copy origin attr
  auto mutable_type_list = ndef.mutable_attr()->at("T").mutable_list();  // Clear origin attr
  mutable_type_list->clear_type();
  for (int32_t i = 0; i < copied_type_attr.list().type().size(); i++) {
    if (substitutes.find(i) == substitutes.end()) {
      DLOG() << "Node " << node->name() << " attr 'T' add " << mutable_type_list->type_size() << " "
             << tensorflow::DataTypeString(copied_type_attr.mutable_list()->type(i));
      mutable_type_list->add_type(copied_type_attr.mutable_list()->type(i));
    }
  }

  auto copied_shape_attr = ndef.attr().at("output_shapes");                           // Copy origin attr
  auto mutable_shape_list = ndef.mutable_attr()->at("output_shapes").mutable_list();  // Clear origin attr
  mutable_shape_list->clear_shape();
  for (int32_t i = 0; i < copied_shape_attr.list().type().size(); i++) {
    if (substitutes.find(i) == substitutes.end()) {
      DLOG() << "Node " << node->name() << " attr 'output_shapes' add " << mutable_shape_list->shape_size() << " "
             << copied_shape_attr.mutable_list()->shape(i).DebugString();
      *mutable_shape_list->add_shape() = copied_shape_attr.mutable_list()->shape(i);
    }
  }

  DLOG() << "Add substitute for node " << ndef.name() << std::endl << ndef.DebugString();
  tensorflow::Status status;
  auto pruned_node = graph->AddNode(ndef, &status);
  NPU_REQUIRES_OK(status);

  for (auto edge : node->in_edges()) {
    if (npu::IsSubstituteNode(edge->src())) {
      continue;
    }
    if (edge->IsControlEdge()) {
      DLOG() << "Add ctrl edge from " << edge->src()->name() << " to " << pruned_node->name();
      (void)graph->AddControlEdge(edge->src(), pruned_node);
    } else {
      auto added_edge = graph->AddEdge(edge->src(), edge->src_output(), pruned_node, pruned_index[edge->dst_input()]);
      DLOG() << "Add input edge " << added_edge->DebugString();
    }
  }

  for (auto edge : node->out_edges()) {
    if (edge->IsControlEdge()) {
      DLOG() << "Add ctrl edge from " << pruned_node->name() << " to " << edge->dst()->name();
      (void)graph->AddControlEdge(pruned_node, edge->dst());
    } else {
      if (substitutes.find(edge->src_output()) != substitutes.end()) {  // Substitute
        const tensorflow::Edge *in_edge;
        NPU_REQUIRES_OK(node->input_edge(edge->src_output(), &in_edge));
        (void)graph->AddControlEdge(pruned_node, edge->dst());
        auto added_edge = graph->AddEdge(in_edge->src(), in_edge->src_output(), edge->dst(), edge->dst_input());
        DLOG() << "Replace output edge " << edge->DebugString() << " with edge " << added_edge->DebugString()
               << " and control edge from " << pruned_node->name() << " to " << edge->dst()->name();
      } else {
        auto added_edge = graph->AddEdge(pruned_node, pruned_index[edge->src_output()], edge->dst(), edge->dst_input());
        DLOG() << "Add output edge " << added_edge->DebugString();
      }
    }
  }

  DLOG() << "Remove node " << node->name();
  graph->RemoveNode(node);
  return tensorflow::Status::OK();
}

tensorflow::Status TransHasSubgraphNode(TFE_Context *context, tensorflow::Graph *graph, tensorflow::Node *node) {
  DLOG() << "Start trans node " << node->name() << std::endl << node->DebugString();
  std::map<int, tensorflow::Node *> substitutes;
  std::map<int32_t, int32_t> pruned_index;
  const int32_t kFunctionArgIndex = node->IsPartitionedCall() ? 0 : 1;
  for (int i = 0; i < node->num_inputs(); i++) {
    const tensorflow::Edge *edge;
    NPU_REQUIRES_OK(node->input_edge(i, &edge));
    if ((i < kFunctionArgIndex) || (!npu::IsSubstituteNode(edge->src()))) {
      (void)pruned_index.emplace(edge->dst_input(), pruned_index.size());
      continue;
    }
    substitutes[i - kFunctionArgIndex] = edge->src();
  }

  std::vector<std::string> functions;
  if (node->IsIfNode()) {
    functions.emplace_back(node->attrs().Find("then_branch")->func().name());
    functions.emplace_back(node->attrs().Find("else_branch")->func().name());
  } else if (node->IsCaseNode()) {
    for (const auto &f : node->attrs().Find("branches")->list().func()) {
      functions.emplace_back(f.name());
    }
  } else {
    functions.emplace_back(node->attrs().Find("f")->func().name());
  }

  for (auto &fn : functions) {
    DLOG() << "Trans function " << fn << " of node " << node->name();
    (void)TransFunctionDef(context, fn, fn, substitutes);
  }

  tensorflow::NodeDef ndef = node->def();
  auto copied_type_attr = ndef.attr().at("Tin");                           // Copy origin attr
  auto mutable_type_list = ndef.mutable_attr()->at("Tin").mutable_list();  // Clear origin attr
  mutable_type_list->clear_type();

  for (int32_t i = 0; i < copied_type_attr.list().type().size(); i++) {
    if (substitutes.find(i) == substitutes.end()) {
      DLOG() << "Node " << node->name() << " attr 'Tin' add " << mutable_type_list->type_size() << " "
             << tensorflow::DataTypeString(copied_type_attr.mutable_list()->type(i));
      mutable_type_list->add_type(copied_type_attr.mutable_list()->type(i));
    }
  }

  DLOG() << "Add substitute for node " << ndef.name() << std::endl << ndef.DebugString();
  tensorflow::Status status;
  auto pruned_node = graph->AddNode(ndef, &status);
  NPU_REQUIRES_OK(status);

  for (auto edge : node->in_edges()) {
    if (npu::IsSubstituteNode(edge->src())) {
      continue;
    }
    if (edge->IsControlEdge()) {
      DLOG() << "Add ctrl edge from " << edge->src()->name() << " to " << pruned_node->name();
      (void)graph->AddControlEdge(edge->src(), pruned_node);
    } else {
      auto added_edge = graph->AddEdge(edge->src(), edge->src_output(), pruned_node, pruned_index[edge->dst_input()]);
      DLOG() << "Add input edge " << added_edge->DebugString();
    }
  }
  for (auto edge : node->out_edges()) {
    auto added_edge = graph->AddEdge(pruned_node, edge->src_output(), edge->dst(), edge->dst_input());
    DLOG() << "Add output edge " << added_edge->DebugString();
  }

  DLOG() << "Remove node " << node->name();
  graph->RemoveNode(node);
  return tensorflow::Status::OK();
}

tensorflow::Status TransResourceInput2Node(TFE_Context *context, tensorflow::Graph *graph,
                                           std::map<int, tensorflow::Node *> arg_substitutes,
                                           bool is_while_body_graph) {
  std::set<tensorflow::Node *> args_to_remove;
  std::map<int32_t, tensorflow::Node *> retvals;
  for (auto node : graph->op_nodes()) {
    if (node->IsRetval()) {
      retvals[node->attrs().Find("index")->i()] = node;
      continue;
    }
    if (node->IsArg()) {
      auto index = node->attrs().Find("index")->i();
      auto iter = arg_substitutes.find(index);
      if (iter != arg_substitutes.cend()) {
        for (auto edge : node->out_edges()) {
          (void)graph->AddEdge(iter->second, edge->src_output(), edge->dst(), edge->dst_input());
        }
        (void)args_to_remove.insert(node);
      }
    }
  }
  for (auto node : args_to_remove) {
    auto index = node->attrs().Find("index")->i();
    DLOG() << "Remove Arg node " << index << " " << node->name();
    graph->RemoveNode(node);
    // For while body, function input and output signature must be same
    if (is_while_body_graph) {
      DLOG() << "Remove Retval node " << index << " " << retvals[index]->name() << " as is while body graph";
      graph->RemoveNode(retvals[index]);
    }
  }

  std::set<tensorflow::Node *> nodes_has_subgraph;
  const std::function<void(tensorflow::Node *)> &enter = [&nodes_has_subgraph](tensorflow::Node *node) {
    if (npu::IsNodeHasSubgraph(node) && npu::IsNodeHasSubstituteInput(node)) {
      DLOG() << "Node " << node->name() << " with function will be pruned";
      (void)nodes_has_subgraph.insert(node);
    };
  };
  tensorflow::DFS(*graph, enter, {}, {}, {});

  for (auto node : nodes_has_subgraph) {
    if (node->IsWhileNode()) {
      NPU_REQUIRES_OK(TransWhileNode(context, graph, node));
    } else if (node->IsCaseNode() || node->IsIfNode() || node->IsPartitionedCall()) {
      NPU_REQUIRES_OK(TransHasSubgraphNode(context, graph, node));
    } else {
      LOG(INFO) << "Node " << node->name() << "has subgraph but not pruned " << node->DebugString();
    }
  }

  (void)tensorflow::FixupSourceAndSinkEdges(graph);

  return tensorflow::Status::OK();
}
}  // namespace

namespace npu {
tensorflow::Status TransResourceInput2NodeOptimize(TFE_Context *context, NpuMutableConcreteGraph *graph,
                                                   std::map<std::string, std::string> options, NpuDevice *device,
                                                   int num_inputs, TFE_TensorHandle **inputs) {
  TF_UNUSED_VARIABLE(options);
  TF_UNUSED_VARIABLE(num_inputs);
  auto mutable_graph = graph->MutableGraph();
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  const tensorflow::FunctionDef *fdef = lib_def->Find(graph->Op());

  std::map<int32_t, tensorflow::ResourceHandle> npu_resources;
  std::map<int32_t, tensorflow::ResourceHandle> cpu_resources;
  std::map<int32_t, tensorflow::ResourceHandle> mirrored_resources;

  std::map<int32_t, int64_t> bypass_outputs;
  std::map<int32_t, tensorflow::Node *> indexed_retvals;
  for (auto node : mutable_graph->op_nodes()) {
    if (!node->IsRetval()) {
      continue;
    }
    indexed_retvals[node->attrs().Find("index")->i()] = node;
  }

  for (auto item : indexed_retvals) {
    const tensorflow::Edge *edge;
    NPU_REQUIRES_OK(item.second->input_edge(0, &edge));
    if (edge->src()->IsArg()) {
      bypass_outputs[item.first] = edge->src()->attrs().Find("index")->i();
      DLOG() << "Remove output " << item.first << " ref form input " << edge->src()->attrs().Find("index")->i();
      mutable_graph->RemoveNode(item.second);
    }
  }
  PruneGraphByFunctionSignature(*fdef, mutable_graph);

  for (auto node : mutable_graph->op_nodes()) {
    if (!node->IsArg()) {
      continue;
    }
    auto index = node->attrs().Find("index")->i();

    const tensorflow::Tensor *tensor = nullptr;
    NPU_REQUIRES_OK(GetTensorHandleTensor(inputs[index], &tensor));
    if (tensor->dtype() == tensorflow::DT_RESOURCE) {
      auto &handle = tensor->flat<tensorflow::ResourceHandle>()(0);
      if (device->Mirrored(handle)) {
        for (auto edge : node->out_edges()) {
          if (edge->IsControlEdge()) {
            continue;
          }
          if (edge->dst()->IsWhileNode()) {
            edge->dst()->AddAttr("_consumed_iterators", true);
          }
        }
        (void)mirrored_resources.emplace(index, handle);
      } else if (IsNpuTensorHandle(inputs[index])) {
        (void)npu_resources.emplace(index, handle);
      } else {
        (void)cpu_resources.emplace(index, handle);
      }
      DLOG() << graph->Op() << " resource input " << index << " " << handle.maybe_type_name() << " from "
             << (device->Mirrored(handle) ? "mirrored" : (IsNpuTensorHandle(inputs[index]) ? "npu" : "cpu"));
    }
  }

  std::map<int, std::shared_ptr<npu::IteratorResourceProvider>> dependent_resources;
  if (cpu_resources.empty()) {
    npu_resources.insert(mirrored_resources.cbegin(), mirrored_resources.cend());
    for (auto resource : mirrored_resources) {
      auto &handle = resource.second;
      auto provider = device->GetIteratorProvider(context, handle);
      NPU_REQUIRES(provider != nullptr,
                   tensorflow::errors::Internal("Resource provider for ", handle.name(), " not found"));
      DLOG() << "Collect iterator provider " << handle.name();
      (void)dependent_resources.emplace(resource.first, provider);
    }
    graph->SetConsumedIterators(dependent_resources);
  } else {
    cpu_resources.insert(mirrored_resources.cbegin(), mirrored_resources.cend());
  }

  std::map<int, tensorflow::Node *> arg_substitutes;
  for (auto resource : npu_resources) {
    auto index = resource.first;
    auto &handle = resource.second;
    std::shared_ptr<ResourceGenerator> generator;
    device->GetResourceGeneratorDef(handle, &generator);
    NPU_REQUIRES(generator != nullptr, tensorflow::errors::Internal("Unknown npu resource ", handle.DebugString()));
    DLOG() << "Generator of input " << index << " " << generator->NodeDef()->name();
    tensorflow::Status status;
    tensorflow::Node *substitute = mutable_graph->AddNode(*generator->NodeDef(), &status);
    NPU_REQUIRES_OK(status);
    substitute->AddAttr("_arg_index", int(index));
    substitute->AddAttr("_is_substitute", true);
    arg_substitutes[index] = substitute;
  }

  NPU_REQUIRES_OK(TransResourceInput2Node(context, mutable_graph, arg_substitutes));

  PruneGraphByFunctionSignature(*fdef, mutable_graph);

  std::set<int32_t> consumed_inputs;
  std::set<int32_t> produced_outputs;
  for (auto node : graph->Graph()->nodes()) {
    if (node->IsArg()) {
      (void)consumed_inputs.insert(node->attrs().Find("index")->i());
    } else if (node->IsRetval()) {
      (void)produced_outputs.insert(node->attrs().Find("index")->i());
    }
  }

  FixGraphArgRetvalIndex(mutable_graph);

  graph->SetNpuResources(npu_resources);
  graph->SetCpuResources(cpu_resources);
  graph->SetConsumedInputs(consumed_inputs);
  graph->SetProducedOutputs(produced_outputs);
  graph->SetBypassOutputs(bypass_outputs);

  return tensorflow::Status::OK();
}

NPU_REGISTER_RT_OPTIMIZER(1, "TransResourceInput2GraphNodeOptimizer", TransResourceInput2NodeOptimize);
}  // namespace npu
