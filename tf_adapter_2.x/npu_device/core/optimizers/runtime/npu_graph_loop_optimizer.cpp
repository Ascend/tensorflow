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
#include "tensorflow/core/grappler/op_types.h"

#include "npu_device.h"
#include "optimizers/npu_algorithm.h"
#include "optimizers/npu_optimizer_manager.h"

namespace {
bool IsGraphNeedLoop(const tensorflow::Graph *graph, tensorflow::Node **key) {
  *key = nullptr;
  for (auto node : graph->op_nodes()) {
    if (node->IsWhileNode()) {
      if (*key != nullptr) {
        DLOG() << "Skip check as multi while nodes in graph first " << (*key)->name() << " another " << node->name();
        *key = nullptr;
        return false;
      }
      *key = node;
    }
  }
  if (*key == nullptr) {
    DLOG() << "Skip check as no while node in graph";
    return false;
  }
  size_t reserved_nums = 0;
  const std::function<void(const tensorflow::Node *)> &enter = [&reserved_nums](const tensorflow::Node *node) {
    if (node->IsOp()) {
      reserved_nums++;
    }
  };
  tensorflow::ReverseDFSFrom(*graph, {*key}, enter, {}, {}, {});
  DLOG() << "Reserved nodes " << reserved_nums << " vs. totally " << graph->num_op_nodes();
  return static_cast<int>(reserved_nums) == graph->num_op_nodes();
}
}  // namespace

namespace npu {
/**
 * @brief: get auto loop graph
 * @param context: tfe context
 * @param origin_graph: tensorflow graph
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle
 * @param loop: is loop or not
 * @param def: tensorflow graph def
 */
tensorflow::Status GetAutoLoopGraph(TFE_Context *context, NpuMutableConcreteGraph *concrete_graph, int num_inputs,
                                    TFE_TensorHandle **inputs) {
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  std::unique_ptr<tensorflow::Graph> graph = std::make_unique<tensorflow::Graph>(lib_def);
  CopyGraph(*concrete_graph->Graph(), graph.get());

  tensorflow::Node *key;
  if (!IsGraphNeedLoop(graph.get(), &key)) {
    concrete_graph->SetBuiltinLoop(key != nullptr);
    concrete_graph->SetNeedLoop(false);
    return tensorflow::Status::OK();
  }

  concrete_graph->SetBuiltinLoop(false);
  concrete_graph->SetNeedLoop(true);

  const auto fn_name = key->attrs().Find("body")->func().name();
  DLOG() << "Inline while body func " << fn_name << " for node " << key->name();
  auto builder = tensorflow::NodeBuilder(fn_name, fn_name, lib_def);
  for (int i = 0; i < key->num_inputs(); i++) {
    const tensorflow::Edge *edge;
    NPU_REQUIRES_OK(key->input_edge(i, &edge));
    builder.Input(edge->src(), edge->src_output());
  }
  for (auto edge : key->in_edges()) {
    if (edge->IsControlEdge()) {
      builder.ControlInput(edge->src());
    }
  }

  tensorflow::Node *fn_node;
  NPU_REQUIRES_OK(builder.Finalize(graph.get(), &fn_node));

  graph->RemoveNode(key);
  tensorflow::FixupSourceAndSinkEdges(graph.get());

  tensorflow::ProcessFunctionLibraryRuntime *pflr = npu::UnwrapCtx(context)->pflr();
  tensorflow::FunctionLibraryRuntime *flr = pflr->GetFLR("/job:localhost/replica:0/task:0/device:CPU:0");

  tensorflow::OptimizeGraph(flr, &graph);

  for (auto node : graph->op_nodes()) {
    if (tensorflow::grappler::IsVariable(node->def())) {
      if (node->attrs().Find("shared_name") != nullptr) {
        DLOG() << "Change node " << node->name() << " name to " << node->attrs().Find("shared_name")->s();
        node->set_name(node->attrs().Find("shared_name")->s());
      }
    }
  }

  MarkGraphNodeInOutDesc(context, graph.get(), num_inputs, inputs);
  concrete_graph->SetGraph(std::move(graph));
  return tensorflow::Status::OK();
}

tensorflow::Status GraphLoopOptimize(TFE_Context *context, NpuMutableConcreteGraph *graph,
                                     std::map<std::string, std::string> options, NpuDevice *device, int num_inputs,
                                     TFE_TensorHandle **inputs) {
  const char *op_name = graph->Op().c_str();

  std::vector<TFE_TensorHandle *> pruned_inputs;
  graph->PruneInputs(num_inputs, inputs, pruned_inputs);
  TF_RETURN_IF_ERROR(device->ValidateInput(op_name, pruned_inputs.size(), pruned_inputs.data()));
  if (!graph->DependentHostResources().empty()) {
    NPU_REQUIRES_OK(GetAutoLoopGraph(context, graph, pruned_inputs.size(), pruned_inputs.data()));
  }
  return tensorflow::Status::OK();
}

NPU_REGISTER_RT_OPTIMIZER(999, "GraphLoopOptimizer", GraphLoopOptimize);
}  // namespace npu
