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
#include "optimizers/runtime/node_placer.h"
#include "optimizers/npu_optimizer_manager.h"

namespace {
const char *const kDynamicNodeTypeGetNext = "0";
const char *const kDynamicNodeTypeData = "1";
const char *const kNodeTypeGetNext = "IteratorGetNext";
const size_t kShapeStrSize = 2;
tensorflow::Status SetShapeToOutputDesc(const std::vector<std::string> &input_shapes, const size_t idx,
                                        tensorflow::AttrValue &attr_shape_value) {
  if (input_shapes.empty()) {
    return tensorflow::errors::InvalidArgument("Input shapes are empty.");
  }
  if (idx >= input_shapes.size()) {
    return tensorflow::errors::InvalidArgument("Index ", idx, " is invalid, input shapes size: ", input_shapes.size());
  }
  DLOG() << "Get input " << idx << " shape: " << input_shapes[idx];

  // e.g. shape:["data:2,3,4"] -> ["data", "2,3,4"]
  std::vector<std::string> shape = tensorflow::str_util::Split(input_shapes[idx], ":");
  if (shape.size() != kShapeStrSize) {
    return tensorflow::errors::InvalidArgument("Invalid shape size ", shape.size(), ", except ", kShapeStrSize);
  }
  if (shape.back().empty()) {
    // scale node has no shape
    return tensorflow::Status::OK();
  }
  // e.g. dims:["2,3,4"] -> ["2", "3", "4"]
  std::vector<std::string> dims = tensorflow::str_util::Split(shape.back(), ",");
  for (const auto &dim : dims) {
    int32_t digit_dim = -1;
    if (!tensorflow::strings::safe_strto32(dim, &digit_dim)) {
      return tensorflow::errors::InvalidArgument("Invalid dim str ", dim);
    }
    attr_shape_value.mutable_list()->add_i(digit_dim);
  }
  return tensorflow::Status::OK();
}

void GetOutputDataIndex(tensorflow::Node *node, std::vector<int32_t> &ordered_indexes) {
  std::set<int32_t> out_index;
  for (const auto &out_edge : node->out_edges()) {
    if (!out_edge->IsControlEdge()) {
      DLOG() << "Node out edge info:" << out_edge->DebugString();
      out_index.insert(out_edge->src_output());
    }
  }
  ordered_indexes.clear();
  ordered_indexes.insert(ordered_indexes.end(), out_index.begin(), out_index.end());
}

tensorflow::Status BuildGetNextShape(tensorflow::Graph *graph, tensorflow::Node *node,
                                     const std::vector<int32_t> &ordered_indexes) {
  for (int32_t idx : ordered_indexes) {
    std::string shape_name = "getnext_shape_" + std::to_string(idx);
    tensorflow::Node *shape_node = nullptr;
    NPU_REQUIRES_OK(tensorflow::NodeBuilder(shape_name, "Shape")
                      .Input(node, idx)
                      .Device(node->def().device())
                      .Finalize(graph, &shape_node));
    std::string identity_name = "shape_identity_" + std::to_string(idx);
    tensorflow::Node *identity_node = nullptr;
    NPU_REQUIRES_OK(tensorflow::NodeBuilder(identity_name, "Identity")
                      .Input(shape_node, 0)
                      .Device(shape_node->def().device())
                      .Finalize(graph, &identity_node));
    npu::AssembleOpDef(shape_node);
    npu::AssembleOpDef(identity_node);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status UpdateTensorDescForDynDims(const std::vector<std::string> &all_input_shapes,
                                              const std::vector<int32_t> &ordered_indexes,
                                              tensorflow::Node *dynamic_dims_node) {
  DLOG() << "Change " << dynamic_dims_node->name() << " shape desc.";
  tensorflow::NodeDef &node_def = const_cast<tensorflow::NodeDef &>(dynamic_dims_node->def());
  tensorflow::AttrValue &out_tensor_desc = (*node_def.mutable_attr())[kOutputDesc];
  for (size_t i = 0; i < ordered_indexes.size(); ++i) {
    tensorflow::AttrValue attr_shape_value;
    attr_shape_value.set_type(tensorflow::DT_INT32);
    NPU_REQUIRES_OK(SetShapeToOutputDesc(all_input_shapes, i, attr_shape_value));
    (*out_tensor_desc.mutable_list()->mutable_func(ordered_indexes[i])->mutable_attr())[kShape] = attr_shape_value;
  }
  DLOG() << "Change input shapes desc successfully for node:" << dynamic_dims_node->name();
  return tensorflow::Status::OK();
}

tensorflow::Status TryToBuildShapeForDynDims(const std::map<std::string, std::string> &options,
                                             tensorflow::Graph *graph) {
  std::string input_shapes = (options.find(ge::INPUT_SHAPE) == options.end()) ? "" : options.at(ge::INPUT_SHAPE);
  std::string dynamic_node_type =
    (options.find(ge::DYNAMIC_NODE_TYPE) == options.end()) ? "" : options.at(ge::DYNAMIC_NODE_TYPE);
  std::string dynamic_dims = (options.find(ge::kDynamicDims) == options.end()) ? "" : options.at(ge::kDynamicDims);
  bool need_dyn_proc = (!input_shapes.empty()) && (!dynamic_node_type.empty()) && (!dynamic_dims.empty());
  if (!need_dyn_proc || dynamic_node_type == kDynamicNodeTypeData) {
    DLOG() << "Skip dynamic dims. Option configuration is complete:" << (need_dyn_proc ? "true" : "false")
           << ", dynamic_node_type(0 for GetNext, 1 for Data):" << dynamic_node_type;
    return tensorflow::Status::OK();
  }

  DLOG() << "Enable dynamic dims for graph.";
  // e.g. all_input_shapes:["data:2,3;data1:3,4"] -> ["data:2,3", "data1:3,4"]
  std::vector<std::string> all_input_shapes = tensorflow::str_util::Split(input_shapes, ";");

  std::vector<int32_t> ordered_indexes;
  for (auto node : graph->op_nodes()) {
    // add shape node to get IteratorGetNext node real shape
    if ((node->type_string() == kNodeTypeGetNext) && (dynamic_node_type == kDynamicNodeTypeGetNext)) {
      GetOutputDataIndex(node, ordered_indexes);
      if (ordered_indexes.size() != all_input_shapes.size()) {
        return tensorflow::errors::InvalidArgument(
          "Invalid input shape size, all input shape size:", all_input_shapes.size(),
          ", GetNext output size:", ordered_indexes.size());
      }
      NPU_REQUIRES_OK(BuildGetNextShape(graph, node, ordered_indexes));
      NPU_REQUIRES_OK(UpdateTensorDescForDynDims(all_input_shapes, ordered_indexes, node));
      return tensorflow::Status::OK();
    }
  }
  return tensorflow::Status::OK();
}
}  // namespace

namespace npu {
tensorflow::Status BuildNpuOpOptimize(TFE_Context *context, NpuMutableConcreteGraph *graph,
                                      std::map<std::string, std::string> options, NpuDevice *device, int num_inputs,
                                      TFE_TensorHandle **inputs) {
  TF_UNUSED_VARIABLE(options);
  TF_UNUSED_VARIABLE(num_inputs);
  TF_UNUSED_VARIABLE(inputs);
  std::stringstream ss;
  ss << device->ValidateInputTypes(graph->ConsumedTypes()).error_message();
  ss << device->ValidateOutputTypes(graph->ProducedTypes()).error_message();
  std::set<std::string> unsupported_ops;
  NPU_REQUIRES_OK(
    GetGraphUnsupportedOps(device, graph->MutableGraph(), npu::UnwrapCtx(context)->FuncLibDef(), unsupported_ops));
  if (!unsupported_ops.empty()) {
    ss << "Unsupported ops " << SetToString(unsupported_ops);
  }
  if (!ss.str().empty()) {
    tensorflow::Node *key;
    if (IsGraphNeedLoop(graph->MutableGraph(), &key) || key != nullptr) {
      graph->SetLoopType(NpuConcreteGraph::LoopType::BUILTIN_LOOP);
    }
    graph->SetExecutionType(NpuConcreteGraph::ExecutionType::MIX);
    LOG(INFO) << graph->Op() << " compiled in mix mode on npu";
    DLOG() << graph->Op() << " not fully compiled on npu as " << std::endl << ss.str();
    NPU_REQUIRES_OK(NodePlacer(context, graph->MutableGraph(), device).Apply());
    tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
    tensorflow::FunctionDefLibrary flib;
    std::string mixed_fn = "partitioned_" + graph->Op();
    NPU_REQUIRES_OK(tensorflow::GraphToFunctionDef(*graph->MutableGraph(), mixed_fn, flib.add_function()));
    NPU_REQUIRES_OK(lib_def->AddLibrary(flib));
    DLOG() << graph->Op() << " run as mix function " << mixed_fn;
    graph->SetMixedFunctionName(mixed_fn);
  } else {
    LOG(INFO) << graph->Op() << " fully compiled on npu";
    graph->SetExecutionType(NpuConcreteGraph::ExecutionType::NPU);
    NPU_REQUIRES_OK(graph->TryTransToNpuLoopGraph(context));
    AssembleParserAddons(context, graph->MutableGraph());
    NPU_REQUIRES_OK(TryToBuildShapeForDynDims(options, graph->MutableGraph()));
  }
  return tensorflow::Status::OK();
}

NPU_REGISTER_RT_OPTIMIZER(999, "BuildNpuOpOptimizer", BuildNpuOpOptimize);
}  // namespace npu
