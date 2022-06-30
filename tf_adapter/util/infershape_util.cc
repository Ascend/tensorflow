/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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

#include "tf_adapter/util/infershape_util.h"
#include <sys/time.h>
#include "tensorflow/core/framework/node_def_util.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/npu_ops_identifier.h"

namespace tensorflow {
struct EdgeInfo {
  EdgeInfo(Node *src, Node *dst, int src_output, int dst_input)
      : src_(src), dst_(dst), src_output_(src_output), dst_input_(dst_input) {}

  Node *src_;
  Node *dst_;
  int src_output_;
  int dst_input_;
};

int64 InferShapeUtil::GetCurrentTimestap() {
  static const long kPerSecHasUsec = 1000000;
  struct timeval tv;
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    ADP_LOG(ERROR) << "Func gettimeofday may failed, ret:" << ret;
    LOG(ERROR) << "Func gettimeofday may failed, ret:" << ret;
    return 0;
  }
  int64 totalUsec = tv.tv_usec + tv.tv_sec * kPerSecHasUsec;
  return totalUsec;
}

Status InferShapeUtil::setArgShapeFromTensorShape(const std::vector<Tensor> vecTensor, const Graph *graph,
                                                  const OpDef &sig, ShapeRefiner &shapeRef) {
  REQUIRES_NOT_NULL(graph);
  size_t idx = 0UL;
  for (const OpDef::ArgDef &arg_def : sig.input_arg()) {
    for (Node *pNode : graph->nodes()) {
      REQUIRES_NOT_NULL(pNode);
      if (pNode->name() == arg_def.name()) {
        TF_RETURN_IF_ERROR(shapeRef.AddNode(pNode));  // here the arg node must add succ
        tensorflow::shape_inference::InferenceContext *pCxt = shapeRef.GetContext(pNode);
        if (pCxt == nullptr)  // this is a protect
        {
          return errors::Internal("The InferenceContext of node ", pNode->name(), " is null, add node failed.");
        }

        tensorflow::shape_inference::ShapeHandle shapeHandle;
        (void)pCxt->MakeShapeFromTensorShape(vecTensor[idx].shape(), &shapeHandle);
        pCxt->set_output(0, shapeHandle);  // this arg has only one output
        idx++;
        break;  // next arg
      }
    }
  }

  return Status::OK();
}

Status InferShapeUtil::GetSubGraphFromFunctionDef(const FunctionLibraryDefinition &flib_def,
                                                  const FunctionDef &func_def, Graph *graph) {
  ADP_LOG(INFO) << "The signature name of FunctionDef is " << func_def.signature().name() << ".";
  InstantiationResult result;
  AttrSlice attrs(&func_def.attr());
  TF_RETURN_IF_ERROR(InstantiateFunction(
      func_def, attrs,
      [&flib_def](const string &op, const OpDef **sig) {
        Status s = OpRegistry::Global()->LookUpOpDef(op, sig);
        if (!s.ok()) {
          return flib_def.LookUpOpDef(op, sig);
        }
        return s;
      },
      &result));

  ADP_LOG(INFO) << "InstantiateFunction " << func_def.signature().name() << " success.";
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = false;
  TF_RETURN_IF_ERROR(ConvertNodeDefsToGraph(opts, result.nodes, graph));
  ADP_LOG(INFO) << "ConvertNodeDefsToGraph " << func_def.signature().name() << " success.";
  return Status::OK();
}

bool InferShapeUtil::IsInitializedGraph(const Node *node) {
  Node *logical_not_node = nullptr;
  (void)node->input_node(0, &logical_not_node);
  if (logical_not_node == nullptr) {
    return false;
  }

  if (logical_not_node->type_string() == "Reshape") {
    Node *reshape_node = logical_not_node;
    (void)reshape_node->input_node(0, &logical_not_node);
    if (logical_not_node == nullptr) {
      return false;
    }
  }
  if (logical_not_node->type_string() != "LogicalNot") {
    return false;
  }

  Node *stack_node = nullptr;
  (void)logical_not_node->input_node(0, &stack_node);
  if (stack_node == nullptr || stack_node->type_string() != "Pack") {
    return false;
  }

  Node *is_var_init_node = nullptr;
  (void)stack_node->input_node(0, &is_var_init_node);
  if (is_var_init_node == nullptr) {
    return false;
  }

  if (is_var_init_node->type_string() == "VarIsInitializedOp" ||
      is_var_init_node->type_string() == "IsVariableInitialized") {
    ADP_LOG(INFO) << "GEOP::IsInitializedGraph";
    return true;
  }

  return false;
}

Status InferShapeUtil::getInputShapesOfNode(const ShapeRefiner &shapeRef, const Node *pNode,
                                            std::vector<tensorflow::shape_inference::ShapeHandle> &inputShapeVec) {
  REQUIRES_NOT_NULL(pNode);
  for (const Edge *pEdge : pNode->in_edges()) {
    REQUIRES_NOT_NULL(pEdge);
    if (pEdge->IsControlEdge()) {
      continue;
    }

    Node *pNodeIn = pEdge->src();
    tensorflow::shape_inference::InferenceContext *pCxtIn = shapeRef.GetContext(pNodeIn);
    if (pCxtIn == nullptr) {
      return errors::Internal("Can't get context of the input ", pNodeIn->name(), " of the node ", pNode->name(), ".");
    }

    int32_t iDstInput = pEdge->dst_input();
    if (iDstInput < 0) {
      return errors::Internal("iDstInput is less than zero");
    }
    inputShapeVec[iDstInput] = pCxtIn->output(pEdge->src_output());
  }

  return Status::OK();
}

void InferShapeUtil::setShapeOfEnterOP(const ShapeRefiner &shapeRef, const Node *pNode) {
  CHECK_NOT_NULL(pNode);
  tensorflow::shape_inference::InferenceContext *pCxt = shapeRef.GetContext(pNode);
  CHECK_NOT_NULL(pCxt);
  tensorflow::shape_inference::ShapeHandle shapeOutOne = pCxt->output(0);  // Enter has only one output
  if (pCxt->DebugString(shapeOutOne).find('?') == std::string::npos)       // Enter op has shape
  {
    return;
  }

  int iInputNums = pNode->num_inputs();  // Enter has only one input
  if (iInputNums != 1) {
    ADP_LOG(ERROR) << "Node " << pNode->name() << ", type is " << pNode->type_string()
                   << ", must has only one input, but now=" << iInputNums;
    LOG(ERROR) << "Node " << pNode->name() << ", type is " << pNode->type_string()
               << ", must has only one input, but now=" << iInputNums;
    return;
  }
  std::vector<tensorflow::shape_inference::ShapeHandle> inputShapes(iInputNums);

  (void) getInputShapesOfNode(shapeRef, pNode, inputShapes);

  pCxt->set_output(0, inputShapes.at(0));  // Enter op can't be unknown shape.
}

void InferShapeUtil::setShapeOfMergeOP(const ShapeRefiner &shapeRef, const Node *pNode) {
  CHECK_NOT_NULL(pNode);
  tensorflow::shape_inference::InferenceContext *pCxt = shapeRef.GetContext(pNode);
  CHECK_NOT_NULL(pCxt);
  tensorflow::shape_inference::ShapeHandle shapeOutOne = pCxt->output(0);  // Set Ref/Merge first output
  if (pCxt->DebugString(shapeOutOne).find('?') == std::string::npos)       // Ref/Merge op has shape
  {
    return;
  }

  for (const Edge *e : pNode->in_edges()) {
    CHECK_NOT_NULL(e);
    if (e->IsControlEdge())
      continue;
    if (e->dst_input() < 0)
      continue;

    if (e->src()->type_string() == "Enter" || e->src()->type_string() == "RefEnter") {
      Node *pNodeIn = e->src();
      tensorflow::shape_inference::InferenceContext *pCxtIn = shapeRef.GetContext(pNodeIn);
      if (pCxtIn == nullptr) {
        ADP_LOG(ERROR) << "Can't get context of the input " << pNodeIn->name() << " of the node " << pNode->name()
                       << ".";
        LOG(ERROR) << "Can't get context of the input " << pNodeIn->name() << " of the node " << pNode->name() << ".";
        return;
      }
      pCxt->set_output(0, pCxtIn->output(e->src_output()));
      return;
    }
  }
}

void InferShapeUtil::inferShapeOfGraph(const Graph *graph, ShapeRefiner &shapeRef, int iTime) {
  CHECK_NOT_NULL(graph);
  for (Node *pNode : graph->nodes()) {
    CHECK_NOT_NULL(pNode);
    if (pNode->type_string() == "NoOp" || shapeRef.GetContext(pNode) != nullptr) {
      continue;
    }

    Status addStatus = shapeRef.AddNode(pNode);
    if (!addStatus.ok()) {
      if (iTime != INFER_SHAPE_FIRST_TIME) {
        ADP_LOG(WARNING) << "AddNode failed, errormsg is " << addStatus.error_message() << ".";
        LOG(WARNING) << "AddNode failed, errormsg is " << addStatus.error_message() << ".";
      }
      continue;
    } else if (iTime == INFER_SHAPE_FIRST_TIME && pNode->type_string() == "Enter") {
      setShapeOfEnterOP(shapeRef, pNode);
    } else if ((iTime == INFER_SHAPE_FIRST_TIME) &&
               ((pNode->type_string() == "Merge") || (pNode->type_string() == "RefMerge"))) {
      setShapeOfMergeOP(shapeRef, pNode);
    }
  }
}

Status InferShapeUtil::addShapeToAttr(ShapeRefiner &shapeRef, Node *pNode) {
  REQUIRES_NOT_NULL(pNode);
  shape_inference::InferenceContext *pCxt = shapeRef.GetContext(pNode);
  if (pCxt == nullptr) {
    ADP_LOG(WARNING) << "The InferenceContext of node " << pNode->name() << " is null.";
    return Status::OK();
  }

  int iOutNums = pCxt->num_outputs();
  if (iOutNums <= 0) {
    return Status::OK();
  }

  AttrSlice attrList = pNode->attrs();
  if (attrList.Find(KEY_SHAPE) != nullptr) {
    ADP_LOG(INFO) << "Node " << pNode->name() << " already has omop_shape attribute.";
    return Status::OK();
  }

  std::vector<TensorShapeProto> shapeVec;
  for (int i = 0; i < iOutNums; i++) {
    tensorflow::shape_inference::ShapeHandle shape = pCxt->output(i);
    TensorShapeProto proto;
    pCxt->ShapeHandleToProto(shape, &proto);
    shapeVec.push_back(proto);

    string strShape = pCxt->DebugString(shape);
    if (strShape.find('?') != std::string::npos) {
      ADP_LOG(WARNING) << "The shape of node " << pNode->name() << " output " << i << " is " << strShape
                       << ", unknown shape.";

      auto identifier = NpuOpsIdentifier::GetInstance(false);
      if (identifier->IsPerformanceSensitive(pNode->type_string())) {
        return errors::Internal("Node ", pNode->name(), " output ", i, " shape is ", strShape, ", type is ",
                                pNode->type_string(), ", performance sensitive op shouldn't has unknown shape.");
      }
    }
  }

  pNode->AddAttr(KEY_SHAPE, gtl::ArraySlice<TensorShapeProto>(shapeVec));
  return Status::OK();
}

Status InferShapeUtil::InferShape(const std::vector<Tensor> &vecTensor, const FunctionLibraryDefinition *flib_def,
                                  const FunctionDef *func_def, Graph *graph) {
  (void) flib_def;
  REQUIRES_NOT_NULL(graph);
  REQUIRES_NOT_NULL(func_def);
  ADP_LOG(INFO) << "InferShapeUtil::InferShape";
  size_t iTensorNums = vecTensor.size();
  const OpDef &sig = func_def->signature();
  size_t iInputArgNums = static_cast<size_t>(sig.input_arg_size());
  if (iTensorNums < iInputArgNums) {
    return errors::Internal("Input tensor num ", iTensorNums, " is less than arg num ", iInputArgNums, ".");
  }

  TF_RETURN_IF_ERROR(GetSubGraphFromFunctionDef(*flib_def, *func_def, graph));

  // Control flow loops in the graph; we have to break them.
  std::vector<EdgeInfo> NextIterationEdges;
  std::unordered_set<const Edge *> needRemoveEdges;
  for (Node *pNode : graph->nodes()) {
    REQUIRES_NOT_NULL(pNode);
    if ((pNode->type_string() != "Merge") && (pNode->type_string() != "RefMerge")) {
      continue;
    }

    needRemoveEdges.clear();
    for (const Edge *e : pNode->in_edges()) {
      REQUIRES_NOT_NULL(e);
      if (e->IsControlEdge())
        continue;
      if (e->dst_input() < 0)
        continue;

      ADP_LOG(INFO) << "in_edges: " << e->src()->name() << " --> " << pNode->name();
      if ((e->src()->type_string() == "NextIteration") || (e->src()->type_string() == "RefNextIteration")) {
        EdgeInfo edgeInfo(e->src(), pNode, e->src_output(), e->dst_input());
        NextIterationEdges.push_back(edgeInfo);
        (void)needRemoveEdges.insert(e);
      }
    }
    for (auto needRemoveEdge : needRemoveEdges) {
      graph->RemoveEdge(needRemoveEdge);  // Use Enter replace NextIteration.
    }
  }

  ShapeRefiner shapeRefinerSub(graph->versions(), graph->op_registry());
  shapeRefinerSub.set_require_shape_inference_fns(false);
  shapeRefinerSub.set_disable_constant_propagation(true);

  TF_RETURN_IF_ERROR(setArgShapeFromTensorShape(vecTensor, graph, sig, shapeRefinerSub));
  inferShapeOfGraph(graph, shapeRefinerSub, INFER_SHAPE_FIRST_TIME);
  inferShapeOfGraph(graph, shapeRefinerSub, INFER_SHAPE_OTHER_TIME);

  for (Node *pNode : graph->nodes()) {
    TF_RETURN_IF_ERROR(addShapeToAttr(shapeRefinerSub, pNode));
  }

  for (auto &edgeInfo : NextIterationEdges) {
    (void)graph->AddEdge(edgeInfo.src_, edgeInfo.src_output_, edgeInfo.dst_, edgeInfo.dst_input_);
  }

  ADP_LOG(INFO) << "InferShapeUtil::InferShape success";
  return Status::OK();
}
}  // namespace tensorflow
