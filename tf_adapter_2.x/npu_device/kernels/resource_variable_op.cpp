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

#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"

#include "absl/algorithm/container.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"

#include "npu_custom_kernel.h"
#include "npu_utils.h"

namespace npu {
namespace {
class AssignVariableGraphBuilder {
 public:
  static tensorflow::GraphDef GetGraph(const std::string &op_name, const std::string &container_name,
                                       const std::string &shared_name, const tensorflow::Tensor &tensor,
                                       TF_Status *status) {
    tensorflow::GraphDef gdef;

    tensorflow::Graph graph(tensorflow::OpRegistry::Global());
    tensorflow::Node *variable;
    tensorflow::Node *value;
    tensorflow::Node *assign_variable;

    NPU_CTX_REQUIRES_OK_RETURN(status,
                               tensorflow::NodeBuilder(WrapResourceName(shared_name), "VarHandleOp")
                                 .Attr("container", container_name)
                                 .Attr("shared_name", shared_name)
                                 .Attr("dtype", tensor.dtype())
                                 .Attr("shape", tensor.shape())
                                 .Finalize(&graph, &variable),
                               gdef);
    NPU_CTX_REQUIRES_OK_RETURN(status,
                               tensorflow::NodeBuilder(op_name + "_Value_" + shared_name, "_Arg")
                                 .Attr("T", tensor.dtype())
                                 .Attr("index", 0)
                                 .Finalize(&graph, &value),
                               gdef);
    NPU_CTX_REQUIRES_OK_RETURN(status,
                               tensorflow::NodeBuilder(op_name + "_" + shared_name, op_name)
                                 .Input(variable, 0)
                                 .Input(value, 0)
                                 .Attr("dtype", tensor.dtype())
                                 .Finalize(&graph, &assign_variable),
                               gdef);

    AssembleOpDef(variable);
    AssembleOpDef(value);
    AssembleOpDef(assign_variable);

    AssembleOutputDesc(TensorShapes({kScalarShape}), {tensorflow::DT_RESOURCE}, variable);
    AssembleOutputDesc(TensorShapes({tensor.shape()}), {tensor.dtype()}, value);
    AssembleInputDesc(TensorShapes({kScalarShape, tensor.shape()}), {tensorflow::DT_RESOURCE, tensor.dtype()},
                      assign_variable);

    graph.ToGraphDef(&gdef);
    return gdef;
  }
};

void VariableOpBaseKernel(const std::string &op_name, TFE_Context *context, NpuDevice *dev, const OpSpec *spec,
                          const TensorShapes &output_shapes, const tensorflow::NodeDef &parser_ndef, int num_inputs,
                          TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  TF_UNUSED_VARIABLE(spec);
  TF_UNUSED_VARIABLE(output_shapes);
  TF_UNUSED_VARIABLE(parser_ndef);
  TF_UNUSED_VARIABLE(num_inputs);
  const tensorflow::Tensor *handle = nullptr;
  const tensorflow::Tensor *value = nullptr;

  ScopeTensorHandleDeleter scope_handle_deleter;
  TFE_TensorHandle *value_handle = inputs[1];
  if (IsNpuTensorHandle(inputs[1])) {
    value_handle = dev->CopyTensorD2H(context, inputs[1], status);
    if (TF_GetCode(status) != TF_OK) return;
    scope_handle_deleter.Guard(value_handle);
  }

  NPU_CTX_REQUIRES_OK(status, GetTensorHandleTensor(inputs[0], &handle));
  auto resource = handle->scalar<tensorflow::ResourceHandle>()();
  NPU_CTX_REQUIRES_OK(status, GetTensorHandleTensor(value_handle, &value));
  DLOG() << "Start run " << op_name << " for resource " << resource.DebugString() << " with value "
         << value->DebugString();
  auto var_init_graph =
    AssignVariableGraphBuilder::GetGraph(op_name, resource.container(), resource.name(), *value, status);
  if (TF_GetCode(status) != TF_OK) return;
  std::string graph_name = op_name + "_" + resource.name();
  if (kDumpExecutionDetail && kDumpGraph) {
    std::string file_name = graph_name + ".pbtxt";
    WriteTextProto(tensorflow::Env::Default(), file_name, var_init_graph);
    LOG(INFO) << "NPU Dump variable resource init graph to: " << file_name;
  }

  dev->RunGeGraphPin2CpuAnonymous(context, graph_name, var_init_graph, 1, &value_handle, num_outputs, outputs, status);
}
}  // namespace

static auto kernel_assign = [](TFE_Context *context, NpuDevice *dev, const OpSpec *spec,
                               const TensorShapes &output_shapes, const tensorflow::NodeDef &parser_ndef,
                               int num_inputs, TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs,
                               TF_Status *status) {
  VariableOpBaseKernel("AssignVariableOp", context, dev, spec, output_shapes, parser_ndef, num_inputs, inputs,
                       num_outputs, outputs, status);
};

static auto kernel_assign_add = [](TFE_Context *context, NpuDevice *dev, const OpSpec *spec,
                                   const TensorShapes &output_shapes, const tensorflow::NodeDef &parser_ndef,
                                   int num_inputs, TFE_TensorHandle **inputs, int num_outputs,
                                   TFE_TensorHandle **outputs, TF_Status *status) {
  VariableOpBaseKernel("AssignAddVariableOp", context, dev, spec, output_shapes, parser_ndef, num_inputs, inputs,
                       num_outputs, outputs, status);
};

static auto kernel_assign_sub = [](TFE_Context *context, NpuDevice *dev, const OpSpec *spec,
                                   const TensorShapes &output_shapes, const tensorflow::NodeDef &parser_ndef,
                                   int num_inputs, TFE_TensorHandle **inputs, int num_outputs,
                                   TFE_TensorHandle **outputs, TF_Status *status) {
  VariableOpBaseKernel("AssignSubVariableOp", context, dev, spec, output_shapes, parser_ndef, num_inputs, inputs,
                       num_outputs, outputs, status);
};

NPU_REGISTER_CUSTOM_KERNEL("AssignVariableOp", kernel_assign);
NPU_REGISTER_CUSTOM_KERNEL("AssignAddVariableOp", kernel_assign_add);
NPU_REGISTER_CUSTOM_KERNEL("AssignSubVariableOp", kernel_assign_sub);
}  // namespace npu
