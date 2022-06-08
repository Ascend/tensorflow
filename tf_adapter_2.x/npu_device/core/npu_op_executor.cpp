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

#include "npu_op_executor.h"

#include "op_executors/npu_kernel_registry.h"

#include "op_executors/npu_concrete_graph.h"
#include "op_executors/npu_custom_kernel_op.h"
#include "op_executors/npu_dynamic_shape_op.h"
#include "op_executors/npu_mirrored_op.h"
#include "op_executors/npu_resource_op.h"
#include "op_executors/npu_resource_generator_op.h"
#include "op_executors/npu_shape_depend_on_value_op.h"
#include "op_executors/npu_static_shape_op.h"
#include "op_executors/npu_unsupported_op.h"

namespace npu {
std::shared_ptr<OpExecutor> OpExecutor::Create(TFE_Context *context, NpuDevice *device, const tensorflow::NodeDef &ndef,
                                               int num_inputs, TFE_TensorHandle **inputs, TF_Status *s) {
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  const tensorflow::OpRegistrationData *op_reg_data;
  const char *op_name = ndef.op().c_str();

  NPU_CTX_REQUIRES_OK_RETURN(s, lib_def->LookUp(op_name, &op_reg_data), nullptr);

  NpuCustomKernelFunc *kernel;
  if (CustomKernelRegistry::Instance().GetCustomKernelFunc(ndef.op(), &kernel)) {
    return std::make_shared<NpuCustomKernelOp>(op_reg_data, ndef, TensorShapes{}, *kernel);
  }

  NpuFallbackHookFunc *hook;
  if (CustomKernelRegistry::Instance().GetFallbackHookFunc(ndef.op(), &hook)) {
    return std::make_shared<NpuMirroredOp>(op_reg_data, ndef, TensorShapes{}, *hook);
  }

  if (op_reg_data->is_function_op) {
    std::unique_ptr<NpuConcreteGraph> concrete_graph;
    device->GetConcreteGraph(context, ndef, num_inputs, inputs, &concrete_graph, s);
    if (TF_GetCode(s) != TF_OK) {
      return nullptr;
    }
    return concrete_graph;
  }

  TensorShapes input_shapes;
  input_shapes.resize(static_cast<size_t>(num_inputs));
  for (int i = 0; i < num_inputs; i++) {
    NPU_CTX_REQUIRES_OK_RETURN(s, npu::GetTensorHandleShape(inputs[i], input_shapes[static_cast<size_t>(i)]), nullptr);
  }

  TensorDataTypes input_types;
  TensorDataTypes output_types;
  NPU_CTX_REQUIRES_OK_RETURN(s, tensorflow::InOutTypesForNode(ndef, op_reg_data->op_def, &input_types, &output_types),
                             nullptr);

  if (!device->Supported(op_name)) {
    return std::make_shared<NpuUnsupportedOp>(op_reg_data, ndef, input_shapes, "Op unsupported by NPU");
  }

  for (auto type : input_types) {
    if (type == tensorflow::DT_RESOURCE) {
      return std::make_shared<NpuResourceOp>(op_reg_data, ndef, input_shapes);
    }
  }

  for (auto type : output_types) {
    if (type == tensorflow::DT_RESOURCE) {
      return std::make_shared<NpuResourceGeneratorOp>(op_reg_data, ndef, input_shapes);
    }
  }

  tensorflow::Status status;
  if (!(status = device->ValidateOutputTypes(output_types)).ok()) {
    return std::make_shared<NpuUnsupportedOp>(op_reg_data, ndef, input_shapes, status.error_message());
  }

  if (!(status = device->ValidateInputTypes(input_types)).ok()) {
    return std::make_shared<NpuUnsupportedOp>(op_reg_data, ndef, input_shapes, status.error_message());
  }

  if (op_reg_data->shape_inference_fn == nullptr) {
    return std::make_shared<NpuUnsupportedOp>(op_reg_data, ndef, input_shapes, "No infer shape function registered");
  }

  tensorflow::shape_inference::InferenceContext ic(
    TF_GRAPH_DEF_VERSION, ndef, op_reg_data->op_def,
    std::vector<tensorflow::shape_inference::ShapeHandle>(input_shapes.size()), {}, {}, {});

  if (!(status = ic.construction_status()).ok()) {
    return std::make_shared<NpuUnsupportedOp>(op_reg_data, ndef, input_shapes, status.error_message());
  }

  for (size_t i = 0; i < input_shapes.size(); i++) {
    std::vector<tensorflow::shape_inference::DimensionHandle> dims_handle;
    for (const auto &dim_size : input_shapes[i].dim_sizes()) {
      dims_handle.push_back(ic.MakeDim(dim_size));
    }
    ic.SetInput(static_cast<int>(i), ic.MakeShape(dims_handle));
  }

  NPU_CTX_REQUIRES_OK_RETURN(s, ic.Run(op_reg_data->shape_inference_fn), nullptr);

  for (size_t i = 0; i < input_shapes.size(); i++) {
    if (ic.requested_input_tensor(static_cast<int>(i))) {
      return std::make_shared<NpuShapeDependOnValueOp>(op_reg_data, ndef, input_shapes);
    }
  }

  TensorPartialShapes partial_shapes(ic.num_outputs());
  for (int i = 0; i < ic.num_outputs(); i++) {
    tensorflow::shape_inference::ShapeHandle shape_handle = ic.output(i);
    auto num_dims = tensorflow::shape_inference::InferenceContext::Rank(shape_handle);
    std::vector<tensorflow::int64> dims;
    if (num_dims == tensorflow::shape_inference::InferenceContext::kUnknownRank) {
      continue;
    }
    for (auto j = 0; j < num_dims; ++j) {
      dims.emplace_back(ic.Value(ic.Dim(shape_handle, j)));
    }

    NPU_CTX_REQUIRES_OK_RETURN(s, tensorflow::PartialTensorShape::MakePartialShape(
        dims.data(), num_dims, &partial_shapes[static_cast<size_t>(i)]), nullptr);
  }

  TensorShapes output_shapes(partial_shapes.size());
  for (size_t i = 0; i < partial_shapes.size(); i++) {
    if (!partial_shapes[i].AsTensorShape(&output_shapes[i])) {
      return std::make_shared<NpuDynamicShapeOp>(op_reg_data, ndef, input_shapes, partial_shapes);
    }
  }
  return std::make_shared<NpuStaticShapeOp>(op_reg_data, ndef, input_shapes, output_shapes);
}
}  // namespace npu
