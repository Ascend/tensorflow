/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include "npu_device.h"
#include "npu_global.h"
#include "npu_logger.h"

using namespace tensorflow;
namespace npu {
class NpuCallOp : public OpKernel {
 public:
  explicit NpuCallOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &attr_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("device", &device_id_));
  }

  ~NpuCallOp() override = default;

  void Compute(OpKernelContext *ctx) override {
    DLOG() << "Compute npu op " << name() << " with function " << attr_.name();
    std::lock_guard<std::mutex> lk(mu_);  // Prevent run same npu graph parallel
    OP_REQUIRES_OK(ctx, Initilize(ctx));
    OP_REQUIRES_OK(ctx, Build(ctx));

    if (empty_ge_graph_) {
      DLOG() << "Compute bypass for cluster graph " << attr_.name() << " of " << name() << " as empty ge graph";
      return;
    }

    TFE_Context *context;
    NpuDevice *device;
    auto status = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>(TF_NewStatus(), TF_DeleteStatus);
    OP_REQUIRES_OK(ctx, global::NpuCtx::GetDeviceCtx(device_id_, &context, &device));

    std::vector<TFE_TensorHandle *> inputs;
    inputs.reserve(ctx->num_inputs());

    npu::ScopeTensorHandleDeleter guarder;
    for (int i = 0; i < ctx->num_inputs(); i++) {
      inputs.push_back(tensorflow::wrap(TensorHandle::CreateLocalHandle(ctx->input(i))));
      guarder.Guard(inputs.back());
    }

    std::vector<TFE_TensorHandle *> outputs(ctx->num_outputs());
    device->RunGeGraphPin2Cpu(context, graph_id_, inputs.size(), inputs.data(), output_types(), outputs.size(),
                              outputs.data(), status.get());
    OP_REQUIRES_OK(ctx, status->status);

    for (auto &handle : outputs) {
      guarder.Guard(handle);
    }
    for (int i = 0; i < outputs.size(); i++) {
      const Tensor *tensor;
      OP_REQUIRES_OK(ctx, npu::GetTensorHandleTensor(outputs[i], &tensor));
      ctx->set_output(i, *tensor);
    }
  }

  tensorflow::Status Initilize(OpKernelContext *ctx) {
    if (initialized_) {
      return tensorflow::Status::OK();
    }

    const tensorflow::FunctionLibraryDefinition *lib_def = ctx->function_library()->GetFunctionLibraryDefinition();
    const tensorflow::FunctionDef *fdef = lib_def->Find(attr_.name());
    NPU_REQUIRES(fdef != nullptr, tensorflow::errors::Internal("Failed lookup function ", attr_.name()));
    std::unique_ptr<tensorflow::FunctionBody> fbody;
    NPU_REQUIRES_OK(FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice{}, lib_def, &fbody));

    graph_ = std::make_unique<tensorflow::Graph>(lib_def);
    graph_def_ = std::make_unique<tensorflow::GraphDef>();
    dumper_ = std::make_unique<OptimizeStageGraphDumper>(name() + "." + attr_.name());
    CopyGraph(*fbody->graph, graph_.get());
    NpuCustomizedOptimizeGraph(ctx->function_library(), &graph_);
    PruneGraphByFunctionSignature(*fdef, graph_.get(), true);

    for (auto node : graph_->op_nodes()) {
      if (!node->IsArg()) {
        continue;
      }
      auto index = node->attrs().Find("index")->i();
      if (index >= args_.size()) {
        args_.resize(index + 1);
      }
      args_[index] = node;
    }
    input_shapes_.resize(args_.size(), absl::nullopt);
    initialized_ = true;
    return tensorflow::Status::OK();
  }

  bool MaybeUpdateShape(OpKernelContext *ctx) {
    bool updated = false;
    for (int i = 0; i < ctx->num_inputs(); i++) {
      auto &shape = input_shapes_[i];
      auto &value_shape = ctx->input(i).shape();
      if (!shape.has_value()) {
        updated = true;
        shape = value_shape;
        DLOG() << "Init input " << i << " shape " << shape.value().DebugString();
        args_[i]->ClearAttr("_output_shapes");
        args_[i]->AddAttr("_output_shapes", std::vector<PartialTensorShape>{shape.value()});
      } else {
        if (shape.value().IsCompatibleWith(value_shape)) {
          continue;
        } else {
          updated = true;
          DLOG() << "Compat input " << i << " shape " << shape.value().DebugString() << " vs. "
                 << value_shape.DebugString();
          shape = MakeCompatShape(shape.value(), value_shape);
          DLOG() << "Refresh input " << i << " shape to " << shape.value().DebugString();
          args_[i]->ClearAttr("_output_shapes");
          args_[i]->AddAttr("_output_shapes", std::vector<PartialTensorShape>{shape.value()});
        }
      }
    }
    return updated;
  }

  tensorflow::Status Build(OpKernelContext *ctx) {
    if (built_ && empty_ge_graph_) {
      DLOG() << "Skip check re-build for empty ge graph " << attr_.name() << " of " << name();
      return tensorflow::Status::OK();
    }
    bool shape_changed = MaybeUpdateShape(ctx);
    if (!built_ || shape_changed) {
      auto lib_def = ctx->function_library()->GetFunctionLibraryDefinition();
      auto graph = std::make_unique<tensorflow::Graph>(lib_def);
      CopyGraph(*graph_, graph.get());
      AssembleParserAddons(lib_def, graph.get());
      graph->ToGraphDef(graph_def_.get());
      dumper_->DumpWithSubGraphs("NPU_FUNCTION.shape_refreshed", *graph_def_, lib_def);
    }

    TFE_Context *context;
    NpuDevice *device;
    auto status = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>(TF_NewStatus(), TF_DeleteStatus);
    NPU_REQUIRES_OK(global::NpuCtx::GetDeviceCtx(device_id_, &context, &device));
    if (!built_) {
      graph_id_ = device->AddGeGraph(context, attr_.name(), *graph_def_, status.get());
      NPU_REQUIRES_OK(status->status);
      empty_ge_graph_ = graph_id_ == kEmptyGeGraphId;
      if (!empty_ge_graph_) {
        DLOG() << "Add ge graph " << attr_.name() << " with id " << graph_id_ << " for " << name();
      } else {
        DLOG() << "Ge graph of " << attr_.name() << " is empty for " << name();
      }
      built_ = true;
    } else {
      if (shape_changed || device->GeSession()->IsGraphNeedRebuild(graph_id_)) {
        DLOG() << "Remove and re-add ge graph " << attr_.name() << " with id " << graph_id_ << " as "
               << (shape_changed ? "shape changed" : "need rebuild");
        [this, &status, &device]() {
          NPU_CTX_REQUIRES_GE_OK(status, "Graph engine remove graph", device->GeSession()->RemoveGraph(graph_id_));
        }();
        NPU_REQUIRES_OK(status->status);
        device->AddGeGraph(context, graph_id_, attr_.name(), *graph_def_, status.get());
        NPU_REQUIRES_OK(status->status);
      }
    }
    return tensorflow::Status::OK();
  }

 private:
  static PartialTensorShape MakeCompatShape(const PartialTensorShape &a, const PartialTensorShape &b) {
    if (a.dims() != b.dims()) {
      return PartialTensorShape();
    }
    PartialTensorShape shape;
    static constexpr int64 kUnknownDim = -1;
    for (int i = 0; i < a.dims(); i++) {
      if (a.dim_size(i) != b.dim_size(i)) {
        shape.AddDim(kUnknownDim);
      } else {
        shape.AddDim(a.dim_size(i));
      }
    }
    return shape;
  }

  bool initialized_{false};
  bool built_{false};
  bool empty_ge_graph_{false};
  int device_id_{0};
  uint64_t graph_id_{0U};
  NameAttrList attr_;
  std::mutex mu_;
  std::unique_ptr<OptimizeStageGraphDumper> dumper_{nullptr};
  std::unique_ptr<tensorflow::Graph> graph_;
  std::unique_ptr<tensorflow::GraphDef> graph_def_;
  std::vector<tensorflow::Node *> args_;
  std::vector<absl::optional<PartialTensorShape>> input_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("NpuCall").Device(DEVICE_CPU), NpuCallOp);
}  // namespace npu