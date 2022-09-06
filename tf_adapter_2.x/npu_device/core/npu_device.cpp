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

#include "npu_device.h"

#include <memory>
#include <utility>
#include <future>
#include <fstream>

#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/graph/algorithm.h"

#include "npu_global.h"
#include "npu_managed_buffer.h"
#include "npu_tensor.h"
#include "npu_utils.h"

#include "optimizers/npu_optimizer_manager.h"

#include "framework/common/ge_inner_error_codes.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_factory.h"
#include "nlohmann/json.hpp"

using Format = ge::Format;

namespace {
template <typename T, typename DT>
class NpuHostFixedAllocator : public tensorflow::Allocator, public tensorflow::core::RefCounted {
 public:
  static tensorflow::Allocator *Create(std::unique_ptr<T, DT> ptr) {
    return new (std::nothrow) NpuHostFixedAllocator(std::move(ptr));
  }

 private:
  explicit NpuHostFixedAllocator(std::unique_ptr<T, DT> ptr) : ptr_(std::move(ptr)) {
    DLOG() << "Zero copied ge tensor " << reinterpret_cast<uintptr_t>(ptr_.get());
  }
  ~NpuHostFixedAllocator() override {
    DLOG() << "Release zero copied ge tensor " << reinterpret_cast<uintptr_t>(ptr_.get());
  }
  std::string Name() override { return "NpuHostFixedAllocator"; }
  void *AllocateRaw(size_t alignment, size_t num_bytes) override {
    (void)alignment;
    (void)num_bytes;
    return ptr_.get();
  }
  void DeallocateRaw(void *ptr) override {
    (void)ptr;
    Unref();
  }
  std::unique_ptr<T, DT> ptr_;
};
}  // namespace

namespace npu {
/**
 * @brief: create iterator providev
 * @param context: tfe context
 * @param tensor: tensorflow tensor
 * @param device_ids: device ids
 * @param status: tf status
 */
void NpuDevice::CreateIteratorProvider(TFE_Context *context, const tensorflow::Tensor *tensor,
                                       std::vector<int> device_ids, TF_Status *status) {
  auto resource = tensor->scalar<tensorflow::ResourceHandle>()();
  TensorPartialShapes shapes;
  TensorDataTypes types;
  NPU_CTX_REQUIRES_OK(status, GetMirroredIteratorShapesAndTypes(resource, shapes, types));
  tensorflow::CancellationManager *cancel_manager = CancellationManager();
  std::string channel_name = npu::WrapResourceName(resource.name());

  NPU_CTX_REQUIRES_OK(status, npu::global::GlobalHdcChannel::GetInstance().Create(
                                channel_name, CreateChannelCapacity(shapes, types), device_ids));
  std::vector<std::shared_ptr<npu::HdcChannel>> channels;
  npu::global::GlobalHdcChannel::GetInstance().Get(channel_name, channels);
  if (!channels.empty()) {
    tensorflow::CancellationToken token = cancel_manager->get_cancellation_token();
    bool cancelled = !cancel_manager->RegisterCallback(
      token, [channel_name]() { npu::global::GlobalHdcChannel::GetInstance().Destroy(channel_name); });
    if (cancelled) {
      status->status = tensorflow::errors::Internal("Iterator resource ", channel_name, " consume after destroyed");
      return;
    }
  }

  auto dp_provider =
    npu::IteratorResourceProvider::GetFunctionDef(channel_name, std::move(device_ids), shapes, types, status);
  NPU_REQUIRES_TFE_OK(status);

  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  NPU_CTX_REQUIRES_OK(status, lib_def->AddFunctionDef(dp_provider));
  tensorflow::ProcessFunctionLibraryRuntime *pflr = npu::UnwrapCtx(context)->pflr();
  tensorflow::FunctionLibraryRuntime *flr = pflr->GetFLR(underlying_device);
  tensorflow::FunctionLibraryRuntime::Handle f_handle;
  NPU_CTX_REQUIRES_OK(status, flr->Instantiate(dp_provider.signature().name(), tensorflow::AttrSlice{}, &f_handle));

  auto consume_func = [flr, f_handle, cancel_manager](tensorflow::Tensor tensor, int64_t nums) -> tensorflow::Status {
    std::vector<tensorflow::Tensor> get_next_outputs;
    tensorflow::FunctionLibraryRuntime::Options options;
    tensorflow::CancellationManager child(cancel_manager);
    options.cancellation_manager = &child;
    return flr->RunSync(options, f_handle, {std::move(tensor), tensorflow::Tensor(tensorflow::int64(nums))},
                        &get_next_outputs);
  };
  auto destroy_func = [channel_name, resource, flr, f_handle]() -> tensorflow::Status {
    LOG(INFO) << "Stopping iterator resource provider for " << resource.name();
    npu::global::GlobalHdcChannel::GetInstance().Destroy(channel_name);
    return flr->ReleaseHandle(f_handle);
  };

  auto provider = std::make_shared<npu::IteratorResourceProvider>(resource.name(), consume_func, destroy_func);
  LOG(INFO) << "Iterator resource provider for " << resource.name() << " created";

  NPU_CTX_REQUIRES(status, provider != nullptr,
                   tensorflow::errors::Internal("Failed create iterator reosurce provider for ", resource.name()));

  iterator_providers_[resource] = provider;

  if (kDumpExecutionDetail || kDumpGraph) {
    std::unique_ptr<tensorflow::FunctionBody> fbody;
    tensorflow::AttrSlice attr_slice;
    tensorflow::FunctionDefToBodyHelper(dp_provider, attr_slice, lib_def, &fbody);
    std::string file_name = "dp_provider_" + resource.name() + ".pbtxt";
    WriteTextProto(tensorflow::Env::Default(), file_name, fbody->graph->ToGraphDefDebug());
  }
}

/**
 * @brief: create iterator providev
 * @param context: tfe context
 * @param tensor: tensorflow tensor
 * @param device_ids: device ids
 * @param status: tf status
 */
std::shared_ptr<IteratorResourceProvider> NpuDevice::GetIteratorProvider(const TFE_Context *const context,
                                                                         const tensorflow::ResourceHandle &resource) {
  (void)context;
  const decltype(iterator_providers_)::const_iterator provider = iterator_providers_.find(resource);
  if (provider == iterator_providers_.cend()) {
    return nullptr;
  }
  return provider->second;
}

/**
 * @brief: erase iterator provider
 * @param context: tfe context
 * @param resource: tensorflow resource handle
 */
void NpuDevice::EraseIteratorProvider(const TFE_Context *const context, const tensorflow::ResourceHandle &resource) {
  (void)context;
  DLOG() << "Start to erase provider: " << resource.DebugString();
  iterator_providers_.erase(resource);
}

/**
 * @brief: create device
 * @param name: device name
 * @param device_index: device id
 * @param device_options: device options
 * @param device: NpuDevice
 */
std::string NpuDevice::CreateDevice(const char *name, int device_index,
                                    const std::map<std::string, std::string> &options, NpuDevice **device) {
  auto *ge_session = new (std::nothrow) ge::Session(options);
  if (ge_session == nullptr) {
    return "Failed init graph engine: create new session failed";
  }

  *device = new (std::nothrow) NpuDevice();
  if (*device == nullptr) {
    return "Failed create new npu device instance";
  }
  (*device)->device_id = device_index;
  (*device)->device_name = name;
  (*device)->device_options = options;
  (*device)->underlying_device = "/job:localhost/replica:0/task:0/device:CPU:0";
  (*device)->ge_session_ = ge_session;
  (*device)->cancellation_manager_ = std::make_unique<tensorflow::CancellationManager>();
  auto status = LoadSupportedOps((*device)->npu_supported_ops_);
  if (!status.ok()) {
    // We do not raise error for compat with old version
    LOG(ERROR) << "Could not load npu supported ops " << status.error_message();
  }
  (*device)->npu_stdout_receiver_ = std::make_unique<NpuStdoutReceiver>(device_index);
  (*device)->npu_stdout_receiver_->Start();
  return "";
}

/**
 * @brief: release resource
 */
void NpuDevice::ReleaseResource() {
  std::vector<std::future<void>> thread_guarder;
  thread_guarder.emplace_back(std::async([this]() { npu_stdout_receiver_->Stop(); }));
  for (auto &iterator_provider : iterator_providers_) {
    auto provider = iterator_provider.second;
    thread_guarder.emplace_back(std::async([provider]() { provider->Destroy(); }));
  }

  DLOG() << "Start cancel all uncompleted async call";
  CancellationManager()->StartCancel();
}

/**
 * @brief: delete device
 * @param device: point to NpuDevice
 */
void NpuDevice::DeleteDevice(void *device) {
  DLOG() << "Start destroy npu device instance";
  if (device == nullptr) {
    return;
  }
  auto npu_device = static_cast<NpuDevice *>(device);
  delete npu_device->ge_session_;
  delete npu_device;
}

bool NpuDevice::SupportedInputType(tensorflow::DataType data_type) const {
  return tensorflow::DataTypeCanUseMemcpy(data_type);
}

bool NpuDevice::SupportedOutputType(tensorflow::DataType data_type) const {
  return tensorflow::DataTypeCanUseMemcpy(data_type);
}

bool NpuDevice::SupportedInputAndOutputType(tensorflow::DataType data_type) const {
  return SupportedInputType(data_type) && SupportedOutputType(data_type);
}

tensorflow::Status NpuDevice::ValidateOutputTypes(const TensorDataTypes &data_types) const {
  std::stringstream ss;
  for (size_t i = 0; i < data_types.size(); i++) {
    auto data_type = data_types[i];
    if (!SupportedOutputType(data_type)) {
      ss << "Output " << i << " unsupported type " << tensorflow::DataTypeString(data_type) << std::endl;
    }
  }
  if (!ss.str().empty()) {
    return tensorflow::errors::Unimplemented(ss.str());
  }
  return tensorflow::Status::OK();
}

tensorflow::Status NpuDevice::ValidateInputTypes(const TensorDataTypes &data_types) const {
  std::stringstream ss;
  for (size_t i = 0; i < data_types.size(); i++) {
    auto data_type = data_types[i];
    if (!SupportedInputType(data_type)) {
      ss << "Input " << i << " unsupported type " << tensorflow::DataTypeString(data_type) << std::endl;
    }
  }
  if (!ss.str().empty()) {
    return tensorflow::errors::Unimplemented(ss.str());
  }
  return tensorflow::Status::OK();
}
/**
 * @brief: new device tensor handle
 * @param context: tfe context
 * @param fmt: format
 * @param shape: tensor shape
 * @param type: tensorflow data type
 * @param status: tf status
 */
TFE_TensorHandle *NpuDevice::NewDeviceTensorHandle(TFE_Context *context, Format fmt,
                                                   const tensorflow::TensorShape &shape, tensorflow::DataType type,
                                                   TF_Status *status) {
  npu::NpuManagedBuffer *npu_managed_buffer;
  NPU_CTX_REQUIRES_OK_RETURN(status, npu::NpuManagedBuffer::Create(fmt, shape, type, &npu_managed_buffer), nullptr);
  std::vector<int64_t> dims;
  for (auto dim_size : shape.dim_sizes()) {
    dims.emplace_back(dim_size);
  }

  auto buf = new TF_ManagedBuffer(npu_managed_buffer, sizeof(npu_managed_buffer), &npu::NpuManagedBufferDeallocator,
                                  nullptr, false);
  auto npu_tensor = std::make_unique<npu::NpuTensor>(tensorflow::Tensor(type, shape, buf));
  buf->Unref();  // buf has been ref two times at create time and when construct npu tensor

  TF_DataType dtype = TFE_TensorHandleDataType(npu_tensor->handle);
  return TFE_NewCustomDeviceTensorHandle(context, device_name.c_str(), dtype, npu_tensor.release(),
                                         npu::NpuTensor::handle_methods, status);
}

/**
 * @brief: new device resource handle
 * @param context: tfe context
 * @param shape: tensor shape
 * @param status: tf status
 */
TFE_TensorHandle *NpuDevice::NewDeviceResourceHandle(TFE_Context *context, const tensorflow::TensorShape &shape,
                                                     TF_Status *status) {
  auto npu_tensor = std::make_unique<npu::NpuTensor>(tensorflow::Tensor(tensorflow::DT_RESOURCE, shape));
  TF_DataType dtype = TFE_TensorHandleDataType(npu_tensor->handle);
  return TFE_NewCustomDeviceTensorHandle(context, device_name.c_str(), dtype, npu_tensor.release(),
                                         npu::NpuTensor::handle_methods, status);
}

/**
 * @brief: copy device tensor to host
 * @param context: tfe context
 * @param tensor: tfe tensor handle
 * @param status: tf status
 */
TFE_TensorHandle *NpuDevice::CopyTensorD2H(const TFE_Context *const context, TFE_TensorHandle *tensor,
                                           TF_Status *status) const {
  (void)context;
  const tensorflow::Tensor *npu_tensor;
  NPU_CTX_REQUIRES_OK_RETURN(status, npu::GetTensorHandleTensor(tensor, &npu_tensor), nullptr);

  if (npu_tensor->dtype() == tensorflow::DT_RESOURCE) {
    tensorflow::ResourceHandle handle = npu_tensor->scalar<tensorflow::ResourceHandle>()();
    status->status =
      tensorflow::errors::Internal("Resources ", handle.DebugString(), " cannot be copied across devices[NPU->CPU]");
    return nullptr;
  }

  const tensorflow::Tensor *local_tensor;
  TFE_TensorHandle *local_handle = tensorflow::wrap(
    tensorflow::TensorHandle::CreateLocalHandle(tensorflow::Tensor(npu_tensor->dtype(), npu_tensor->shape())));
  NPU_CTX_REQUIRES_RETURN(status, local_handle != nullptr, tensorflow::errors::Internal("Failed create local handle"),
                          nullptr);
  NPU_CTX_REQUIRES_OK_RETURN(status, npu::GetTensorHandleTensor(local_handle, &local_tensor), nullptr);
  NPU_CTX_REQUIRES_OK_RETURN(
    status, npu::Unwrap<npu::NpuManagedBuffer>(npu_tensor)->AssembleTo(const_cast<tensorflow::Tensor *>(local_tensor)),
    local_handle);
  return local_handle;
}

/**
 * @brief: copy host tensor to device
 * @param context: tfe context
 * @param tensor: tfe tensor handle
 * @param status: tf status
 */
TFE_TensorHandle *NpuDevice::CopyTensorH2D(TFE_Context *context, TFE_TensorHandle *tensor, TF_Status *status) {
  return CopyTensorH2D(context, tensor, Format::FORMAT_ND, status);
}

/**
 * @brief: copy host tensor to device
 * @param context: tfe context
 * @param tensor: tfe tensor handle
 * @param fmt: format
 * @param status: tf status
 */
TFE_TensorHandle *NpuDevice::CopyTensorH2D(TFE_Context *context, TFE_TensorHandle *tensor, Format fmt,
                                           TF_Status *status) {
  TFE_TensorHandle *local_handle = tensor;
  npu::ScopeTensorHandleDeleter scope_handle_deleter;
  if (!npu::IsCpuTensorHandle(tensor)) {
    local_handle = TFE_TensorHandleCopyToDevice(tensor, context, underlying_device.c_str(), status);
    scope_handle_deleter.Guard(local_handle);
  }

  if (TF_GetCode(status) != TF_OK) {
    return nullptr;
  }
  const tensorflow::Tensor *local_tensor = nullptr;
  NPU_CTX_REQUIRES_OK_RETURN(status, npu::GetTensorHandleTensor(local_handle, &local_tensor), nullptr);
  if (local_tensor->dtype() == tensorflow::DT_RESOURCE) {
    tensorflow::ResourceHandle handle = local_tensor->scalar<tensorflow::ResourceHandle>()();
    status->status =
      tensorflow::errors::Internal("Resources ", handle.DebugString(), " cannot be copied across devices[CPU->NPU]");
    return nullptr;
  }

  TFE_TensorHandle *npu_handle =
    NewDeviceTensorHandle(context, fmt, local_tensor->shape(), local_tensor->dtype(), status);
  if (TF_GetCode(status) != TF_OK) {
    return nullptr;
  }
  const tensorflow::Tensor *npu_tensor = nullptr;

  NPU_CTX_REQUIRES_OK_RETURN(status, npu::GetTensorHandleTensor(npu_handle, &npu_tensor), nullptr);
  NPU_CTX_REQUIRES_OK_RETURN(status, npu::Unwrap<npu::NpuManagedBuffer>(npu_tensor)->AssembleFrom(local_tensor),
                             npu_handle);
  return npu_handle;
}

/**
 * @brief: infer shape
 * @param context: tfe context
 * @param op_reg_data: op registration data
 * @param ndef: tensorflow node def
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param shapes: tensor partial shapes
 */
tensorflow::Status NpuDevice::InferShape(const TFE_Context *const context,
                                         const tensorflow::OpRegistrationData *op_reg_data,
                                         const tensorflow::NodeDef &ndef, int num_inputs, TFE_TensorHandle **inputs,
                                         TensorPartialShapes &shapes) const {
  NPU_REQUIRES(op_reg_data->shape_inference_fn,
               tensorflow::errors::Unimplemented("No infer shape function registered for op ", ndef.op()));

  tensorflow::shape_inference::InferenceContext ic(TF_GRAPH_DEF_VERSION, ndef, op_reg_data->op_def,
                                                   std::vector<tensorflow::shape_inference::ShapeHandle>(num_inputs),
                                                   {}, {}, {});
  NPU_REQUIRES_OK(ic.construction_status());
  for (int i = 0; i < num_inputs; i++) {
    auto input = tensorflow::unwrap(inputs[i]);
    std::vector<tensorflow::shape_inference::DimensionHandle> dims_handle;
    int num_dims;
    TF_RETURN_IF_ERROR(input->NumDims(&num_dims));
    for (int j = 0; j < num_dims; j++) {
      int64_t dim_size;
      TF_RETURN_IF_ERROR(input->Dim(j, &dim_size));
      dims_handle.push_back(ic.MakeDim(dim_size));
    }
    ic.SetInput(i, ic.MakeShape(dims_handle));
  }

  for (int i = 0; i < num_inputs; i++) {
    auto input = inputs[i];
    if (tensorflow::unwrap(input)->DataType() == tensorflow::DT_RESOURCE) {
      const tensorflow::Tensor *tensor;
      NPU_REQUIRES_OK(npu::GetTensorHandleTensor(input, &tensor));
      auto handle = tensor->flat<tensorflow::ResourceHandle>()(0);
      const auto &dtypes_and_shapes = handle.dtypes_and_shapes();
      std::vector<tensorflow::shape_inference::ShapeAndType> inference_shapes_and_types;
      for (auto &dtype_and_shape : dtypes_and_shapes) {
        std::vector<tensorflow::shape_inference::DimensionHandle> dims_handle(dtype_and_shape.shape.dims());
        for (size_t j = 0; j < dims_handle.size(); j++) {
          dims_handle[j] = ic.MakeDim(dtype_and_shape.shape.dim_size(j));
        }
        inference_shapes_and_types.emplace_back(ic.MakeShape(dims_handle), dtype_and_shape.dtype);
      }
      ic.set_input_handle_shapes_and_types(i, inference_shapes_and_types);
    }
  }
  // We need to feed the input tensors. TensorFlow performs inference based on the input shape for the first time.
  // If the shape function of an operator depends on the value of the input tensor, the shape function is marked for the
  // first time and the actual tensor value is used for inference for the second time.
  NPU_REQUIRES_OK(ic.Run(op_reg_data->shape_inference_fn));

  std::vector<const tensorflow::Tensor *> input_tensors;
  input_tensors.resize(num_inputs);
  ScopeTensorHandleDeleter scope_handle_deleter;
  bool input_requested = false;
  for (int i = 0; i < num_inputs; i++) {
    auto input = inputs[i];
    if (!ic.requested_input_tensor(i)) {
      continue;
    }
    // If requested, this must be a normal tensor
    if (npu::IsNpuTensorHandle(input)) {
      auto s = TF_NewStatus();
      if (s == nullptr) {
        continue;
      }
      input = CopyTensorD2H(context, input, s);
      if (TF_GetCode(s) != TF_OK) {
        TF_DeleteStatus(s);
        continue;
      }
      DLOG() << "Copying " << ndef.op() << " input:" << i << " from NPU to CPU for infer shape";
      scope_handle_deleter.Guard(input);
    }
    const tensorflow::Tensor *tensor;
    NPU_REQUIRES_OK(npu::GetTensorHandleTensor(input, &tensor));
    input_tensors[i] = tensor;
    input_requested = true;
  }
  if (input_requested) {
    ic.set_input_tensors(input_tensors);
    NPU_REQUIRES_OK(ic.Run(op_reg_data->shape_inference_fn));
  }

  for (int i = 0; i < ic.num_outputs(); i++) {
    shapes.emplace_back(tensorflow::PartialTensorShape());
    tensorflow::shape_inference::ShapeHandle shape_handle = ic.output(i);
    auto num_dims = ic.Rank(shape_handle);
    std::vector<tensorflow::int64> dims;
    if (num_dims == tensorflow::shape_inference::InferenceContext::kUnknownRank) {
      continue;
    }
    for (auto j = 0; j < num_dims; ++j) {
      dims.emplace_back(ic.Value(ic.Dim(shape_handle, j)));
    }
    NPU_REQUIRES_OK(tensorflow::PartialTensorShape::MakePartialShape(dims.data(), num_dims, &shapes[i]));
  }
  return tensorflow::Status::OK();
}

void NpuDevice::GetConcreteGraph(TFE_Context *context, const tensorflow::NodeDef &ndef, int num_inputs,
                                 TFE_TensorHandle **inputs, std::unique_ptr<NpuConcreteGraph> *concrete_graph,
                                 TF_Status *s) {
  const char *op_name = ndef.op().c_str();
  tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
  const tensorflow::FunctionDef *fdef = lib_def->Find(op_name);
  std::unique_ptr<tensorflow::FunctionBody> fbody;
  FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice(&ndef.attr()), lib_def, &fbody);
  std::unique_ptr<tensorflow::Graph> optimize_graph = std::make_unique<tensorflow::Graph>(lib_def);
  CopyGraph(*fbody->graph, optimize_graph.get());

  OptimizeStageGraphDumper graph_dumper(op_name);

  NPU_CTX_REQUIRES_OK(
    s, NpuOptimizerManager::Instance().MetaOptimize(context, &optimize_graph, device_options, graph_dumper));

  TensorDataTypes input_dtypes;
  TensorDataTypes output_dtypes;
  const tensorflow::OpRegistrationData *op_reg_data;
  NPU_CTX_REQUIRES_OK(s, lib_def->LookUp(op_name, &op_reg_data));
  tensorflow::InOutTypesForNode(ndef, op_reg_data->op_def, &input_dtypes, &output_dtypes);
  auto mutable_concrete_graph = std::make_unique<NpuMutableConcreteGraph>(op_name, input_dtypes, output_dtypes,
                                                                          NextUUID(), std::move(optimize_graph));
  NPU_CTX_REQUIRES_OK(
    s, NpuOptimizerManager::Instance().RuntimeOptimize(context, mutable_concrete_graph.get(), device_options, this,
                                                       num_inputs, inputs, graph_dumper));

  LOG(INFO) << "Concrete graph for " << op_name << " loop type " << mutable_concrete_graph->GraphLoopTypeString();
  *concrete_graph = std::move(mutable_concrete_graph);
}

/**
 * @brief: get or create spec
 * @param context: tfe context
 * @param op_name: op name
 * @param attributes: tfe op attrs
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param spec: shared point to OpExecutor
 * @param s: tf status
 */
void NpuDevice::GetOrCreateOpExecutor(TFE_Context *context, const char *op_name, const TFE_OpAttrs *attributes,
                                      int num_inputs, TFE_TensorHandle **inputs,
                                      std::shared_ptr<const npu::OpExecutor> *spec, TF_Status *s) {
  tensorflow::NodeDef ndef;
  ndef.set_op(op_name);
  npu::UnwrapAttrs(attributes)->FillAttrValueMap(ndef.mutable_attr());
  bool request_shape = false;
  GetOpExecutor(ndef, spec, request_shape);
  if (request_shape) {
    TensorShapes input_shapes;
    input_shapes.resize(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
      NPU_CTX_REQUIRES_OK(s, npu::GetTensorHandleShape(inputs[i], input_shapes[i]));
    }
    GetOpExecutor(ndef, input_shapes, spec);
  }
  if (*spec != nullptr) {
    DLOG() << "Found cached " << (*spec)->Type() << " executor for " << std::endl << ndef.DebugString();
    return;
  }
  DLOG() << "No cached op executor for " << op_name << ", start create and cache";
  *spec = OpExecutor::Create(context, this, ndef, num_inputs, inputs, s);
  NPU_REQUIRES_TFE_OK(s);
  DLOG() << "Cache " << (*spec)->Type() << " op_executor for " << ndef.DebugString() << std::endl
         << (*spec)->DebugString();
  CacheOpExecutor(*spec);
}

void NpuDevice::FallbackCPU(TFE_Context *context, const char *op_name, const TFE_OpAttrs *attributes, int num_inputs,
                            TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  DLOG() << "Start fallback executing " << op_name << " by " << underlying_device;
  TFE_Op *op(TFE_NewOp(context, op_name, status));
  NPU_REQUIRES_TFE_OK(status);
  TFE_OpAddAttrs(op, attributes);
  TFE_OpSetDevice(op, underlying_device.c_str(), status);
  ScopeTensorHandleDeleter scope_handle_deleter;
  for (int j = 0; j < num_inputs; ++j) {
    TFE_TensorHandle *input = inputs[j];
    if (npu::IsNpuTensorHandle(input)) {
      input = CopyTensorD2H(context, input, status);  // 创建完成计数为1
      scope_handle_deleter.Guard(input);
      NPU_REQUIRES_TFE_OK(status);
    }
    if (kDumpExecutionDetail) {
      const tensorflow::Tensor *tensor = nullptr;
      npu::GetTensorHandleTensor(input, &tensor);
      LOG(INFO) << "    input " << j << "  " << tensor->DebugString();
    }
    TFE_OpAddInput(op, input, status);  // add完成计数为2
    NPU_REQUIRES_TFE_OK(status);
  }

  std::vector<TFE_TensorHandle *> op_outputs(num_outputs);
  TFE_Execute(op, op_outputs.data(), &num_outputs, status);
  TFE_DeleteOp(op);
  NPU_REQUIRES_TFE_OK(status);
  for (int i = 0; i < num_outputs; ++i) {
    outputs[i] = op_outputs[i];
  }
}

/**
 * @brief: fallback cpu
 * @param spec: op spec
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param status: tf status
 */
void NpuDevice::FallbackCPU(TFE_Context *context, const tensorflow::NodeDef &ndef, int num_inputs,
                            TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  tensorflow::AttrBuilder attr_builder;
  attr_builder.Reset(ndef.op().c_str());
  attr_builder.BuildNodeDef();
  for (auto &attr : ndef.attr()) {
    attr_builder.Set(attr.first, attr.second);
  }
  FallbackCPU(context, ndef.op().c_str(), tensorflow::wrap(&attr_builder), num_inputs, inputs, num_outputs, outputs,
              status);
}

/**
 * @brief: execute
 * @param op: tfe op
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param s: tf status
 */
void NpuDevice::Execute(const TFE_Op *op, int num_outputs, TFE_TensorHandle **outputs, TF_Status *s) {
  auto context = TFE_OpGetContext(op, s);
  NPU_REQUIRES_TFE_OK(s);

  auto num_inputs = TFE_OpGetFlatInputCount(op, s);
  NPU_REQUIRES_TFE_OK(s);

  std::vector<TFE_TensorHandle *> inputs;
  for (int i = 0; i < num_inputs; i++) {
    inputs.push_back(TFE_OpGetFlatInput(op, i, s));
    NPU_REQUIRES_TFE_OK(s);
  }
  auto op_name = TFE_OpGetName(op, s);
  NPU_REQUIRES_TFE_OK(s);

  auto attributes = TFE_OpGetAttrs(op);
  DLOG() << "NPU Start executing " << op_name;

  std::shared_ptr<const npu::OpExecutor> spec;
  GetOrCreateOpExecutor(context, op_name, attributes, inputs.size(), inputs.data(), &spec, s);
  NPU_REQUIRES_TFE_OK(s);

  spec->Run(context, this, num_inputs, inputs.data(), num_outputs, outputs, s);
}

namespace {
tensorflow::Status AddVarInitToGraph(const TFE_Context *const context, std::string name, tensorflow::Tensor tensor,
                                     tensorflow::Graph *graph) {
  (void)context;
  tensorflow::Node *variable = nullptr;
  tensorflow::Node *value = nullptr;
  tensorflow::Node *assign_variable = nullptr;

  NPU_REQUIRES_OK(tensorflow::NodeBuilder(name, "VarHandleOp")
                    .Attr("container", "")
                    .Attr("shared_name", name)
                    .Attr("dtype", tensor.dtype())
                    .Attr("shape", tensor.shape())
                    .Finalize(graph, &variable));
  NPU_REQUIRES_OK(tensorflow::NodeBuilder(name + "_v", "Const")
                    .Attr("value", tensor)
                    .Attr("dtype", tensor.dtype())
                    .Finalize(graph, &value));
  NPU_REQUIRES_OK(tensorflow::NodeBuilder(name + "_op", "AssignVariableOp")
                    .Input(variable, 0)
                    .Input(value, 0)
                    .Attr("dtype", tensor.dtype())
                    .Finalize(graph, &assign_variable));

  AssembleOpDef(variable);
  AssembleOpDef(value);
  AssembleOpDef(assign_variable);

  AssembleOutputDesc(TensorShapes({kScalarShape}), {tensorflow::DT_RESOURCE}, variable);
  AssembleOutputDesc(TensorShapes({tensor.shape()}), {tensor.dtype()}, value);
  AssembleInputDesc(TensorShapes({kScalarShape, tensor.shape()}), {tensorflow::DT_RESOURCE, tensor.dtype()},
                    assign_variable);
  return tensorflow::Status::OK();
}
}  // namespace

/**
 * @brief: set npu loop size
 * @param context: tfe context
 * @param loop: loop size
 * @param status: tf status
 */
void NpuDevice::SetNpuLoopSize(TFE_Context *context, int64_t loop, TF_Status *status) {
  static std::atomic_bool initialized{false};
  static std::atomic_int64_t current_loop_size{1};
  static tensorflow::Status init_status = tensorflow::Status::OK();
  static std::uint64_t loop_var_graph_id = 0;
  const static std::string kLoopVarName = "npu_runconfig/iterations_per_loop";

  if (current_loop_size == loop) return;

  LOG(INFO) << "Set npu loop size to " << loop;

  if (!initialized.exchange(true)) {
    tensorflow::Graph graph(tensorflow::OpRegistry::Global());
    NPU_CTX_REQUIRES_OK(
      status, AddVarInitToGraph(context, "npu_runconfig/loop_cond", tensorflow::Tensor(tensorflow::int64(0)), &graph));
    NPU_CTX_REQUIRES_OK(
      status, AddVarInitToGraph(context, "npu_runconfig/one", tensorflow::Tensor(tensorflow::int64(1)), &graph));
    NPU_CTX_REQUIRES_OK(
      status, AddVarInitToGraph(context, "npu_runconfig/zero", tensorflow::Tensor(tensorflow::int64(0)), &graph));

    RunGeGraphPin2CpuAnonymous(context, "set_npu_loop_conditions", graph.ToGraphDefDebug(), 0, nullptr, 0, nullptr,
                               status);
    NPU_REQUIRES_TFE_OK(status);

    tensorflow::Node *variable;
    tensorflow::Node *arg;
    tensorflow::Node *assign_variable;

    tensorflow::Graph graph2(tensorflow::OpRegistry::Global());

    NPU_CTX_REQUIRES_OK(status, tensorflow::NodeBuilder(kLoopVarName, "VarHandleOp")
                                  .Attr("container", "")
                                  .Attr("shared_name", kLoopVarName)
                                  .Attr("dtype", tensorflow::DT_INT64)
                                  .Attr("shape", kScalarShape)
                                  .Finalize(&graph2, &variable));
    NPU_CTX_REQUIRES_OK(status, tensorflow::NodeBuilder(kLoopVarName + "_v", "_Arg")
                                  .Attr("T", tensorflow::DT_INT64)
                                  .Attr("index", 0)
                                  .Finalize(&graph2, &arg));
    NPU_CTX_REQUIRES_OK(status, tensorflow::NodeBuilder(kLoopVarName + "_op", "AssignVariableOp")
                                  .Input(variable, 0)
                                  .Input(arg, 0)
                                  .Attr("dtype", tensorflow::DT_INT64)
                                  .Finalize(&graph2, &assign_variable));

    AssembleOpDef(variable);
    AssembleOpDef(arg);
    AssembleOpDef(assign_variable);

    AssembleOutputDesc(TensorShapes({kScalarShape}), {tensorflow::DT_RESOURCE}, variable);
    AssembleOutputDesc(TensorShapes({kScalarShape}), {tensorflow::DT_INT64}, arg);
    AssembleInputDesc(TensorShapes({kScalarShape, kScalarShape}), {tensorflow::DT_RESOURCE, tensorflow::DT_INT64},
                      assign_variable);

    loop_var_graph_id = AddGeGraph(context, "set_loop_var", graph2.ToGraphDefDebug(), status);
    init_status = status->status;
    NPU_REQUIRES_TFE_OK(status);
  }

  status->status = init_status;
  NPU_REQUIRES_TFE_OK(status);

  std::vector<TFE_TensorHandle *> inputs(1);
  inputs[0] =
    tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(tensorflow::Tensor(tensorflow::int64(loop - 1))));

  RunGeGraphPin2Cpu(context, loop_var_graph_id, inputs.size(), inputs.data(), {}, 0, nullptr, status);

  if (TF_GetCode(status) == TF_OK) {
    current_loop_size = loop;
  }
  for (auto handle : inputs) {
    TFE_DeleteTensorHandle(handle);
  }
}

/**
 * @brief: run ge graph async
 * @param context: tfe context
 * @param graph_id: graph id
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param pin_to_npu: if pin to npu or not
 * @param output_types: tensor data types
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param done: done callback
 * @param status: tf status
 */
void NpuDevice::RunGeGraphAsync(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                                bool pin_to_npu, const TensorDataTypes &output_types, int num_outputs,
                                TFE_TensorHandle **outputs, DoneCallback done, TF_Status *status) {
  std::vector<ge::Tensor> ge_inputs;
  TransTfInputs2GeInputs(num_inputs, inputs, status, ge_inputs);
  NPU_REQUIRES_TFE_OK(status);

  DLOG() << "Ge graph " << graph_id << " input info";

  auto ge_callback = [this, context, status, done, pin_to_npu, output_types, num_outputs, outputs, graph_id](
                       ge::Status s, std::vector<ge::Tensor> &ge_outputs) {
    DLOG() << "Graph engine callback with status:" << s;
    if (s == ge::END_OF_SEQUENCE) {
      done(tensorflow::errors::OutOfRange("Graph engine process graph ", graph_id, " reach end of sequence"));
      return;
    } else if (s != ge::SUCCESS) {
      std::string err_msg = ge::GEGetErrorMsg();
      if (err_msg.empty()) {
        err_msg = "<unknown error> code:" + std::to_string(s);
      }
      done(tensorflow::errors::Internal("Graph engine process graph failed: ", err_msg));
      return;
    } else if (ge_outputs.size() != static_cast<std::size_t>(num_outputs)) {
      done(tensorflow::errors::Internal("Graph engine process graph succeed but output num ", ge_outputs.size(),
                                        " mismatch with expected ", num_outputs));
      return;
    }

    DLOG() << "Ge graph " << graph_id << " output info";
    for (size_t i = 0; i < ge_outputs.size(); i++) {
      auto &ge_tensor = ge_outputs[i];
      std::vector<tensorflow::int64> dims;
      for (auto dim_size : ge_tensor.GetTensorDesc().GetShape().GetDims()) {
        dims.push_back(dim_size);
      }
      tensorflow::TensorShape shape;
      tensorflow::Status tf_status = tensorflow::TensorShapeUtils::MakeShape(dims.data(), dims.size(), &shape);
      if (!tf_status.ok()) {
        done(tensorflow::errors::Internal("Graph engine process graph succeed but output ", i, " dims invalid ",
                                          VecToString(ge_tensor.GetTensorDesc().GetShape().GetDims()), " ",
                                          tf_status.error_message()));
        return;
      }
      DLOG() << "    output " << i << " ge type enum " << ge_tensor.GetTensorDesc().GetDataType() << " tf type "
             << tensorflow::DataTypeString(output_types[i]) << shape.DebugString();

      const static int64_t kTensorAlignBytes = 64;
      if (reinterpret_cast<uintptr_t>(ge_tensor.GetData()) % kTensorAlignBytes == 0) {
        DLOG() << "Zero copy ge tensor " << reinterpret_cast<uintptr_t>(ge_tensor.GetData()) << " as aligned with "
               << kTensorAlignBytes << " bytes";
        size_t ge_tensor_total_bytes = ge_tensor.GetSize();
        tensorflow::Allocator *allocator =
          NpuHostFixedAllocator<uint8_t[], std::function<void(uint8_t *)>>::Create(std::move(ge_tensor.ResetData()));
        tensorflow::Tensor cpu_tensor(allocator, output_types[i], shape);
        if (ge_tensor_total_bytes != cpu_tensor.TotalBytes()) {
          done(tensorflow::errors::Internal("Graph engine process graph succeed but output ", i, " total bytes ",
                                            ge_tensor_total_bytes, " mismatch with expected ",
                                            cpu_tensor.TotalBytes()));
          return;
        }
        outputs[i] = tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(cpu_tensor));
      } else {
        DLOG() << "Skip zero copy as ge tensor " << reinterpret_cast<uintptr_t>(ge_tensor.GetData())
               << " not aligned with " << kTensorAlignBytes << " bytes";
        tensorflow::Tensor cpu_tensor(output_types[i], shape);
        if (ge_tensor.GetSize() != cpu_tensor.TotalBytes()) {
          done(tensorflow::errors::Internal("Graph engine process graph succeed but output ", i, " total bytes ",
                                            ge_tensor.GetSize(), " mismatch with expected ", cpu_tensor.TotalBytes()));
          return;
        }
        memcpy(cpu_tensor.data(), ge_tensor.GetData(), ge_tensor.GetSize());
        outputs[i] = tensorflow::wrap(tensorflow::TensorHandle::CreateLocalHandle(cpu_tensor));
      }

      if (pin_to_npu) {
        TFE_TensorHandle *handle = outputs[i];
        outputs[i] = CopyTensorH2D(context, handle, status);
        TFE_DeleteTensorHandle(handle);
        if (TF_GetCode(status) != TF_OK) {
          done(tensorflow::Status(status->status.code(),
                                  std::string("Graph engine process graph succeed but copy output ") +
                                    std::to_string(i) + " to npu failed " + status->status.error_message()));
          return;
        }
      }
    }
    done(tensorflow::Status::OK());
  };
  NPU_CTX_REQUIRES_GE_OK(status, "NPU Schedule graph to graph engine",
                         ge_session_->RunGraphAsync(graph_id, ge_inputs, ge_callback));
}

void NpuDevice::TransTfInputs2GeInputs(int num_inputs, TFE_TensorHandle **inputs, TF_Status *status,
                                       std::vector<ge::Tensor> &ge_inputs) const {
  for (int i = 0; i < num_inputs; i++) {
    const tensorflow::Tensor *tensor = nullptr;
    npu::GetTensorHandleTensor(inputs[i], &tensor);

    const static std::shared_ptr<domi::ModelParser> parser =
      domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
    if (parser == nullptr) {
      status->status = tensorflow::errors::Internal("NPU Create new tensorflow model parser failed");
      return;
    }
    ge::DataType ge_type = parser->ConvertToGeDataType(static_cast<uint32_t>(tensor->dtype()));
    NPU_CTX_REQUIRES(
      status, ge_type != ge::DT_UNDEFINED,
      tensorflow::errors::InvalidArgument("Failed map tensorflow data type ",
                                          tensorflow::DataTypeString(tensor->dtype()), " to ge data type"));
    ge::Tensor input;
    std::vector<int64_t> dims;
    for (auto dim_size : tensor->shape().dim_sizes()) {
      dims.emplace_back(dim_size);
    }
    input.SetTensorDesc(ge::TensorDesc(ge::Shape(dims), ge::FORMAT_ND, ge_type));
    input.SetData(static_cast<const uint8_t *>(tensor->data()), tensor->TotalBytes());
    ge_inputs.emplace_back(input);
    DLOG() << "    input " << i << " ge enum " << ge_type << " tf type " << tensorflow::DataTypeString(tensor->dtype())
           << VecToString(dims);
  }
}

/**
 * @brief: add ge graph inner
 * @param context: tfe context
 * @param graph_id: graph id
 * @param name: name
 * @param def: tensorflow graph def
 * @param loop: is loop or not
 * @param status: tf status
 */
uint64_t NpuDevice::AddGeGraphInner(TFE_Context *context, uint64_t graph_id, const std::string &name,
                                    const tensorflow::GraphDef &def, bool loop, TF_Status *status,
                                    const std::map<std::string, std::string> &options) {
  if (def.node_size() == 0) {
    return kEmptyGeGraphId;
  }
  ge::Graph ge_graph;
  NPU_CTX_REQUIRES_OK_RETURN(status, TransTfGraph2GeGraph(context, name, def, ge_graph), graph_id);
  ge_graph.SetNeedIteration(loop);

  if (kDumpExecutionDetail && !options.empty()) {
    LOG(INFO) << "Add ge graph " << graph_id << " with options:";
    for (auto &option : options) {
      LOG(INFO) << "  " << option.first << ":" << option.second;
    }
  }
  NPU_CTX_REQUIRES_GE_OK_RETURN(status, "Graph engine Add graph", GeSession()->AddGraph(graph_id, ge_graph, options),
                                graph_id);
  return graph_id;
}

tensorflow::Status NpuDevice::TransTfGraph2GeGraph(TFE_Context *context, const std::string &name,
                                                   const tensorflow::GraphDef &def, ge::Graph &ge_graph) const {
  auto ge_compute_graph = std::make_shared<ge::ComputeGraph>(name);
  std::shared_ptr<domi::ModelParser> parser =
    domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  NPU_REQUIRES(parser != nullptr, tensorflow::errors::Internal("NPU Create new tensorflow model parser failed"));

  auto request_subgraph = [name, context](const std::string &fn) -> std::string {
    DLOG() << "Tensorflow model parser requesting subgraph " << fn << " for ge graph " << name;
    tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
    const tensorflow::FunctionDef *fdef = lib_def->Find(fn);
    if (fdef == nullptr) {
      return "";
    }
    std::unique_ptr<tensorflow::FunctionBody> fbody;
    auto tf_status = FunctionDefToBodyHelper(*fdef, tensorflow::AttrSlice{}, lib_def, &fbody);
    if (!tf_status.ok()) {
      LOG(ERROR) << "Failed trans function body to graph " << tf_status.ToString();
      return "";
    }

    tensorflow::ProcessFunctionLibraryRuntime *pflr = npu::UnwrapCtx(context)->pflr();
    tensorflow::FunctionLibraryRuntime *flr = pflr->GetFLR("/job:localhost/replica:0/task:0/device:CPU:0");

    std::unique_ptr<tensorflow::Graph> graph = std::make_unique<tensorflow::Graph>(lib_def);
    CopyGraph(*fbody->graph, graph.get());
    NpuCustomizedOptimizeGraph(*flr, &graph);

    PruneGraphByFunctionSignature(*fdef, *graph.get(), true);

    AssembleParserAddons(context, graph.get());

    if (kDumpExecutionDetail || kDumpGraph) {
      WriteTextProto(tensorflow::Env::Default(), name + "_subgraph_" + fn + ".pbtxt", graph->ToGraphDefDebug());
    }
    return graph->ToGraphDefDebug().SerializeAsString();
  };

  NPU_REQUIRES(
    parser->ParseProtoWithSubgraph(def.SerializeAsString(), request_subgraph, ge_compute_graph) == ge::SUCCESS,
    tensorflow::errors::Internal("NPU Parse tensorflow model failed"));

  if (ge_compute_graph->GetAllNodesSize() != 0) {
    ge_graph = ge::GraphUtils::CreateGraphFromComputeGraph(ge_compute_graph);
  }

  return tensorflow::Status::OK();
}

/**
 * @brief: add ge graph
 * @param context: tfe context
 * @param graph_id: graph id
 * @param name: name
 * @param def: tensorflow graph def
 * @param status: tf status
 */
uint64_t NpuDevice::AddGeGraph(TFE_Context *context, uint64_t graph_id, const std::string &name,
                               const tensorflow::GraphDef &def, TF_Status *status,
                               const std::map<std::string, std::string> &options) {
  return AddGeGraphInner(context, graph_id, name, def, false, status, options);
}

/**
 * @brief: add ge graph
 * @param context: tfe context
 * @param name: name
 * @param def: tensorflow graph def
 * @param status: tf status
 */
uint64_t NpuDevice::AddGeGraph(TFE_Context *context, const std::string &name, const tensorflow::GraphDef &def,
                               TF_Status *status, const std::map<std::string, std::string> &options) {
  uint64_t graph_id = NextUUID();
  return AddGeGraph(context, graph_id, name, def, status, options);
}

/**
 * @brief: remove ge graph
 * @param context: tfe context
 * @param graph_id: graph id
 * @param status: tf status
 */
void NpuDevice::RemoveGeGraph(const TFE_Context *const context, uint64_t graph_id, TF_Status *status) {
  (void)context;
  NPU_CTX_REQUIRES_GE_OK(status, "Graph engine remove graph", GeSession()->RemoveGraph(graph_id));
}

/**
 * @brief: run ge graph
 * @param context: tfe context
 * @param graph_id: graph id
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param pin_to_npu: if pin to npu or not
 * @param output_types: tensor data types
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param status: tf status
 */
void NpuDevice::RunGeGraph(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                           bool pin_to_npu, const TensorDataTypes &output_types, int num_outputs,
                           TFE_TensorHandle **outputs, TF_Status *status) {
  tensorflow::Notification notification;
  auto done = [status, &notification](tensorflow::Status s) {
    status->status = std::move(s);
    notification.Notify();
  };
  RunGeGraphAsync(context, graph_id, num_inputs, inputs, pin_to_npu, output_types, num_outputs, outputs, done, status);
  NPU_REQUIRES_TFE_OK(status);
  notification.WaitForNotification();
}

/**
 * @brief: run ge graph pin to cpu async
 * @param context: tfe context
 * @param graph_id: graph id
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param output_types: tensor data types
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param done: done callback
 * @param status: tf status
 */
void NpuDevice::RunGeGraphPin2CpuAsync(TFE_Context *context, uint64_t graph_id, int num_inputs,
                                       TFE_TensorHandle **inputs, const TensorDataTypes &output_types, int num_outputs,
                                       TFE_TensorHandle **outputs, DoneCallback done, TF_Status *status) {
  RunGeGraphAsync(context, graph_id, num_inputs, inputs, false, output_types, num_outputs, outputs, std::move(done),
                  status);
}

/**
 * @brief: run ge graph pin to npu async
 * @param context: tfe context
 * @param graph_id: graph id
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param output_types: tensor data types
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param done: done callback
 * @param status: tf status
 */
void NpuDevice::RunGeGraphPin2NpuAsync(TFE_Context *context, uint64_t graph_id, int num_inputs,
                                       TFE_TensorHandle **inputs, const TensorDataTypes &output_types, int num_outputs,
                                       TFE_TensorHandle **outputs, DoneCallback done, TF_Status *status) {
  RunGeGraphAsync(context, graph_id, num_inputs, inputs, true, output_types, num_outputs, outputs, std::move(done),
                  status);
}

/**
 * @brief: run ge graph pin to cpu
 * @param context: tfe context
 * @param graph_id: graph id
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param output_types: tensor data types
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param status: tf status
 */
void NpuDevice::RunGeGraphPin2Cpu(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                                  const TensorDataTypes &output_types, int num_outputs, TFE_TensorHandle **outputs,
                                  TF_Status *status) {
  RunGeGraph(context, graph_id, num_inputs, inputs, false, output_types, num_outputs, outputs, status);
}

/**
 * @brief: run ge graph pin to npu
 * @param context: tfe context
 * @param graph_id: graph id
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param output_types: tensor data types
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param status: tf status
 */
void NpuDevice::RunGeGraphPin2Npu(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                                  const TensorDataTypes &output_types, int num_outputs, TFE_TensorHandle **outputs,
                                  TF_Status *status) {
  RunGeGraph(context, graph_id, num_inputs, inputs, true, output_types, num_outputs, outputs, status);
}

/**
 * @brief: run ge graph anonymous
 * @param context: tfe context
 * @param name: name
 * @param gdef: tensorflow graph def
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param pin_to_npu: if pip to npu or not
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param status: tf status
 */
void NpuDevice::RunGeGraphAnonymous(TFE_Context *context, const std::string &name, const tensorflow::GraphDef &gdef,
                                    int num_inputs, TFE_TensorHandle **inputs, bool pin_to_npu, int num_outputs,
                                    TFE_TensorHandle **outputs, TF_Status *status) {
  uint64_t graph_id = AddGeGraph(context, name, gdef, status);
  NPU_REQUIRES_TFE_OK(status);

  std::map<int, tensorflow::DataType> indexed_types;

  for (const auto &node : gdef.node()) {
    if (node.op() == "_Retval") {
      tensorflow::DataType type;
      tensorflow::GetNodeAttr(node, "T", &type);
      int index;
      tensorflow::GetNodeAttr(node, "index", &index);
      indexed_types[index] = type;
    }
  }
  TensorDataTypes types;
  for (auto indexed_type : indexed_types) {
    types.emplace_back(indexed_type.second);
  }

  RunGeGraph(context, graph_id, num_inputs, inputs, pin_to_npu, types, num_outputs, outputs, status);
  NPU_REQUIRES_TFE_OK(status);

  RemoveGeGraph(context, graph_id, status);
  NPU_REQUIRES_TFE_OK(status);
}

/**
 * @brief: run ge graph pin to cpu anonymous
 * @param context: tfe context
 * @param name: name
 * @param gdef: tensorflow graph def
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param status: tf status
 */
void NpuDevice::RunGeGraphPin2CpuAnonymous(TFE_Context *context, const std::string &name,
                                           const tensorflow::GraphDef &gdef, int num_inputs, TFE_TensorHandle **inputs,
                                           int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  RunGeGraphAnonymous(context, name, gdef, num_inputs, inputs, false, num_outputs, outputs, status);
}

/**
 * @brief: run ge graph pin to npu anonymous
 * @param context: tfe context
 * @param name: name
 * @param gdef: tensorflow graph def
 * @param num_inputs: number of inputs
 * @param inputs: tfe tensor handle inputs
 * @param num_outputs: number of outputs
 * @param outputs: tfe tensor handle outputs
 * @param status: tf status
 */
void NpuDevice::RunGeGraphPin2NpuAnonymous(TFE_Context *context, const std::string &name,
                                           const tensorflow::GraphDef &gdef, int num_inputs, TFE_TensorHandle **inputs,
                                           int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  RunGeGraphAnonymous(context, name, gdef, num_inputs, inputs, true, num_outputs, outputs, status);
}

/**
 * @brief: get cached op executor
 * @param ndef: tensorflow node def
 * @param spec: shared point to OpExecutor
 * @param request_shape: if request shape or not
 */
void NpuDevice::GetOpExecutor(const tensorflow::NodeDef &ndef, std::shared_ptr<const npu::OpExecutor> *spec,
                              bool &request_shape) {
  *spec = nullptr;
  const auto &op = ndef.op();
  if (cached_func_specs_.find(op) == cached_func_specs_.end()) {
    HashKey attr_hash = Hash(ndef);
    request_shape = (cached_op_specs_.count(op) > 0) && (cached_op_specs_[op].count(attr_hash) > 0);
    return;
  }
  *spec = cached_func_specs_[op];
}

void NpuDevice::CacheOpExecutor(std::shared_ptr<const OpExecutor> spec) {
  if (spec->GetCacheStrategy() == CacheStrategy::BY_OP_NAME) {
    cached_func_specs_.emplace(std::make_pair(spec->Op(), spec));
  } else {
    cached_op_specs_[spec->Op()][Hash(spec->NodeDef())][Hash(spec->InputShapes())] = spec;
  }
}

/**
 * @brief: get cached op executor
 * @param ndef: tensorflow node def
 * @param shapes: tensor shapes
 * @param spec: op executor
 */
void NpuDevice::GetOpExecutor(const tensorflow::NodeDef &ndef, const TensorShapes &shapes,
                              std::shared_ptr<const npu::OpExecutor> *spec) {
  *spec = nullptr;
  bool request_shape = false;
  GetOpExecutor(ndef, spec, request_shape);
  if (*spec != nullptr) {
    return;
  }
  if (!request_shape) {
    return;
  }
  HashKey attr_hash = Hash(ndef);
  HashKey shape_hash = Hash(shapes);
  const auto &op = ndef.op();
  if (cached_op_specs_.count(op) > 0 && cached_op_specs_[op].count(attr_hash) > 0 &&
      cached_op_specs_[op][attr_hash].count(shape_hash) > 0) {
    *spec = cached_op_specs_[op][attr_hash][shape_hash];
  }
}

tensorflow::Status NpuDevice::LoadSupportedOps(std::unordered_set<std::string> &ops) {
  NPU_REQUIRES(!kOppInstallPath.empty(),
               tensorflow::errors::Internal("No specific opp install path, set it by ASCEND_OPP_PATH env and retry"));
  NPU_REQUIRES_OK(tensorflow::Env::Default()->IsDirectory(kOppInstallPath));
  auto supported_ops_json = kOppInstallPath + "/framework/built-in/tensorflow/npu_supported_ops.json";
  NPU_REQUIRES_OK(tensorflow::Env::Default()->FileExists(supported_ops_json));
  std::ifstream fs(supported_ops_json, std::ifstream::in);
  NPU_REQUIRES(fs.is_open(), tensorflow::errors::Internal("Failed open config file ", supported_ops_json));
  nlohmann::json root;
  try {
    fs >> root;
  } catch (nlohmann::json::exception &e) {
    fs.close();
    return tensorflow::errors::Internal("Parse json from ", supported_ops_json, " failed ", e.what());
  }
  for (auto iter = root.begin(); iter != root.end(); iter++) {
    DLOG() << "Npu supported op " << (iter.key());
    ops.insert(iter.key());
  }
  fs.close();
  const static std::vector<std::string> kAddonOps{"IteratorV2", "IteratorGetNext"};
  ops.insert(kAddonOps.cbegin(), kAddonOps.cend());
  return tensorflow::Status::OK();
}

/**
 * @brief: if op is supported or not
 * @param op: op type
 */
bool NpuDevice::Supported(const std::string &op) const {
  const static std::unordered_set<std::string> kUnsupportedOps = {"_EagerConst"};
  return npu_supported_ops_.count(op) > 0 || (npu_supported_ops_.empty() && kUnsupportedOps.count(op) == 0);
}

bool NpuDevice::IsNpuSpecificOp(const std::string &op) const {
  // Ops by npu may want to run on cpu, we must check npu supported here
  return global::g_npu_specify_ops.count(op) > 0 && Supported(op);
}

/**
 * @brief: is supported resource generator or not
 * @param op: op type
 */
bool NpuDevice::SupportedResourceGenerator(const std::string &op) const {
  const static std::unordered_set<std::string> kSupportedOps = {"VarHandleOp"};
  return kSupportedOps.count(op) > 0;
}

void NpuDevice::RecordResourceGeneratorDef(const tensorflow::ResourceHandle &key,
                                           std::shared_ptr<ResourceGenerator> src) {
  tensorflow::mutex_lock lk(mutex_);
  device_resources_.emplace(key, src);
}

void NpuDevice::GetResourceGeneratorDef(const tensorflow::ResourceHandle &key,
                                        std::shared_ptr<ResourceGenerator> *src) {
  tensorflow::tf_shared_lock lk(mutex_);
  auto iter = device_resources_.find(key);
  if (iter != device_resources_.end()) {
    *src = iter->second;
  }
}

/**
 * @brief: record iterator mirror
 * @param src: tensorflow resource handle
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 */
void NpuDevice::RecordIteratorMirror(const tensorflow::ResourceHandle &src, const TensorPartialShapes &shapes,
                                     const TensorDataTypes &types) {
  tensorflow::mutex_lock lk(mutex_);
  iterator_mirrors_.emplace(src, std::make_pair(shapes, types));
}

/**
 * @brief: mirrored iterator
 * @param src: tensorflow resource handle
 */
bool NpuDevice::MirroredIterator(const tensorflow::ResourceHandle &src) {
  tensorflow::tf_shared_lock lk(mutex_);
  return iterator_mirrors_.find(src) != iterator_mirrors_.end();
}

bool NpuDevice::Mirrored(const tensorflow::ResourceHandle &src) {
  // 可能后续还有其他需要mirror的资源，外层判断资源兼容时务必使用这个接口
  tensorflow::tf_shared_lock lk(mutex_);
  return iterator_mirrors_.find(src) != iterator_mirrors_.end();
}

/**
 * @brief: get mirrored iterator shapes and types
 * @param src: tensorflow resource handle
 * @param shapes: tensor partial shapes
 * @param types: tensor data types
 */
tensorflow::Status NpuDevice::GetMirroredIteratorShapesAndTypes(const tensorflow::ResourceHandle &src,
                                                                TensorPartialShapes &shapes, TensorDataTypes &types) {
  tensorflow::tf_shared_lock lk(mutex_);
  const decltype(iterator_mirrors_)::const_iterator iter = iterator_mirrors_.find(src);
  if (iter == iterator_mirrors_.cend()) {
    return tensorflow::errors::Internal("Resource ", src.DebugString(), " has not been mirrored");
  }
  shapes.assign(iter->second.first.cbegin(), iter->second.first.cend());
  types.assign(iter->second.second.cbegin(), iter->second.second.cend());
  return tensorflow::Status::OK();
}
}  // namespace npu