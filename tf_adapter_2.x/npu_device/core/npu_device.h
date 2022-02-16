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

#ifndef NPU_DEVICE_CORE_NPU_DEVICE_H
#define NPU_DEVICE_CORE_NPU_DEVICE_H

#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"

#include "ge/ge_api.h"

#include "npu_op_executor.h"
#include "npu_dp.h"
#include "npu_types.h"
#include "npu_unwrap.h"
#include "npu_utils.h"

namespace npu {
class NpuDevice {
  using HashKey = uint64_t;

  using ShapeTasks = std::map<HashKey, std::shared_ptr<const OpExecutor>>;
  using AttrTasks = std::map<HashKey, ShapeTasks>;
  using CachedOpSpecs = std::map<const std::string, AttrTasks>;
  using CachedFuncSpecs = std::map<const std::string, std::shared_ptr<const OpExecutor>>;
  using DoneCallback = std::function<void(tensorflow::Status)>;

 public:
  static std::string CreateDevice(const char *name, int device_index, const std::map<std::string, std::string> &options,
                                  NpuDevice **device);

  static void DeleteDevice(void *device);

  void ReleaseResource();

  tensorflow::Status InferShape(TFE_Context *context, const tensorflow::OpRegistrationData *op_reg_data,
                                const tensorflow::NodeDef &ndef, int num_inputs, TFE_TensorHandle **inputs,
                                TensorPartialShapes &shapes, bool &requested_input_value) const;

  void GetConcreteGraph(TFE_Context *context, const tensorflow::NodeDef &ndef, int num_inputs,
                        TFE_TensorHandle **inputs, std::unique_ptr<NpuConcreteGraph> *concrete_graph, TF_Status *s);

  tensorflow::Status ValidateOutputTypes(const TensorDataTypes &data_types) const;
  tensorflow::Status ValidateInputTypes(const TensorDataTypes &data_types) const;

  TFE_TensorHandle *NewDeviceTensorHandle(TFE_Context *context, ge::Format fmt, const tensorflow::TensorShape &shape,
                                          tensorflow::DataType type, TF_Status *status);

  TFE_TensorHandle *NewDeviceResourceHandle(TFE_Context *context, const tensorflow::TensorShape &shape,
                                            TF_Status *status);

  TFE_TensorHandle *CopyTensorD2H(TFE_Context *context, TFE_TensorHandle *tensor, TF_Status *status) const;

  TFE_TensorHandle *CopyTensorH2D(TFE_Context *context, TFE_TensorHandle *tensor, TF_Status *status);

  TFE_TensorHandle *CopyTensorH2D(TFE_Context *context, TFE_TensorHandle *tensor, ge::Format fmt, TF_Status *status);

  void GetOrCreateOpExecutor(TFE_Context *context, const char *op_name, const TFE_OpAttrs *attributes, int num_inputs,
                             TFE_TensorHandle **inputs, std::shared_ptr<const OpExecutor> *spec, TF_Status *s);

  void FallbackCPU(TFE_Context *context, const char *op_name, const TFE_OpAttrs *attributes, int num_inputs,
                   TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs, TF_Status *status);

  void FallbackCPU(TFE_Context *context, const tensorflow::NodeDef &ndef, int num_inputs, TFE_TensorHandle **inputs,
                   int num_outputs, TFE_TensorHandle **outputs, TF_Status *status);

  // NPU Device对外的顶层方法
  void Execute(const TFE_Op *op, int num_outputs, TFE_TensorHandle **outputs, TF_Status *s);

  void SetNpuLoopSize(TFE_Context *context, int64_t loop, TF_Status *status);

  void RunGeGraphAnonymous(TFE_Context *context, const std::string &name, const tensorflow::GraphDef &gdef,
                           int num_inputs, TFE_TensorHandle **inputs, bool pin_to_npu, int num_outputs,
                           TFE_TensorHandle **outputs, TF_Status *status);

  void RunGeGraphPin2CpuAnonymous(TFE_Context *context, const std::string &name, const tensorflow::GraphDef &gdef,
                                  int num_inputs, TFE_TensorHandle **inputs, int num_outputs,
                                  TFE_TensorHandle **outputs, TF_Status *status);

  void RunGeGraphPin2NpuAnonymous(TFE_Context *context, const std::string &name, const tensorflow::GraphDef &gdef,
                                  int num_inputs, TFE_TensorHandle **inputs, int num_outputs,
                                  TFE_TensorHandle **outputs, TF_Status *status);

  uint64_t AddGeGraph(TFE_Context *context, const std::string &name, const tensorflow::GraphDef &def,
                      TF_Status *status);

  uint64_t AddGeGraph(TFE_Context *context, uint64_t graph_id, const std::string &name, const tensorflow::GraphDef &def,
                      TF_Status *status);

  uint64_t AddGeGraphInner(TFE_Context *context, uint64_t graph_id, const std::string &name,
                           const tensorflow::GraphDef &def, bool loop, TF_Status *status);

  void RemoveGeGraph(TFE_Context *context, uint64_t graph_id, TF_Status *status);

  void RunGeGraph(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs, bool pin_to_npu,
                  const TensorDataTypes &output_types, int num_outputs, TFE_TensorHandle **outputs, TF_Status *status);

  void RunGeGraphPin2Cpu(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                         const TensorDataTypes &output_types, int num_outputs, TFE_TensorHandle **outputs,
                         TF_Status *status);

  void RunGeGraphPin2Npu(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                         const TensorDataTypes &output_types, int num_outputs, TFE_TensorHandle **outputs,
                         TF_Status *status);

  void RunGeGraphAsync(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                       bool pin_to_npu, const TensorDataTypes &output_types, int num_outputs,
                       TFE_TensorHandle **outputs, DoneCallback done, TF_Status *status);

  void RunGeGraphPin2CpuAsync(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                              const TensorDataTypes &output_types, int num_outputs, TFE_TensorHandle **outputs,
                              DoneCallback done, TF_Status *status);

  void RunGeGraphPin2NpuAsync(TFE_Context *context, uint64_t graph_id, int num_inputs, TFE_TensorHandle **inputs,
                              const TensorDataTypes &output_types, int num_outputs, TFE_TensorHandle **outputs,
                              DoneCallback done, TF_Status *status);

  void GetOpExecutor(const tensorflow::NodeDef &ndef, std::shared_ptr<const OpExecutor> *spec, bool &request_shape);

  void GetOpExecutor(const tensorflow::NodeDef &ndef, const TensorShapes &shapes,
                     std::shared_ptr<const OpExecutor> *spec);

  void CacheOpExecutor(std::shared_ptr<const OpExecutor> spec);

  bool Supported(const std::string &op) const;
  bool SupportedResourceGenerator(const std::string &op) const;

  void RecordResourceGeneratorDef(const tensorflow::ResourceHandle &key, std::shared_ptr<ResourceGenerator> src);
  void GetResourceGeneratorDef(const tensorflow::ResourceHandle &key, std::shared_ptr<ResourceGenerator> *src);

  void RecordIteratorMirror(const tensorflow::ResourceHandle &src, const TensorPartialShapes &shapes,
                            const TensorDataTypes &types);

  bool Mirrored(const tensorflow::ResourceHandle &src);

  bool MirroredIterator(const tensorflow::ResourceHandle &src);

  void CreateIteratorProvider(TFE_Context *context, const tensorflow::Tensor *tensor, std::vector<int> device_ids,
                              TF_Status *status);

  std::shared_ptr<IteratorResourceProvider> GetIteratorProvider(TFE_Context *context,
                                                                const tensorflow::ResourceHandle &resource);

  tensorflow::Status GetMirroredIteratorShapesAndTypes(const tensorflow::ResourceHandle &src,
                                                       TensorPartialShapes &shapes, TensorDataTypes &types);

  ge::Session *GeSession() { return ge_session_; }

  tensorflow::CancellationManager *CancellationManager() { return cancellation_manager_.get(); }

  int device_id;
  tensorflow::string device_name;
  tensorflow::string underlying_device;
  std::map<std::string, std::string> device_options;

 private:
  NpuDevice() = default;
  ~NpuDevice() = default;
  TF_DISALLOW_COPY_AND_ASSIGN(NpuDevice);

  static HashKey Hash(const TensorDataTypes &types) {
    if (types.empty()) {
      return 0;
    }
    HashKey hash = tensorflow::Hash64(tensorflow::DataTypeString(types[0]));
    for (size_t i = 1; i < types.size(); i++) {
      hash = tensorflow::Hash64Combine(hash, tensorflow::Hash64(tensorflow::DataTypeString(types[i])));
    }
    return hash;
  }
  static HashKey Hash(const TensorShapes &shapes) {
    if (shapes.empty()) {
      return 0;
    }
    HashKey hash = tensorflow::Hash64(shapes[0].DebugString());
    for (size_t i = 1; i < shapes.size(); i++) {
      hash = tensorflow::Hash64Combine(hash, tensorflow::Hash64(shapes[i].DebugString()));
    }
    return hash;
  }
  static HashKey Hash(const TFE_OpAttrs *attributes) {
    tensorflow::AttrValueMap attrs;
    UnwrapAttrs(attributes)->FillAttrValueMapWithoutDefaults(&attrs);
    if (attrs.empty()) {
      return 0;
    }
    auto iter = attrs.begin();
    HashKey hash = tensorflow::Hash64(iter->second.DebugString());
    iter++;
    while (iter != attrs.end()) {
      hash = tensorflow::Hash64Combine(hash, tensorflow::Hash64(iter->second.DebugString()));
      iter++;
    }
    return hash;
  }

  static HashKey Hash(const tensorflow::NodeDef &ndef) { return tensorflow::Hash64(ndef.DebugString()); }

  ge::Session *ge_session_;
  std::unique_ptr<tensorflow::CancellationManager> cancellation_manager_;
  CachedOpSpecs cached_op_specs_;
  CachedFuncSpecs cached_func_specs_;
  std::map<tensorflow::ResourceHandle, std::shared_ptr<ResourceGenerator>, ResourceCompare> device_resources_;
  std::map<tensorflow::ResourceHandle, std::pair<TensorPartialShapes, TensorDataTypes>, ResourceCompare>
    iterator_mirrors_;
  std::map<tensorflow::ResourceHandle, std::shared_ptr<IteratorResourceProvider>, ResourceCompare> iterator_providers_;
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_DEVICE_H
