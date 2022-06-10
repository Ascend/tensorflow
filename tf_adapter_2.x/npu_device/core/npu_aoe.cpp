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

#include "npu_aoe.h"

#include <dlfcn.h>

namespace npu {
tensorflow::Status NpuAoe::RunAoeTuning(NpuDevice *device, TFE_Context *context,
                                        std::vector<TFE_TensorHandle *> &inputs, TF_Status *status) {
  DLOG() << "Start to tune ge graph id: " << graph_id_ << ", name: " << name_;
  void *handle = dlopen("libaoe_tuning.so", RTLD_NOW);
  NPU_REQUIRES(handle != nullptr, tensorflow::errors::Internal("libaoe_tuning.so dlopen failed"));

  if (!AoeTuningInit(device, handle).ok()) {
    (void)dlclose(handle);
    return tensorflow::errors::Internal("aoe tuning init failed");
  }

  if (!CallAoeFuncToTuning(device, context, inputs, status).ok()) {
    (void)dlclose(handle);
    return tensorflow::errors::Internal("call aoe func to tuning failed");
  }

  if (!AoeTuningFinalize().ok()) {
    (void)dlclose(handle);
    return tensorflow::errors::Internal("aoe tuning finalize failed");
  }

  if (handle != nullptr) {
    (void)dlclose(handle);
  }

  DLOG() << "Success to tune ge graph: " << graph_id_;
  return tensorflow::Status::OK();
}

tensorflow::Status NpuAoe::AoeTuningInit(NpuDevice *device, void *handle) {
  DLOG() << "Start to run aoe initialize";

  NPU_REQUIRES_OK(LoadAoeFunc(handle));

  std::map<Aoe::AscendString, Aoe::AscendString> global_options;
  global_options.insert(
      {Aoe::AscendString("work_path"), Aoe::AscendString(device->device_options["work_path"].c_str())});
  auto ret = (*aoe_func_.aoe_initialize)(global_options);
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe initialize func failed"));

  std::map<Aoe::AscendString, Aoe::AscendString> session_options;
  session_options.insert(
      {Aoe::AscendString("job_type"), Aoe::AscendString(device->device_options["aoe_mode"].c_str())});
  ret = (*aoe_func_.aoe_create_session)(session_options, aoe_session_id_);
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe create session func failed"));

  DLOG() << "run aoe initialize success";
  return tensorflow::Status::OK();
}

tensorflow::Status NpuAoe::LoadAoeFunc(void *handle) {
  DLOG() << "Start to load aoe function";

  // aoe init
  aoe_func_.aoe_initialize = (AoeInitializeFunc)dlsym(handle, "AoeInitialize");
  NPU_REQUIRES(aoe_func_.aoe_initialize != nullptr, tensorflow::errors::Internal("dlsym Aoe Initialize API failed"));

  // aoe finalize
  aoe_func_.aoe_finalize = (AoeFinalizeFunc)dlsym(handle, "AoeFinalize");
  NPU_REQUIRES(aoe_func_.aoe_finalize != nullptr, tensorflow::errors::Internal("dlsym Aoe Finalize API failed"));

  // aoe create session
  aoe_func_.aoe_create_session = (AoeCreateSessionFunc)dlsym(handle, "AoeCreateSession");
  NPU_REQUIRES(aoe_func_.aoe_create_session != nullptr,
               tensorflow::errors::Internal("dlsym Aoe create session API failed"));

  // aoe destroy session
  aoe_func_.aoe_destroy_session = (AoeDestroySessionFunc)dlsym(handle, "AoeDestroySession");
  NPU_REQUIRES(aoe_func_.aoe_destroy_session != nullptr,
               tensorflow::errors::Internal("dlsym Aoe destroy session API failed"));

  // aoe set session
  aoe_func_.aoe_set_gesession = (AoeSetGeSessionFunc)dlsym(handle, "AoeSetGeSession");
  NPU_REQUIRES(aoe_func_.aoe_set_gesession != nullptr,
               tensorflow::errors::Internal("dlsym Aoe set session API failed"));

  // aoe set depend graphs
  aoe_func_.aoe_set_dependgraphs = (AoeSetDependGraphFunc)dlsym(handle, "AoeSetDependGraphs");
  NPU_REQUIRES(aoe_func_.aoe_set_dependgraphs != nullptr,
               tensorflow::errors::Internal("dlsym Aoe set depend graphs API failed"));

  // aoe set tuning graph
  aoe_func_.aoe_set_tuninggraph = (AoeSetTuningGraphFunc)dlsym(handle, "AoeSetTuningGraph");
  NPU_REQUIRES(aoe_func_.aoe_set_tuninggraph != nullptr,
               tensorflow::errors::Internal("dlsym Aoe set tuning graph API failed"));

  // aoe tuning
  aoe_func_.aoe_tuning_graph = (AoeTuningGraphFunc)dlsym(handle, "AoeTuningGraph");
  NPU_REQUIRES(aoe_func_.aoe_tuning_graph != nullptr,
               tensorflow::errors::Internal("dlsym Aoe tuning graph API failed"));

  // aoe set tuning depend graphs inputs
  aoe_func_.aoe_set_depend_graphs_inputs = (AoeSetDependGraphsInputsFunc)dlsym(handle, "AoeSetDependGraphsInputs");
  NPU_REQUIRES(aoe_func_.aoe_set_depend_graphs_inputs != nullptr,
               tensorflow::errors::Internal("dlsym Aoe set tuning depend graphs inputs API failed"));

  // aoe set tuning graph inputs
  aoe_func_.aoe_set_tuning_graph_input = (AoeSetTuningGraphInputFunc)dlsym(handle, "AoeSetTuningGraphInput");
  NPU_REQUIRES(aoe_func_.aoe_set_tuning_graph_input != nullptr,
               tensorflow::errors::Internal("dlsym Aoe set tuning graph inputs API failed"));

  DLOG() << "Load aoe function success";
  return tensorflow::Status::OK();
}

tensorflow::Status NpuAoe::AoeTuningFinalize() {
  DLOG() << "Start to run aoe finalize";

  auto ret = (*aoe_func_.aoe_destroy_session)(aoe_session_id_);
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe destroy session func failed"));

  ret = (*aoe_func_.aoe_finalize)();
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe finalize func failed"));

  DLOG() << "run aoe finalize success";
  return tensorflow::Status::OK();
}

tensorflow::Status NpuAoe::CallAoeFuncToTuning(NpuDevice *device, TFE_Context *context,
                                               std::vector<TFE_TensorHandle *> &inputs, TF_Status *status) {
  DLOG() << "Start to run aoe tuning";

  auto ret = (*aoe_func_.aoe_set_gesession)(aoe_session_id_, device->GeSession());
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe set session func failed"));

  ge::Graph ge_graph;
  NPU_REQUIRES_OK(device->TransTfGraph2GeGraph(context, name_, graph_def_, status, ge_graph));
  ge_graph.SetNeedIteration(false);

  // set tuning graph
  ret = (*aoe_func_.aoe_set_tuninggraph)(aoe_session_id_, ge_graph);
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe set tuning graph func failed"));

  std::vector<ge::Tensor> ge_inputs;
  device->TransTfInputs2GeInputs(inputs.size(), inputs.data(), status, ge_inputs);
  if (TF_GetCode(status) != TF_OK) {
    return tensorflow::errors::Internal("get ge tensor inputs failed");
  }

  // set tuning inputs
  ret = (*aoe_func_.aoe_set_tuning_graph_input)(aoe_session_id_, ge_inputs);
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe set tuning inputs func failed"));

  // aoe tuning
  std::map<Aoe::AscendString, Aoe::AscendString> tuning_options;
  ret = (*aoe_func_.aoe_tuning_graph)(aoe_session_id_, tuning_options);
  NPU_REQUIRES((ret == Aoe::AOE_SUCCESS) || (ret == Aoe::AOE_ERROR_NO_AICORE_GRAPH),
               tensorflow::errors::Internal("exec aoe tuning graph func failed"));

  DLOG() << "run aoe tuning success";
  return tensorflow::Status::OK();
}
}  // namespace npu