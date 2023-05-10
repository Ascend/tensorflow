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
NpuAoe &NpuAoe::GetInstance() {
  static NpuAoe instance;
  return instance;
}

tensorflow::Status NpuAoe::RunAoeTuning(NpuDevice &device, TFE_Context *context, bool need_build, uint64_t graph_id,
                                        const std::string &name, const tensorflow::GraphDef &graph_def,
                                        std::vector<TFE_TensorHandle *> &inputs) {
  DLOG() << "Start to tune graph id: " << graph_id << ", name: " << name;
  ++exec_num_;

  SessionId aoe_session_id = 0UL;
  auto ret = aoe_func_.aoe_create_session(aoe_session_id);
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe create session func failed"));

  ret = aoe_func_.aoe_set_gesession(aoe_session_id, device.GeSession());
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe set session func failed"));

  ge::Graph ge_graph;
  const auto iter = ge_graph_.find(graph_id);
  if (need_build || (iter == ge_graph_.cend())) {
    DLOG() << "Convert tf graph to ge graph of graph id: " << graph_id;
    NPU_REQUIRES_OK(device.TransTfGraph2GeGraph(context, name, graph_def, ge_graph));
    ge_graph.SetNeedIteration(false);
    ge_graph_[graph_id] = ge_graph;
  } else {
   ge_graph = iter->second;
   DLOG() << "Get ge graph cache of graph id: " << graph_id;
  }

  // set tuning graph
  ret = aoe_func_.aoe_set_tuninggraph(aoe_session_id, ge_graph);
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe set tuning graph func failed"));

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  std::vector<ge::Tensor> ge_inputs;
  device.TransTfInputs2GeInputs(static_cast<int32_t>(inputs.size()), inputs.data(), *status.get(), ge_inputs);
  if (TF_GetCode(status.get()) != TF_OK) {
    return tensorflow::errors::Internal("get ge tensor inputs failed");
  }

  // set tuning inputs
  ret = aoe_func_.aoe_set_tuning_graph_input(aoe_session_id, ge_inputs);
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe set tuning inputs func failed"));

  // aoe tuning
  std::map<ge::AscendString, ge::AscendString> tuning_options;
  (void)tuning_options.emplace(ge::AscendString("ge.aoe_config_file"),
                               ge::AscendString(device.device_options["ge.aoe_config_file"].c_str()));
  ret = aoe_func_.aoe_tuning_graph(aoe_session_id, tuning_options);
  NPU_REQUIRES((ret == Aoe::AOE_SUCCESS) || (ret == Aoe::AOE_ERROR_NO_AICORE_GRAPH),
               tensorflow::errors::Internal("exec aoe tuning graph func failed"));

  ret = aoe_func_.aoe_destroy_session(aoe_session_id);
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe destroy session func failed"));

  DLOG() << "Success to tune graph: " << graph_id;
  return tensorflow::Status::OK();
}

tensorflow::Status NpuAoe::AoeTuningInitialize(const std::string &work_path, const std::string &job_type) {
  DLOG() << "Start to run aoe initialize";

  handle_ = dlopen("libaoe_tuning.so", RTLD_NOW);
  NPU_REQUIRES(handle_ != nullptr, tensorflow::errors::Internal("libaoe_tuning.so dlopen failed"));

  NPU_REQUIRES_OK(LoadAoeFunc());

  std::map<ge::AscendString, ge::AscendString> global_options;
  (void)global_options.emplace(ge::AscendString("work_path"), ge::AscendString(work_path.c_str()));
  (void)global_options.emplace(ge::AscendString("job_type"),
                               ge::AscendString(job_type.c_str()));
  auto ret = aoe_func_.aoe_initialize(global_options);
  NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe initialize func failed"));

  DLOG() << "Run aoe initialize success";
  return tensorflow::Status::OK();
}

tensorflow::Status NpuAoe::LoadAoeFunc() {
  DLOG() << "Start to load aoe function";

  // aoe init
  aoe_func_.aoe_initialize = reinterpret_cast<AoeInitializeFunc>(dlsym(handle_, "AoeInitialize"));
  NPU_REQUIRES(aoe_func_.aoe_initialize != nullptr,
               tensorflow::errors::Internal("dlsym Aoe InmmDladdritialize API failed"));

  // aoe finalize
  aoe_func_.aoe_finalize = reinterpret_cast<AoeFinalizeFunc>(dlsym(handle_, "AoeFinalize"));
  NPU_REQUIRES(aoe_func_.aoe_finalize != nullptr, tensorflow::errors::Internal("dlsym Aoe Finalize API failed"));

  // aoe create session
  aoe_func_.aoe_create_session = reinterpret_cast<AoeCreateSessionFunc>(dlsym(handle_, "AoeCreateSession"));
  NPU_REQUIRES(aoe_func_.aoe_create_session != nullptr,
               tensorflow::errors::Internal("dlsym Aoe create session API failed"));

  // aoe destroy session
  aoe_func_.aoe_destroy_session = reinterpret_cast<AoeDestroySessionFunc>(dlsym(handle_, "AoeDestroySession"));
  NPU_REQUIRES(aoe_func_.aoe_destroy_session != nullptr,
               tensorflow::errors::Internal("dlsym Aoe destroy session API failed"));

  // aoe set session
  aoe_func_.aoe_set_gesession = reinterpret_cast<AoeSetGeSessionFunc>(dlsym(handle_, "AoeSetGeSession"));
  NPU_REQUIRES(aoe_func_.aoe_set_gesession != nullptr,
               tensorflow::errors::Internal("dlsym Aoe set session API failed"));

  // aoe set depend graphs
  aoe_func_.aoe_set_dependgraphs = reinterpret_cast<AoeSetDependGraphFunc>(dlsym(handle_, "AoeSetDependGraphs"));
  NPU_REQUIRES(aoe_func_.aoe_set_dependgraphs != nullptr,
               tensorflow::errors::Internal("dlsym Aoe set depend graphs API failed"));

  // aoe set tuning graph
  aoe_func_.aoe_set_tuninggraph = reinterpret_cast<AoeSetTuningGraphFunc>(dlsym(handle_, "AoeSetTuningGraph"));
  NPU_REQUIRES(aoe_func_.aoe_set_tuninggraph != nullptr,
               tensorflow::errors::Internal("dlsym Aoe set tuning graph API failed"));

  // aoe tuning
  aoe_func_.aoe_tuning_graph = reinterpret_cast<AoeTuningGraphFunc>(dlsym(handle_, "AoeTuningGraph"));
  NPU_REQUIRES(aoe_func_.aoe_tuning_graph != nullptr,
               tensorflow::errors::Internal("dlsym Aoe tuning graph API failed"));

  // aoe set tuning depend graphs inputs
  aoe_func_.aoe_set_depend_graphs_inputs =
    reinterpret_cast<AoeSetDependGraphsInputsFunc>(dlsym(handle_, "AoeSetDependGraphsInputs"));
  NPU_REQUIRES(aoe_func_.aoe_set_depend_graphs_inputs != nullptr,
               tensorflow::errors::Internal("dlsym Aoe set tuning depend graphs inputs API failed"));

  // aoe set tuning graph inputs
  aoe_func_.aoe_set_tuning_graph_input =
    reinterpret_cast<AoeSetTuningGraphInputFunc>(dlsym(handle_, "AoeSetTuningGraphInput"));
  NPU_REQUIRES(aoe_func_.aoe_set_tuning_graph_input != nullptr,
               tensorflow::errors::Internal("dlsym Aoe set tuning graph inputs API failed"));

  DLOG() << "Load aoe function success";
  return tensorflow::Status::OK();
}

tensorflow::Status NpuAoe::AoeTuningFinalize() {
  if (handle_ != nullptr) {
    DLOG() << "Start to run aoe finalize";

    auto ret = aoe_func_.aoe_finalize();
    NPU_REQUIRES(ret == Aoe::AOE_SUCCESS, tensorflow::errors::Internal("exec aoe finalize func failed"));

    DLOG() << "Total number of aoe executions is: " << exec_num_;
    DLOG() << "Run aoe finalize success";
  }

  return tensorflow::Status::OK();
}

NpuAoe::~NpuAoe() {
  if (handle_ != nullptr) {
    DLOG() << "close handle";
    (void)dlclose(handle_);
    handle_ = nullptr;
  }
}
}  // namespace npu