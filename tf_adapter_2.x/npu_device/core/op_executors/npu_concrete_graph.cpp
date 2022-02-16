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

#include "npu_concrete_graph.h"

#include "npu_device.h"
#include "npu_global.h"

namespace npu {
std::string NpuConcreteGraph::AttachedDebugString() const {
  std::stringstream ss;
  return ss.str();
}

void NpuConcreteGraph::RunImpl(TFE_Context *context, NpuDevice *device, int tf_num_inputs, TFE_TensorHandle **tf_inputs,
                               int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  if (is_cpu_graph_) {
    DLOG() << "Run function graph " << Op() << " on cpu";
    device->FallbackCPU(context, NodeDef(), tf_num_inputs, tf_inputs, num_outputs, outputs, status);
    return;
  }
  if (empty_ge_graph_) {
    DLOG() << "Skipped run empty ge graph";
    return;
  }
  std::vector<TFE_TensorHandle *> pruned_inputs;
  PruneInputs(tf_num_inputs, tf_inputs, pruned_inputs);
  int num_inputs = pruned_inputs.size();
  TFE_TensorHandle **inputs = pruned_inputs.data();
  // 注意，因为GE当前执行图的时候，输入输出内存都是Host的，所以这里和ACL执行相反，如果输入是NPU，则需要转回CPU，
  // 特别的，对于资源类，当前采取的策略是资源入图
  std::vector<TFE_TensorHandle *> npu_inputs(num_inputs);
  ScopeTensorHandleDeleter scope_handle_deleter;
  for (int i = 0; i < num_inputs; ++i) {
    TFE_TensorHandle *input = inputs[i];
    // 到达这里的Resource，要么是CPU的镜像 要么是NPU
    if (npu::IsNpuTensorHandle(input) && tensorflow::unwrap(input)->DataType() != tensorflow::DT_RESOURCE) {
      tensorflow::Status tf_status;
      auto src_name = tensorflow::unwrap(input)->DeviceName(&tf_status);
      NPU_CTX_REQUIRES_OK(status, tf_status);
      DLOG() << "Copying " << Op() << " input:" << i
             << " type:" << tensorflow::DataTypeString(tensorflow::unwrap(input)->DataType()) << " from " << src_name
             << " to CPU for graph engine executing";
      // 这里需要根据算子选择输入格式了
      input = device->CopyTensorD2H(context, input, status);
      scope_handle_deleter.Guard(input);
      if (TF_GetCode(status) != TF_OK) return;
    }
    npu_inputs[i] = input;
  }

  // 这里根据小循环策略修改值
  int64_t iterations_per_loop = 1;
  if (NeedLoop()) {
    iterations_per_loop = npu::global::g_npu_loop_size;
    device->SetNpuLoopSize(context, iterations_per_loop, status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  int64_t consume_resource_times = 1;
  if (NeedLoop() || BuiltinLoop()) {
    consume_resource_times = npu::global::g_npu_loop_size;
  }

  bool looped = NeedLoop() || BuiltinLoop();
  for (const auto &resource : DependentHostResources()) {
    if (looped || kDumpExecutionDetail) {
      LOG(INFO) << "Start consume iterator resource " << resource.second->Name() << " " << consume_resource_times
                << " times";
    }
    const tensorflow::Tensor *tensor;
    NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(tf_inputs[resource.first], &tensor));
    // 注意，这个callback不能引用捕获，防止中途因为消费某个资源失败而导致coredump
    auto done = [resource, consume_resource_times, looped](const tensorflow::Status &s) {
      if (looped || !s.ok() || kDumpExecutionDetail) {
        LOG(INFO) << "Iterator resource " << resource.second->Name() << " consume " << consume_resource_times
                  << " times done with status " << s.ToString();
      }
    };
    NPU_CTX_REQUIRES_OK(status, resource.second->ConsumeAsync(*tensor, consume_resource_times, done));
  }

  Load(context, device, status);
  if (empty_ge_graph_) {
    DLOG() << "Skipped run empty ge graph";
    return;
  }

  if (NeedLoop() || kDumpExecutionDetail) {
    LOG(INFO) << "Start run ge graph " << GeGraphId() << " pin to cpu, loop size " << iterations_per_loop;
  }
  npu::Timer timer("Graph engine run ", iterations_per_loop, " times for graph ", GeGraphId());
  timer.Start();
  device->RunGeGraphPin2Cpu(context, GeGraphId(), num_inputs, npu_inputs.data(), OutputTypes(), num_outputs, outputs,
                            status);
  timer.Stop();
}

void NpuConcreteGraph::Load(TFE_Context *context, NpuDevice *device, TF_Status *status) const {
  if (Built() && device->GeSession()->IsGraphNeedRebuild(GeGraphId())) {
    LOG(INFO) << "Unload ge graph " << GeGraphId() << " for rebuild of op " << Op();
    device->RemoveGeGraph(context, GeGraphId(), status);
    if (TF_GetCode(status) != TF_OK) return;
    built_ = false;
  }

  if (!built_) {
    DLOG() << "Load ge graph " << GeGraphId() << " of op " << Op();
    if (kEmptyGeGraphId == device->AddGeGraphInner(context, GeGraphId(), Op(), GraphDef(), NeedLoop(), status)) {
      empty_ge_graph_ = true;
    }
    if (TF_GetCode(status) != TF_OK) return;
    built_ = true;
    graph_def_serialized_ = true;
  }
}

void NpuConcreteGraph::UnLoad(TFE_Context *context, NpuDevice *device, TF_Status *status) const {
  if (!Built()) return;
  DLOG() << "Unload ge graph " << GeGraphId() << " of op " << Op();
  device->RemoveGeGraph(context, GeGraphId(), status);
  if (TF_GetCode(status) != TF_OK) return;
  built_ = false;
}

void NpuConcreteGraph::RunOneShot(TFE_Context *context, NpuDevice *device, int num_inputs, TFE_TensorHandle **inputs,
                                  int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) const {
  DLOG() << "Run one shot ge graph " << GeGraphId() << " for resource consume op " << Op();
  RunImpl(context, device, num_inputs, inputs, num_outputs, outputs, status);
  if (TF_GetCode(status) != TF_OK) return;
  UnLoad(context, device, status);
}
}  // namespace npu
