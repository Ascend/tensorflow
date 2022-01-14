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

#ifndef NPU_DEVICE_CORE_OPTIMIZERS_NPU_OPTIMIZER_MANAGER_H
#define NPU_DEVICE_CORE_OPTIMIZERS_NPU_OPTIMIZER_MANAGER_H

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

#include "npu_device.h"
#include "npu_logger.h"
#include "npu_parser.h"
#include "npu_unwrap.h"
#include "npu_utils.h"
#include "op_executors/npu_concrete_graph.h"

namespace npu {
using NpuMetaOptimizeFunc =
  std::function<tensorflow::Status(TFE_Context *, tensorflow::Graph *, std::map<std::string, std::string>)>;

using NpuRtOptimizeFunc = std::function<tensorflow::Status(
  TFE_Context *, NpuMutableConcreteGraph *, std::map<std::string, std::string>, NpuDevice *, int, TFE_TensorHandle **)>;

class NpuOptimizerManager {
 public:
  static NpuOptimizerManager &Instance() {
    static NpuOptimizerManager inst;
    return inst;
  }

  void RegisterMeta(int level, const std::string &name, const NpuMetaOptimizeFunc &func) {
    std::lock_guard<std::mutex> lk(mu_);
    meta_optimizers_[level][name] = func;
  }

  void RegisterRt(int level, const std::string &name, const NpuRtOptimizeFunc &func) {
    std::lock_guard<std::mutex> lk(mu_);
    runtime_optimizers_[level][name] = func;
  }

  tensorflow::Status MetaOptimize(TFE_Context *context, std::unique_ptr<tensorflow::Graph> *graph,
                                  std::map<std::string, std::string> options,
                                  OptimizeStageGraphDumper *graph_dumper = nullptr) {
    tensorflow::FunctionLibraryDefinition *lib_def = npu::UnwrapCtx(context)->FuncLibDef();
    if (graph_dumper != nullptr) {
      graph_dumper->DumpWithSubGraphs("before_meta_optimize", (*graph)->ToGraphDefDebug(), lib_def);
    }

    tensorflow::ProcessFunctionLibraryRuntime *pflr = npu::UnwrapCtx(context)->pflr();
    tensorflow::FunctionLibraryRuntime *flr = pflr->GetFLR("/job:localhost/replica:0/task:0/device:CPU:0");

    tensorflow::OptimizeGraph(flr, graph);

    if (graph_dumper != nullptr) {
      graph_dumper->DumpWithSubGraphs("after_tf_meta_optimize", (*graph)->ToGraphDefDebug(), lib_def);
    }

    for (const auto &item : meta_optimizers_) {
      for (const auto &name2optimizer : item.second) {
        NPU_REQUIRES_OK(name2optimizer.second(context, graph->get(), options));
        if (graph_dumper != nullptr) {
          graph_dumper->Dump("after_npu_meta_optimizer_" + name2optimizer.first, (*graph)->ToGraphDefDebug());
        }
      }
    }
    return tensorflow::Status::OK();
  }

  tensorflow::Status RuntimeOptimize(TFE_Context *context, NpuMutableConcreteGraph *graph,
                                     std::map<std::string, std::string> options, NpuDevice *device, int num_inputs,
                                     TFE_TensorHandle **inputs, OptimizeStageGraphDumper *dumper = nullptr) {
    for (const auto &item : runtime_optimizers_) {
      for (const auto &name2optimizer : item.second) {
        NPU_REQUIRES_OK(name2optimizer.second(context, graph, options, device, num_inputs, inputs));
        if (dumper != nullptr) {
          dumper->Dump("after_runtime_optimizer_" + name2optimizer.first, graph->GraphDef());
        }
      }
    }
    return tensorflow::Status::OK();
  }

 private:
  NpuOptimizerManager() = default;
  std::mutex mu_;
  std::map<int, std::map<std::string, NpuMetaOptimizeFunc>> meta_optimizers_;
  std::map<int, std::map<std::string, NpuRtOptimizeFunc>> runtime_optimizers_;
};

class NpuMetaOptimizerReceiver {
 public:
  explicit NpuMetaOptimizerReceiver(int level, const std::string &name, NpuMetaOptimizeFunc func) {
    DLOG() << "NPU Register meta optimizer " << name << " at phase " << level;
    NpuOptimizerManager::Instance().RegisterMeta(level, name, func);
  }
};

class NpuRtOptimizerReceiver {
 public:
  explicit NpuRtOptimizerReceiver(int level, const std::string &name, NpuRtOptimizeFunc func) {
    DLOG() << "NPU Register runtime optimizer " << name << " at phase " << level;
    NpuOptimizerManager::Instance().RegisterRt(level, name, func);
  }
};
}  // namespace npu

#define NPU_REGISTER_META_OPTIMIZER(level, name, func) NPU_REGISTER_META_OPTIMIZER_1(__COUNTER__, level, name, func)
#define NPU_REGISTER_META_OPTIMIZER_1(ctr, level, name, func) NPU_REGISTER_META_OPTIMIZER_2(ctr, level, name, func)
#define NPU_REGISTER_META_OPTIMIZER_2(ctr, level, name, func) \
  static NpuMetaOptimizerReceiver __preserved_op##ctr(level, name, func)

#define NPU_REGISTER_RT_OPTIMIZER(level, name, func) NPU_REGISTER_RT_OPTIMIZER_1(__COUNTER__, level, name, func)
#define NPU_REGISTER_RT_OPTIMIZER_1(ctr, level, name, func) NPU_REGISTER_RT_OPTIMIZER_2(ctr, level, name, func)
#define NPU_REGISTER_RT_OPTIMIZER_2(ctr, level, name, func) \
  static NpuRtOptimizerReceiver __preserved_op##ctr(level, name, func)

#endif  // NPU_DEVICE_CORE_OPTIMIZERS_NPU_OPTIMIZER_MANAGER_H
