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

#include <deque>
#include <iostream>
#include <memory>
#include <string>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/public/session_options.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/infershape_util.h"
#include "tf_adapter/util/npu_attrs.h"

namespace tensorflow {
static const int64 kMicrosToMillis = 1000;
static std::atomic<int> graph_run_num(1);
static mutex graph_num_mutex(LINKER_INITIALIZED);

class GetAttrOptimizePass : public GraphOptimizationPass {
 public:
  GetAttrOptimizePass() = default;
  ~GetAttrOptimizePass() override = default;
  Status Run(const GraphOptimizationPassOptions &options) override;
};

Status GetAttrOptimizePass::Run(const GraphOptimizationPassOptions &options) {
  if (options.graph == nullptr || options.flib_def == nullptr || options.session_options == nullptr) {
    return Status::OK();
  }
  int graph_num = graph_run_num++;

  std::map<std::string, std::string> pass_options = NpuAttrs::GetPassOptions(options);
  std::string job = pass_options["job"];
  ADP_LOG(INFO) << "NpuAttrs job is " << job;
  if (job == "ps" || job == "default") {
    ADP_LOG(INFO) << "job is " << job << " Skip the optimizer : GetAttrOptimizePass.";
    return Status::OK();
  }

  for (Node *n : options.graph->get()->nodes()) {
    REQUIRES_NOT_NULL(n);
    if (n->attrs().Find("_NoNeedOptimize")) {
      ADP_LOG(INFO) << "Found mark of noneed optimize on node [" << n->name() << "], skip AddSrcOpAttrPass.";
      return Status::OK();
    }

    if (n->attrs().Find("_NpuOptimizer")) {
      ADP_LOG(INFO) << "Found mark of get attr optimize on node [" << n->name() << "], skip AddSrcOpAttrPass.";
      return Status::OK();
    }
  }

  int64 startTime = InferShapeUtil::GetCurrentTimestap();

  if (kDumpGraph) {
    GraphDef ori_graph_def;
    options.graph->get()->ToGraphDef(&ori_graph_def);
    string ori_model_path = GetDumpPath() + "BeforeGetAttrOptimize_";
    string omodel_path = ori_model_path + std::to_string(graph_num) + ".pbtxt";
    (void)WriteTextProto(Env::Default(), omodel_path, ori_graph_def);
  }

  for (Node *n : options.graph->get()->nodes()) {
    REQUIRES_NOT_NULL(n);
    std::string device_name = n->assigned_device_name();
    if (!device_name.empty() && device_name.find("/job:ps") == std::string::npos) {
      Status s = NpuAttrs::SetNpuOptimizerAttr(options, n);
      if (s != Status::OK()) {
        ADP_LOG(INFO) << "set npu optimizer ret != 0.";
        return s;
      }
      break;
    }
  }

  if (job != "localhost") {
    for (Node *node : options.graph->get()->op_nodes()) {
      REQUIRES_NOT_NULL(node);
      std::string device_name = node->assigned_device_name();
      if (device_name.find("/device:") != std::string::npos) {
        device_name = device_name.substr(0, device_name.find("/device:"));
        device_name += std::string("/device:CPU:0");
      }
      node->set_assigned_device_name(device_name);
    }
  }

  if (kDumpGraph) {
    GraphDef omg_graph_def;
    options.graph->get()->ToGraphDef(&omg_graph_def);
    string tmpmodel_path = GetDumpPath() + "AfterGetAttrOptimize_";
    string tmodel_path = tmpmodel_path + std::to_string(graph_num) + ".pbtxt";
    (void)WriteTextProto(Env::Default(), tmodel_path, omg_graph_def);
  }
  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "GetAttrOptimizePass_" << std::to_string(graph_num) << " success. ["
                << ((endTime - startTime) / kMicrosToMillis) << " ms]";

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0, GetAttrOptimizePass);
}  // namespace tensorflow
