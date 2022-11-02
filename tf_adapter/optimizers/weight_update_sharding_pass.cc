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

#include "tf_adapter/optimizers/weight_update_sharding_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/public/session_options.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/infershape_util.h"

namespace tensorflow {
static const int64 kMicrosToMillis = 1000;

static std::atomic<int32_t> graph_run_num(1);
Status WeightUpdateShardingPass::Run(const GraphOptimizationPassOptions &options) {
  if (options.graph == nullptr || options.flib_def == nullptr || options.session_options == nullptr) {
    return Status::OK();
  }
  int32_t graph_num = graph_run_num++;

  Graph *graphIn = (options.graph)->get();
  std::map<std::string, std::string> pass_options = NpuAttrs::GetPassOptions(options);
  std::string job = pass_options["job"];
  if (job == "ps" || job == "default") {
    ADP_LOG(INFO) << "job is " << job << " Skip the optimizer : WeightUpdateShardingPass. ";
    return Status::OK();
  }

  bool weight_update_sharding = false;
  bool npu_loss_scale = false;
  for (Node *node : graphIn->op_nodes()) {
    REQUIRES_NOT_NULL(node);
    std::string op_name;
    std::string op_type;
    op_name = node->name();
    op_type = node->type_string();
    if (op_name.find("_Broadcast_Weight_Update_Sharding") != std::string::npos) {
      weight_update_sharding = true;
      if (npu_loss_scale) {
        break;
      }
    }
    if (op_name.find("NPULossScaleOptimizer") != std::string::npos && op_type == "NpuAllocFloatStatus") {
      npu_loss_scale = true;
      if (weight_update_sharding) {
        break;
      }
    }
  }

  if (weight_update_sharding) {
    int64 startTime = InferShapeUtil::GetCurrentTimestap();
    if (kDumpGraph) {
      GraphDef ori_graph_def;
      graphIn->ToGraphDef(&ori_graph_def);
      std::string ori_model_path = GetDumpPath() + "BeforeWeightUpdateSharding_";
      std::string omodel_path = ori_model_path + std::to_string(graph_num) + ".pbtxt";
      (void)WriteTextProto(Env::Default(), omodel_path, ori_graph_def);
    }

    std::vector<Node *> in_nodes;
    (void) std::copy(graphIn->nodes().begin(), graphIn->nodes().end(), std::back_inserter(in_nodes));
    for (int32_t i = static_cast<int32_t>(in_nodes.size()) - 1; i >= 0; i--) {
      Node *node = in_nodes.at(static_cast<size_t>(i));
      REQUIRES_NOT_NULL(node);
      std::string op_type = node->type_string();
      std::string dst_name;
      std::string dst_type;
      if (op_type == "VarHandleOp" || op_type == "Identity" || op_type == "ReadVariableOp") {
        Node *var_node = nullptr;
        Node *broadcast_node = nullptr;
        std::vector<const Edge *> remove_edges;
        for (auto in_edge : node->in_edges()) {
          REQUIRES_NOT_NULL(in_edge);
          REQUIRES_NOT_NULL(in_edge->src());
          REQUIRES_NOT_NULL(in_edge->dst());
          if (in_edge->src()->IsVariable()) {
            var_node = in_edge->src();
            break;
          }
        }
        std::vector<const Edge *> out_edges;
        (void) std::copy(node->out_edges().begin(), node->out_edges().end(), std::back_inserter(out_edges));
        for (auto out_edge : out_edges) {
          REQUIRES_NOT_NULL(out_edge);
          REQUIRES_NOT_NULL(out_edge->src());
          REQUIRES_NOT_NULL(out_edge->dst());
          dst_name = out_edge->dst()->name();
          dst_type = out_edge->dst()->type_string();
          if (!npu_loss_scale) {
            if (dst_name.find("_Broadcast_Weight_Update_Sharding") != std::string::npos &&
                dst_type == "HcomBroadcast") {
              bool find_broadcast = false;
              for (auto broadcast_edge : out_edge->dst()->in_edges()) {
                REQUIRES_NOT_NULL(broadcast_edge);
                REQUIRES_NOT_NULL(broadcast_edge->src());
                REQUIRES_NOT_NULL(broadcast_edge->dst());
                if (broadcast_edge->IsControlEdge()) {
                  find_broadcast = true;
                  // remove edge : reduce/apply --> broadcast
                  remove_edges.push_back(broadcast_edge);
                }
              }
              if (find_broadcast) {
                broadcast_node = out_edge->dst();
                // remove edge : VarHandleOp/Identity --> broadcast
                remove_edges.push_back(out_edge);
                for (auto broadcast_edge : out_edge->dst()->out_edges()) {
                  REQUIRES_NOT_NULL(broadcast_edge);
                  REQUIRES_NOT_NULL(broadcast_edge->src());
                  REQUIRES_NOT_NULL(broadcast_edge->dst());
                  if (broadcast_edge->IsControlEdge()) {
                    // remove edge : broadcast --> group
                    remove_edges.push_back(broadcast_edge);
                  }
                }
                break;
              }
            }
          } else {
            if (dst_type == "Switch") {
              for (auto switch_out_edge : out_edge->dst()->out_edges()) {
                REQUIRES_NOT_NULL(switch_out_edge);
                REQUIRES_NOT_NULL(switch_out_edge->src());
                REQUIRES_NOT_NULL(switch_out_edge->dst());
                std::string node_name = switch_out_edge->dst()->name();
                std::string node_type = switch_out_edge->dst()->type_string();
                if (node_name.find("_Broadcast_Weight_Update_Sharding") != std::string::npos &&
                    node_type == "HcomBroadcast") {
                  bool find_broadcast = false;
                  for (auto broadcast_edge : switch_out_edge->dst()->in_edges()) {
                    REQUIRES_NOT_NULL(broadcast_edge);
                    REQUIRES_NOT_NULL(broadcast_edge->src());
                    REQUIRES_NOT_NULL(broadcast_edge->dst());
                    if (broadcast_edge->IsControlEdge()) {
                      find_broadcast = true;
                      // remove edge : reduce/apply --> broadcast
                      remove_edges.push_back(broadcast_edge);
                    }
                  }
                  if (find_broadcast) {
                    broadcast_node = switch_out_edge->dst();
                    // remove edge : Switch --> broadcast
                    remove_edges.push_back(switch_out_edge);
                    for (auto broadcast_edge : switch_out_edge->dst()->out_edges()) {
                      REQUIRES_NOT_NULL(broadcast_edge);
                      REQUIRES_NOT_NULL(broadcast_edge->src());
                      REQUIRES_NOT_NULL(broadcast_edge->dst());
                      if (broadcast_edge->IsControlEdge()) {
                        // remove edge : broadcast --> group
                        remove_edges.push_back(broadcast_edge);
                      }
                    }
                    break;
                  }
                }
              }
            }
          }
        }
        if (broadcast_node != nullptr && var_node != nullptr) {
          for (auto edge : remove_edges) {
            graphIn->RemoveEdge(edge);
          }
          // add edge : variable --> broadcast
          (void) graphIn->AddEdge(var_node, 0, broadcast_node, 0);
          for (auto var_edge : var_node->out_edges()) {
            REQUIRES_NOT_NULL(var_edge);
            REQUIRES_NOT_NULL(var_edge->src());
            REQUIRES_NOT_NULL(var_edge->dst());
            if (var_edge->dst() != broadcast_node) {
              (void) graphIn->AddControlEdge(broadcast_node, var_edge->dst());
            }
          }
        }
      }
    }

    if (kDumpGraph) {
      GraphDef omg_graph_def;
      graphIn->ToGraphDef(&omg_graph_def);
      string tmpmodel_path = GetDumpPath() + "AfterWeightUpdateSharding_";
      string tmodel_path = tmpmodel_path + std::to_string(graph_num) + ".pbtxt";
      (void)WriteTextProto(Env::Default(), tmodel_path, omg_graph_def);
    }
    int64 endTime = InferShapeUtil::GetCurrentTimestap();
    ADP_LOG(INFO) << "WeightUpdateSharding_" << std::to_string(graph_num) << " success. ["
                  << ((endTime - startTime) / kMicrosToMillis) << " ms]";
  }

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 2, WeightUpdateShardingPass);
}  // namespace tensorflow
