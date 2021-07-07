#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
{
"need_conver": false,
"gelu":                         ["npu_unary_ops",           "tf.gelu",                      "npu_unary_ops.gelu"],
"dropout":                      ["npu_ops",                 "tf.nn.dropout",                "npu_ops.dropout"],
"init":                         ["print",                   "hvd.init",                     "None"],
"DistributedOptimizer":         ["NPUDistributedOptimizer", "hvd.DistributedOptimizer",     "NPUDistributedOptimizer"],
"rank":                         ["get_npu_rank_id",         "hvd.rank",                     "get_npu_rank_id"],
"local_rank":                   ["get_npu_local_rank_id",   "hvd.local_rank",               "get_npu_local_rank_id"],
"size":                         ["get_npu_rank_size",       "hvd.size",                     "get_npu_rank_size"],
"BroadcastGlobalVariablesHook": ["print",                   "hvd.BroadcastGlobalVariablesHook", "None"],
"shard":                        ["",                        "dataset.shard(xxx, xxx)",      "dataset.shard(get_rank_size(), get_rank_id())"],
"EstimatorSpec":                ["NPUEstimatorSpec",        "tf.estimator.EstimatorSpec",   "NPUEstimatorSpec"],
"RunConfig":                    ["NPURunConfig",            "tf.estimator.RunConfig",       "NPURunConfig"],
"Estimator":                    ["NPUEstimator",            "tf.estimator.Estimator",       "NPUEstimator"],
"import":                       ["",                        "",                             "'from npu_bridge.npu_init import *'"],
"import config_pb2":            ["",                        "",                             "import tensorflow.core.protobuf.config_pb2"],
"import RewriterConfig":        ["",                        "",                             "from tensorflow.core.protobuf.config_pb2 import RewriterConfig"],
"NPUDistributedOptimizer":      ["",                        "",                             "NPUDistributedOptimizer(xxx)"],
"KerasDistributeOptimizer":      ["",                        "",                             "KerasDistributeOptimizer(xxx)"],
"get_distribution_strategy":    ["npu_strategy", "distribution_utils.get_distribution_strategy", "NPUStrategy"],
"MirroredStrategy":             ["npu_strategy",               "tf.distribute.MirroredStrategy", "NPUStrategy"],
"MultiWorkerMirroredStrategy":  ["npu_strategy",    "tf.distribute.MultiWorkerMirroredStrategy", "NPUStrategy"],

"batch":                        ["",                        "batch(xxx)",                   "batch(xxx, drop_remainder=True)"],
"map_and_batch":                ["",                        "map_and_batch(xxx)",           "map_and_batch(xxx, drop_remainder=True)"],
"device":                       ["",                        "tf.device(xxx)",               "tf.device('/cpu:0')"],
"max_pooling2d":                ["",                        "max_pooling2d",                "max_pool_with_argmax"],
"TPUEstimator(use_tpu=*)":      ["",                        "TPUEstimator(use_tpu=*)",                      "TPUEstimator(use_tpu=False)"],
"TPUEstimator(eval_on_tpu=*)":  ["",                        "TPUEstimator(eval_on_tpu=*)",                  "TPUEstimator(eval_on_tpu=False)"],
"TPUEstimator(export_to_tpu=*)":["",                        "TPUEstimator(export_to_tpu=*)",                "TPUEstimator(export_to_tpu=False)"],

"*.global_jit_level":           ["",                        "global_jit_level=*",           "global_jit_level=OFF"],
"OptimizerOptions.global_jit_level":
                                ["",                        "",                             "OptimizerOptions.global_jit_level=OFF"],
"GraphOptions.global_jit_level":
                                ["",                        "",                             "GraphOptions.optimizer_options.global_jit_level=OFF"],
"ConfigProto.global_jit_level": ["",                        "",                             "ConfigProto.graph_options.optimizer_options.global_jit_level=OFF"],
"add_npu_config":               ["",                        "",                             "add NPU config"],
"VirtualDeviceConfiguration":   ["",                        "",                             "set/add memory_limit=None"],
"set_soft_device_placement":    ["",                        "",                             "set/add enabled=True"],
"set_memory_growth":            ["",                        "*.set_memory_growth()",        "None"],
"set_virtual_device_configuration":
                                ["",                        "*.set_virtual_device_configuration()",         "None"],
"*.xla.experimental.jit_scope": ["",                        "",                             "set/add compile_ops=False"],
"Estimators":                   ["Estimator",                           "TPUEstimator",
                                 "BaselineClassifier",                  "BaselineEstimator",                            "BaselineRegressor",
                                 "BoostedTreesClassifier",              "BoostedTreesEstimator",                        "BoostedTreesRegressor",
                                 "DNNClassifier",                       "DNNEstimator",                                 "DNNRegressor",
                                 "DNNLinearCombinedClassifier",         "DNNLinearCombinedEstimator",                   "DNNLinearCombinedRegressor",
                                 "LinearClassifier",                    "LinearEstimator",                              "LinearRegressor"],
"EstimatorFunc":                ["train"],
"Session()":                    ["",                        "*.*Session()",                  "*.*Session(config=npu_session_config_init())"],
"ConfigProto()":                ["",                        "*.ConfigProto()",              "npu_config_proto(config_proto=*.ConfigProto())"],
"GraphOptions()":               ["",                        "*.GraphOptions()",             "npu_graph_options(graph_options=*.GraphOptions())"],
"OptimizerOptions()":           ["",                        "*.OptimizerOptions()",         "npu_optimizer_options(optimizer_options=*.OptimizerOptions())"],
"MonitoredTrainingSession":     ["",                        "*.MonitoredTrainingSession()", "*.MonitoredTrainingSession(hooks=npu_hooks_append())"],
"TrainSpec":                    ["",                        "*.TrainSpec()",                "*.TrainSpec(hooks=npu_hooks_append())"],
"EvalSpec":                     ["",                        "*.EvalSpec()",                 "*.EvalSpec(hooks=npu_hooks_append())"],

"hvd":                          ["init", "rank", "local_rank", "size"],
"estimator":                    ["Estimator",               "RunConfig",                    "EstimatorSpec"],
"nn_layers":                    ["dropout"],
"keras":                        [""],

"report_file":                  ["success_report.txt",      "failed_report.txt",             "need_migration_doc.txt"],
"report_file_status": 0
}