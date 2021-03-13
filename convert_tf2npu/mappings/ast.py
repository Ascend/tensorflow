{
"need_conver": false,
"gelu":                         ["npu_unary_ops",           "tf.gelu",                      "npu_unary_ops.gelu"],
"dropout":                      ["npu_ops",                 "tf.nn.dropout",                "npu_ops.dropout"],
"init":                         ["print",                   "hvd.init",                     "None"],
"DistributedOptimizer":         ["NPUDistributedOptimizer", "hvd.DistributedOptimizer",     "NPUDistributedOptimizer"],
"rank":                         ["get_npu_rank_id",             "hvd.rank",                     "get_npu_rank_id"],
"local_rank":                   ["get_npu_local_rank_id",       "hvd.local_rank",               "get_npu_local_rank_id"],
"size":                         ["get_npu_rank_size",           "hvd.size",                     "get_npu_rank_size"],
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

"tf.profiler.AdviceProto":      ["",                        "tf.profiler.AdviceProto",                      "None"],
"tf.profiler.AdviceProto.Checker":
                                ["",                        "tf.profiler.AdviceProto.Checker",              "None"],
"tf.profiler.AdviceProto.CheckersEntry":
                                ["",                        "tf.profiler.AdviceProto.CheckersEntry",        "None"],
"tf.profiler.GraphNodeProto":   ["",                        "tf.profiler.GraphNodeProto",      "None"],
"tf.profiler.GraphNodeProto.InputShapesEntry":
                                ["",                        "tf.profiler.GraphNodeProto.InputShapesEntry",  "None"],
"tf.profiler.MultiGraphNodeProto":
                                ["",                        "tf.profiler.MultiGraphNodeProto",              "None"],
"tf.profiler.OpLogProto":       ["",                        "tf.profiler.OpLogProto",                       "None"],
"tf.profiler.OpLogProto.IdToStringEntry":
                                ["",                        "tf.profiler.OpLogProto.IdToStringEntry",       "None"],
"tf.profiler.ProfileOptionBuilder":
                                ["",                        "tf.profiler.ProfileOptionBuilder",             "None"],
"tf.profiler.advise":           ["",                        "tf.profiler.advise",                           "None"],
"tf.profiler.profile":          ["",                        "tf.profiler.profile",                          "None"],
"tf.profiler.write_op_log":     ["",                        "tf.profiler.write_op_log",                     "None"],
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
"Session()":                    ["",                        "*.Session()",                  "*.Session(config=npu_session_config_init())"],
"ConfigProto()":                ["",                        "*.ConfigProto()",              "npu_config_proto(config_proto=*.ConfigProto())"],
"GraphOptions()":               ["",                        "*.GraphOptions()",             "npu_graph_options(graph_options=*.GraphOptions())"],
"OptimizerOptions()":           ["",                        "*.OptimizerOptions()",         "npu_optimizer_options(optimizer_options=*.OptimizerOptions())"],
"MonitoredTrainingSession":     ["",                        "*.MonitoredTrainingSession()", "*.MonitoredTrainingSession(hooks=npu_hooks_append())"],

"hvd":                          ["init", "rank", "local_rank", "size"],
"estimator":                    ["Estimator",               "RunConfig",                    "EstimatorSpec"],
"nn_layers":                    ["dropout"],
"keras":                        [""],

"report_file":                  ["success_report.txt",      "failed_report.txt",             "need_migration_doc.txt"],
"report_file_status": 0
}