{
"need_conver": false,
"gelu":                         ["npu_unary_ops",           "tf.gelu",                      "npu_unary_ops.gelu"],
"dropout":                      ["npu_ops",                 "tf.xxx.dropout",                "npu_ops.dropout"],
"Dropout":                      ["npu_ops",                 "tf.xxx.Dropout",                "npu_ops.dropout"],
"init":                         ["print",                   "hvd.init",                     "None"],
"DistributedOptimizer":         ["NPUDistributedOptimizer", "hvd.DistributedOptimizer",     "NPUDistributedOptimizer"],
"rank":                         ["get_rank_id",             "hvd.rank",                     "get_rank_id"],
"local_rank":                   ["get_local_rank_id",       "hvd.local_rank",               "get_local_rank_id"],
"size":                         ["get_rank_size",           "hvd.size",                     "get_rank_size"],
"BroadcastGlobalVariablesHook": ["print",                   "BroadcastGlobalVariablesHook", "None"],
"shard":                        ["",                        "dataset.shard(xxx, xxx)",      "dataset.shard(get_rank_size(), get_rank_id())"],
"EstimatorSpec":                ["NPUEstimatorSpec",        "tf.estimator.EstimatorSpec",   "NPUEstimatorSpec"],
"RunConfig":                    ["NPURunConfig",            "tf.estimator.RunConfig",       "NPURunConfig"],
"Estimator":                    ["NPUEstimator",            "tf.estimator.Estimator",       "NPUEstimator"],
"import":                       ["",                        "",                             "'from npu_bridge.npu_init import *'"],

"batch":                        ["",                        "batch(xxx)",                   "batch(xxx, drop_remainder=True)"],
"map_and_batch":                ["",                        "map_and_batch(xxx)",           "map_and_batch(xxx, drop_remainder=True)"],
"device":                       ["",                        "tf.device(xxx)",               "tf.device(/cpu:0)"],

"hvd":                          ["init","rank",             "local_rank","size",           "DistributedOptimizer"],
"estimator":                    ["Estimator",               "RunConfig",                    "EstimatorSpec"],
"nn_layers":                    ["dropout",                 "Dropout"],
"keras":                        [""],

"report_file":                  ["success_report.txt",      "failed_report.txt",             "need_migration_doc.txt"],
"report_file_status": 0
}