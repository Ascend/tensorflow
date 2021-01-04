# TF1.x脚本迁移工具使用说明

[View English](README.en.md)

## 功能介绍
该工具适用于原生的Tensorflow训练脚本迁移场景，AI算法工程师通过该工具分析原生的TensorFlow Python API在昇腾AI处理器上的支持度情况，同时将原生的TensorFlow训练自动迁移成昇腾AI处理器支持的脚本。对于无法自动迁移的API，您可以参考工具输出的迁移报告，对训练脚本进行相应的适配修改。

使用该工具前，需要了解工具对训练脚本的限制要求：
1. 该工具仅配套TensorFlow 1.15版本的训练脚本。


2. 目前仅支持Python3的脚本


3. 目前支持的TensorFlow模块引入方式如下，如果您的训练脚本未按照如下方式引用TensorFlow模块，请先修改训练脚本。

    import tensorflow as tf

    import tensorflow.compat.v1 as tf

## 使用指导
1. 安装依赖。

    pip install pandas  
    pip install xlrd==1.2.0  
    pip install openpyxl  
    pip install tkintertable  


2. 执行如下命令进行脚本迁移。

   a. 命令举例(Linux)：

   python3 main.py -i /home/BERT -l /home/TF1.15_API支持度清单.xlsx -o /home/out -r /home/report

   其中：

    main.py：为工具的主函数入口

    /home/BERT：被迁移的脚本路径

    /home/TF1.15_API支持度清单.xlsx：TensorFlow 1.15在昇腾AI处理器上的支持度清单

    /home/out：迁移后的脚本路径

    /home/report：生成的迁移报告路径
    > 通过**python3 main.py -h**可以获取脚本的使用帮助。
	
	b. 命令举例(Windows)：
	
	python3 main_win.py  
	
	- 在弹出的窗口依次选择“原始脚本路径”、“API支持度清单”、“输出迁移脚本路径”、“输出分析报告路径”
	
	- 点击“开始分析”，就能看到分析结果，并在指定的“输出迁移脚本路径”下生成迁移后的脚本，在指定的“输出分析报告路径”下生成分析报告
	
	- 点击“重新开始分析”，则返回选择路径窗口，可以重新指定输入脚本，再次分析
	
	- 点击“退出”，则退出分析工具

3. 在/home/report下可以看到迁移报告

    - api_brief_report.txt：为脚本中API的统计结果，例如：
      ```
      1.In brief: Total API: 304, in which Support: 180, Support after migrated by tool: 6, Support after migrated manually: 1, Analysing: 6, Unsupport: 109, Deprecated: 2
      2.After eliminate duplicate: Total API: 122, in which Support: 69, Support after migrated by tool: 3, Support after migrated manually: 1, Analysing: 3, Unsupport: 44, Deprecated: 2
      ```
    - api_analysis_report.txt：为详细的API分析报告，分析报告包括：序号、脚本文件名、代码行、模块名、API名、支持度、迁移建议。

      请参考该表中的迁移建议进行相关修改和适配。其中部分API（**支持度**为**支持但需工具迁移**的这一项对应的API）迁移工具已为您完成自动迁移，您无需再进行手工适配。


4. 附录。

    目前支持自动迁移的功能列表：
    
     Tensorflow函数 --> 迁移后的NPU函数

     "tf.gelu" --> "npu_unary_ops.gelu"

     "tf.nn.dropout" --> "npu_ops.dropout"

     "hvd.init" --> "None"

     "hvd.DistributedOptimizer" --> "NPUDistributedOptimizer"

     "hvd.rank" --> "get_rank_id"

     "hvd.local_rank" --> "get_local_rank_id"

     "hvd.size" --> "get_rank_size"

     "BroadcastGlobalVariablesHook" --> "None"

     "dataset.shard(xxx, xxx)" --> "dataset.shard(get_rank_size(), get_rank_id())"

     "tf.estimator.EstimatorSpec" --> "NPUEstimatorSpec"

     "tf.estimator.RunConfig" --> "NPURunConfig"

     "tf.estimator.Estimator" --> "NPUEstimator"

     "batch(xxx)" --> "batch(xxx, drop_remainder=True)"

     "map_and_batch(xxx)" --> "map_and_batch(xxx, drop_remainder=True)"

     "tf.device(xxx)" --> "tf.device(/cpu:0)"

     "tf.cpmpat.v1.layers.max_pooling2d" --> "tf.compat.v1.nn.max_pool_with_argmax"

    目前迁移工具默认设置的配置项：
    
    config中的remapping默认设置为RewriterConfig.OFF

    custom_op.parameter_map["use_off_line"].b = True默认设置为True

    custom_op.parameter_map["enable_data_pre_proc"].b = True默认设置为True
	
	目前API支持度：
	
	支持：此类API在昇腾AI处理器海思上绝对支持，无需适配修改。
	
	支持但需手工迁移：此类API需要您参考迁移建议，进行适配修改。
	
	支持但需工具迁移：此类API工具已为您自动迁移，无需用户修改适配。
	
	不支持：此类API或者部分数据类型在昇腾AI处理器上不支持，建议您不要使用，否则会引起训练失败。
	
	分析中：此类API我们正在分析中，不保证绝对支持，建议您不要使用，否则可能会引起训练失败。
	
	废弃：此类API在Tensorflow1.15版本已废弃，建议您不要使用，否则可能会引起训练失败。
	
	NA：此类API不在《TF1.15_API支持度清单》的扫描范围内，不保证绝对支持，建议您不要使用，否则可能会引起训练失败。

## 贡献

欢迎参与贡献。

## Release Notes

Release Notes请参考[RELEASE](RELEASE.md).

## License

[Apache License 2.0](LICENSE)
