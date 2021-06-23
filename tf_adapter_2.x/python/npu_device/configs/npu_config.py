from npu_device.configs.dump_config import NpuDumpConfig
from npu_device.configs.profiling_config import NpuProfilingConfig
from npu_device.configs.option_base import OptionValue
from npu_device.configs.option_base import NpuBaseConfig


class NpuConfig(NpuBaseConfig):

    def __init__(self):
        self.graph_run_mode = OptionValue(1, [0, 1])
        self.graph_memory_max_size = OptionValue(None, None)
        self.variable_memory_max_size = OptionValue(None, None)
        self.variable_format_optimize = OptionValue(True, [True, False])
        self.enable_scope_fusion_passes = OptionValue(None, None)
        self.fusion_switch_file = OptionValue(None, None)
        self.precision_mode = OptionValue('allow_fp32_to_fp16',
                                          ['force_fp32', 'allow_fp32_to_fp16', 'force_fp16', 'must_keep_origin_dtype',
                                           'allow_mix_precision'])
        self.auto_tune_mode = OptionValue(None, None)
        self.op_select_implmode = OptionValue('high_performance', ['high_performance', 'high_precision'])
        self.optypelist_for_implmode = OptionValue(None, None)
        self.op_compiler_cache_mode = OptionValue('disable', ['enable', 'disable', 'force'])
        self.op_compiler_cache_dir = OptionValue(None, None)
        self.stream_max_parallel_num = OptionValue(None, None)
        self.hcom_parallel = OptionValue(False, [True, False])
        self.hcom_multi_mode = OptionValue(None, None)
        self.is_tailing_optimization = OptionValue(False, [True, False])
        self.op_debug_level = OptionValue(0, [0, 1, 2, 3])
        self.debug_dir = OptionValue(None, None)
        self.modify_mixlist = OptionValue(None, None)
        self.enable_exception_dump = OptionValue(0, [0, 1])
        self.dump_config = NpuDumpConfig()
        self.profiling_config = NpuProfilingConfig()

        super(NpuConfig, self).__init__()
