from npu_device.configs.option_base import OptionValue
from npu_device.configs.option_base import NpuBaseConfig


class NpuProfilingConfig(NpuBaseConfig):
    def __init__(self):
        self.enable_profiling = OptionValue(False, [True, False])
        self.profiling_options = OptionValue(None, None)

        super(NpuProfilingConfig, self).__init__()
