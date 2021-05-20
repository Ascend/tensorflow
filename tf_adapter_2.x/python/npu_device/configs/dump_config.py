from npu_device.configs.option_base import OptionValue
from npu_device.configs.option_base import NpuBaseConfig


class NpuDumpConfig(NpuBaseConfig):
    def __init__(self):
        self.enable_dump = OptionValue(False, [True, False])
        self.dump_path = OptionValue(None, None)
        self.dump_step = OptionValue(None, None)
        self.dump_mode = OptionValue('output', ['input', 'output', 'all'])
        self.enable_dump_debug = OptionValue(False, [True, False])
        self.dump_debug_mode = OptionValue(None, None)

        super(NpuDumpConfig, self).__init__()
