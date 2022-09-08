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

"""Construct NPU configuration"""

from npu_device.configs.dump_config import NpuDumpConfig
from npu_device.configs.profiling_config import NpuProfilingConfig
from npu_device.configs.experimental_config import NpuExperimentalConfig
from npu_device.configs.option_base import OptionValue
from npu_device.configs.option_base import NpuBaseConfig
from npu_device.configs.aoe_config import NpuAoeConfig


class NpuConfig(NpuBaseConfig):
    """Set NPU configuration"""
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
        self.op_debug_level = OptionValue(0, [0, 1, 2, 3, 4])
        self.debug_dir = OptionValue(None, None)
        self.modify_mixlist = OptionValue(None, None)
        self.enable_exception_dump = OptionValue(0, [0, 1])
        self.dump_config = NpuDumpConfig()
        self.aoe_config = NpuAoeConfig()
        self.profiling_config = NpuProfilingConfig()
        self.enable_small_channel = OptionValue(False, [True, False])
        self.graph_exec_timeout = OptionValue(None, None)
        self.jit_compile = OptionValue(True, [True, False])
        self.topo_sorting_mode = OptionValue(None, [0, 1, None])
        self.customize_dtypes = OptionValue(None, None)

        # Configuration for experiment
        self.experimental = NpuExperimentalConfig()

        super(NpuConfig, self).__init__()
