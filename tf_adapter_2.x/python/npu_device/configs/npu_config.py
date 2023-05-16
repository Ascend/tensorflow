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
from npu_device.configs.option_base import DeprecatedValue
from npu_device.configs.option_base import NpuBaseConfig
from npu_device.configs.aoe_config import NpuAoeConfig
from npu_device.configs.memory_config import MemoryConfig


class NpuConfig(NpuBaseConfig):
    """Set NPU configuration"""
    def __init__(self):
        self.graph_run_mode = OptionValue(1, [0, 1])
        self.graph_memory_max_size = OptionValue(None, None)
        self.variable_memory_max_size = OptionValue(None, None)
        self.variable_format_optimize = DeprecatedValue([True, False], replacement=None)
        self.enable_scope_fusion_passes = OptionValue(None, None)
        self.fusion_switch_file = OptionValue(None, None)
        self.precision_mode = OptionValue(None,
                                          ['force_fp32', 'allow_fp32_to_fp16', 'force_fp16', 'must_keep_origin_dtype',
                                           'allow_mix_precision', 'cube_fp16in_fp32out', 'allow_mix_precision_fp16',
                                           'allow_mix_precision_bf16', 'allow_fp32_to_bf16'])
        self.op_select_implmode = DeprecatedValue(['high_performance', 'high_precision'],
                                                  replacement='op_precision_mode')
        self.optypelist_for_implmode = DeprecatedValue(None, replacement='op_precision_mode')
        self.op_compiler_cache_mode = OptionValue('disable', ['enable', 'disable', 'force'])
        self.op_compiler_cache_dir = OptionValue(None, None)
        self.stream_max_parallel_num = OptionValue(None, None)
        self.hcom_parallel = OptionValue(True, [True, False])
        self.hcom_multi_mode = OptionValue(None, None)
        self.is_tailing_optimization = OptionValue(False, [True, False])
        self.op_debug_level = DeprecatedValue([0, 1, 2, 3, 4], replacement='op_debug_config')
        self.op_debug_config = OptionValue(None, None)
        self.debug_dir = OptionValue(None, None)
        self.modify_mixlist = OptionValue(None, None)
        self.enable_exception_dump = OptionValue(0, [0, 1])
        self.dump_config = NpuDumpConfig()
        self.aoe_config = NpuAoeConfig()
        self.profiling_config = NpuProfilingConfig()
        self.enable_small_channel = OptionValue(False, [True, False])
        self.deterministic = OptionValue(0, [0, 1])
        self.op_precision_mode = OptionValue(None, None)
        self.graph_exec_timeout = OptionValue(None, None)
        self.topo_sorting_mode = OptionValue(None, [0, 1, None])
        self.customize_dtypes = OptionValue(None, None)
        self.overflow_flag = OptionValue(1, [0, 1])
        self.stream_sync_timeout = OptionValue(-1, None)
        self.event_sync_timeout = OptionValue(-1, None)
        self.external_weight = OptionValue(False, [True, False])
        self.memory_config = MemoryConfig()
        self.jit_compile = OptionValue(None, [True, False])
        self.graph_compiler_cache_dir = OptionValue(None, None)

        # Configuration for experiment
        self.experimental = NpuExperimentalConfig()

        super(NpuConfig, self).__init__()
