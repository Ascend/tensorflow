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

"""Configuration for dumping NPU data"""

from npu_device.configs.option_base import OptionValue
from npu_device.configs.option_base import NpuBaseConfig


class NpuDumpConfig(NpuBaseConfig):
    """Config for dumping npu training data"""
    def __init__(self):
        self.enable_dump = OptionValue(False, [True, False])
        self.dump_path = OptionValue(None, None)
        self.dump_step = OptionValue(None, None)
        self.dump_mode = OptionValue('output', ['input', 'output', 'all'])
        self.enable_dump_debug = OptionValue(False, [True, False])
        self.dump_debug_mode = OptionValue('all', ['aicore_overflow', 'atomic_overflow', 'all'])
        self.dump_data = OptionValue('tensor', ['tensor', 'stats'])
        self.dump_layer = OptionValue(None, None)

        super(NpuDumpConfig, self).__init__()
