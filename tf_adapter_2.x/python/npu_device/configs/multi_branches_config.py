#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Configuration for multi branches"""

from npu_device.configs.option_base import OptionValue
from npu_device.configs.option_base import NpuBaseConfig


class NpuMultiBranchesConfig(NpuBaseConfig):
    """Config for multi branches"""
    def __init__(self):
        self.input_shape = OptionValue(None, None)
        self.dynamic_node_type = OptionValue('0', ['0'])
        self.dynamic_dims = OptionValue(None, None)

        super(NpuMultiBranchesConfig, self).__init__()
