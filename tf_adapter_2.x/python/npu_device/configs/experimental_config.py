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

"""Configuration for experiment"""

from npu_device.configs.option_base import NpuBaseConfig
from npu_device.configs.option_base import OptionValue
from npu_device.configs.multi_branches_config import NpuMultiBranchesConfig
from npu_device.configs.logical_device_deploy_config import LogicalDeviceDeployConfig
from npu_device.configs.memory_optimize_config import GraphMemoryOptimizeConfig
from npu_device.configs.graph_parallel_config import GraphParallelConfig


class NpuExperimentalConfig(NpuBaseConfig):
    """Config for experiment"""
    def __init__(self):
        self.multi_branches_config = NpuMultiBranchesConfig()
        self.logical_device_deploy_config = LogicalDeviceDeployConfig()

        # run context options
        self.graph_memory_optimize_config = GraphMemoryOptimizeConfig()
        self.graph_parallel_config = GraphParallelConfig()
        self.resource_config_path = OptionValue(None, None)

        super(NpuExperimentalConfig, self).__init__()
