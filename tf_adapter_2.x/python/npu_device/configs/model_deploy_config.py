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

"""Configuration for model deploy"""

from npu_device.configs.option_base import OptionValue
from npu_device.configs.option_base import NpuBaseConfig


class ModelDeployConfig(NpuBaseConfig):
    """Config for model deploy"""
    def __init__(self):
        self.model_deploy_mode = OptionValue(None, None)
        self.model_deploy_devicelist = OptionValue(None, None)

        super(ModelDeployConfig, self).__init__()
