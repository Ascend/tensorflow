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

"""Public functions for NPU compat"""

import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


class SessionConfigBuilder:
    def __init__(self, config=None):
        if isinstance(config, tf.compat.v1.ConfigProto):
            self._session_config = config
        else:
            self._session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self._session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        self._npu_config = self._session_config.graph_options.rewrite_options.custom_optimizers.add()
        self._npu_config.name = 'NpuOptimizer'

    @property
    def parameter_map(self):
        return self._npu_config.parameter_map

    @property
    def session_config(self):
        return self._session_config
