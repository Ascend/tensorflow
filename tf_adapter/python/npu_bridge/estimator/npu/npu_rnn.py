#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""NPU implemented rnn"""

import tensorflow as tf


def npu_dynamic_rnn(cell,
                    inputs,
                    initial_state=None,
                    dtype=None,
                    sequence_length=None,
                    scope=None):
    """Creates a high performance neural network specified by RNNCell `cell`.
    """
    # tf origin static_rnn
    inputs = tf.unstack(inputs, axis=0)
    encoder_outputs, encoder_state = tf.nn.static_rnn(
        cell,
        inputs,
        initial_state=initial_state,
        dtype=dtype,
        sequence_length=sequence_length,
        scope=scope)
    encoder_outputs = tf.stack(encoder_outputs, axis=0)

    return encoder_outputs, encoder_state
