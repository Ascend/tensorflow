#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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

import os
import math
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from npu_bridge.helper import helper
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops

gen_npu_ops = helper.get_gen_ops()

DYNAMIC_RNN_UNIDIRECTION = "UNIDIRECTIONAL"
DYNAMIC_RNN_BIDIRECTION = "BIDIRECTIONAL"


class _DynamicBasic(base_layer.Layer):
    """Create a basic class for dynamic using Layer."""

    def __init__(self,
                 hidden_size,
                 dtype,
                 direction=DYNAMIC_RNN_UNIDIRECTION,
                 cell_depth=1,
                 keep_prob=1.0,
                 cell_clip=-1.0,
                 num_proj=0,
                 time_major=True,
                 activation="tanh",
                 is_training=True):
        super(_DynamicBasic, self).__init__()
        self._direction = direction
        self._cell_depth = cell_depth
        self._keep_prob = keep_prob
        self._cell_clip = cell_clip
        self._num_proj = num_proj
        self._time_major = time_major
        self._activation = activation
        self._is_training = is_training
        self._hidden_size = hidden_size
        self._dtype = dtype
        self._args = {
            "direction": self._direction,
            "cell_depth": self._cell_depth,
            "keep_prob": self._keep_prob,
            "cell_clip": self._cell_clip,
            "num_proj": self._num_proj,
            "time_major": self._time_major,
            "activation": self._activation,
            "is_training": self._is_training
        }
        self._seq_length = None
        self._init_h = None

    @property
    def direction(self):
        return self._direction

    @property
    def cell_depth(self):
        return self._cell_depth

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def cell_clip(self):
        return self._cell_clip

    @property
    def num_proj(self):
        return self._num_proj

    @property
    def time_major(self):
        return self._time_major

    @property
    def activation(self):
        return self._activation

    @property
    def is_training(self):
        return self._is_training

    def check_direction(self):
        """Check validity of direction."""
        if self._direction not in (DYNAMIC_RNN_UNIDIRECTION, DYNAMIC_RNN_BIDIRECTION):
            raise ValueError("Invalid direction: %s, expecting %s or %s" %
                             (self._direction, DYNAMIC_RNN_UNIDIRECTION, DYNAMIC_RNN_BIDIRECTION))

    def build(self, input_shape):
        time_size = input_shape[0].value
        batch_size = input_shape[1].value
        if time_size is None:
            time_size = 1
        if batch_size is None:
            batch_size = 16
        self._seq_length = self.add_variable(
            "dynamicbase/seq_length",
            shape=[batch_size],
            dtype=dtypes.int32,
            initializer=init_ops.constant_initializer(time_size, dtype=dtypes.int32),
            trainable=False)
        super(_DynamicBasic, self).build(input_shape)

    def call(self,
             x,
             seq_length=None):
        """Dynamic GRU.
        """
        self.check_direction()
        self._args["x"] = x
        if seq_length is None:
            seq_length = self._seq_length
        self._args["seq_length"] = seq_length


class DynamicGRUV2(_DynamicBasic):
    """Create a basic class for dynamic using Layer."""

    def __init__(self,
                 hidden_size,
                 dtype,
                 direction=DYNAMIC_RNN_UNIDIRECTION,
                 cell_depth=1,
                 keep_prob=1.0,
                 cell_clip=-1.0,
                 num_proj=0,
                 time_major=True,
                 activation="tanh",
                 gate_order="zrh",
                 reset_after=True,
                 is_training=True):
        super(DynamicGRUV2, self).__init__(
            hidden_size,
            dtype,
            direction=direction,
            cell_depth=cell_depth,
            keep_prob=keep_prob,
            cell_clip=cell_clip,
            num_proj=num_proj,
            activation=activation,
            time_major=time_major,
            is_training=is_training)
        self._gate_order = gate_order
        self._reset_after = reset_after
        self._args["gate_order"] = self._gate_order
        self._args["reset_after"] = self._reset_after
        self._gruv2_weight_input = None
        self._gruv2_weight_hidden = None
        self._bias_input = None
        self._bias_hidden = None

    @property
    def gate_order(self):
        return self._gate_order

    @property
    def reset_after(self):
        return self._reset_after

    def build(self, input_shape):
        if input_shape[2].value is None:
            raise ValueError("Expected input_shape[2] to be known, saw shape: input_size.")
        input_size = input_shape[2].value
        stdv = 1.0 / math.sqrt(self._hidden_size)
        self._gruv2_weight_input = self.add_variable(
            "dynamicgruv2/weight_input",
            shape=[input_size, 3 * self._hidden_size],
            dtype=self._dtype,
            initializer=init_ops.random_uniform_initializer(-stdv, stdv))
        self._gruv2_weight_hidden = self.add_variable(
            "dynamicgruv2/weight_hidden",
            shape=[self._hidden_size, 3 * self._hidden_size],
            dtype=self._dtype,
            initializer=init_ops.random_uniform_initializer(-stdv, stdv))
        self._bias_input = self.add_variable(
            "dynamicgruv2/bias_input",
            shape=[3 * self._hidden_size],
            dtype=self._dtype,
            initializer=init_ops.random_uniform_initializer(-stdv, stdv))
        self._bias_hidden = self.add_variable(
            "dynamicgruv2/bias_hidden",
            shape=[3 * self._hidden_size],
            dtype=self._dtype,
            initializer=init_ops.random_uniform_initializer(-stdv, stdv))
        self._init_h = array_ops.zeros([batch_size, self._hidden_size], dtype=self._dtype)
        super(DynamicGRUV2, self).build(input_shape)

    def call(self,
             x,
             seq_length=None,
             init_h=None):
        """Dynamic GRU.
        """
        super(DynamicGRUV2, self).call(x, seq_length=seq_length)
        if init_h is None:
            init_h = self._init_h
        self._args["init_h"] = init_h
        self._args["weight_input"] = self._gruv2_weight_input
        self._args["weight_hidden"] = self._gruv2_weight_hidden
        self._args["bias_input"] = self._bias_input
        self._args["bias_hidden"] = self._bias_hidden
        return gen_npu_ops.dynamic_gru_v2(**self._args)


class DynamicRNN(_DynamicBasic):
    """Create a basic class for dynamic using Layer."""

    def __init__(self,
                 hidden_size,
                 dtype,
                 cell_type="LSTM",
                 direction=DYNAMIC_RNN_UNIDIRECTION,
                 cell_depth=1,
                 use_peephole=False,
                 keep_prob=1.0,
                 cell_clip=-1.0,
                 num_proj=0,
                 time_major=True,
                 activation="tanh",
                 forget_bias=0.0,
                 is_training=True):
        super(DynamicRNN, self).__init__(
            hidden_size,
            dtype,
            direction=direction,
            cell_depth=cell_depth,
            keep_prob=keep_prob,
            cell_clip=cell_clip,
            num_proj=num_proj,
            activation=activation,
            time_major=time_major,
            is_training=is_training)
        self._cell_type = cell_type
        self._use_peephole = use_peephole
        self._forget_bias = forget_bias
        self._args["cell_type"] = self._cell_type
        self._args["use_peephole"] = self._use_peephole
        self._args["forget_bias"] = self._forget_bias
        self._rnn_w = None
        self._rnn_b = None
        self._init_c = None

    @property
    def cell_type(self):
        return self._cell_type

    @property
    def use_peephole(self):
        return self._use_peephole

    @property
    def forget_bias(self):
        return self._forget_bias

    def build(self, input_shape):
        batch_size = input_shape[1].value
        if batch_size is None:
            batch_size = 16
        if input_shape[2].value is None:
            raise ValueError("Expected input_shape[2] to be known, saw shape: input_size.")
        input_size = input_shape[2].value

        self._rnn_w = self.add_variable(
            "dynamicrnn/w",
            shape=[input_size + self._hidden_size, 4 * self._hidden_size],
            dtype=self._dtype,
            initializer=init_ops.glorot_uniform_initializer(seed=10, dtype=self._dtype))
        self._rnn_b = self.add_variable(
            "dynamicrnn/b",
            shape=[4 * self._hidden_size],
            dtype=self._dtype,
            initializer=init_ops.zeros_initializer(dtype=self._dtype))
        super(DynamicRNN, self).build(input_shape)

    def call(self,
             x,
             seq_length=None,
             init_h=None,
             init_c=None,
             weight=None,
             bias=None,):
        """Dynamic RNN.
        """
        super(DynamicRNN, self).call(x, seq_length=seq_length)
        batch_size = array_ops.shape(x)[1]

        if init_h is None:
            self._init_h = array_ops.zeros([1, batch_size, self._hidden_size], dtype=self._dtype)
            init_h = self._init_h
        if init_c is None:
            self._init_c = array_ops.zeros([1, batch_size, self._hidden_size], dtype=self._dtype)
            init_c = self._init_c

        if weight is None:
            weight = self._rnn_w
        if bias is None:
            bias = self._rnn_b
        self._args["w"] = weight
        self._args["b"] = bias
        self._args["init_h"] = init_h
        self._args["init_c"] = init_c
        if seq_length is None:
            self._args.pop("seq_length")
            return gen_npu_ops.dynamic_rnn_v2(**self._args)
        else:
            return gen_npu_ops.dynamic_rnn(**self._args)
