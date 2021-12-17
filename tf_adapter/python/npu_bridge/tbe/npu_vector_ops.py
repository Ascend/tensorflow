#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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

"""Ops for aicore cube."""
from tensorflow import Tensor
from tensorflow.python.eager import context
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from npu_bridge.helper import helper
from npu_bridge.estimator.npu_aicore_ops import prelu
gen_npu_ops = helper.get_gen_ops()

def lamb_apply_optimizer_assign(input0, input1, input2, input3, mul0_x, mul1_x,
                                mul2_x, mul3_x, add2_y, steps, do_use_weight, weight_decay_rate, name=None):
    if context.executing_eagerly():
      raise RuntimeError("tf.lamb_apply_optimizer_assign() is not compatible with "
                        "eager execution.")
    update, nextv, nextm = gen_npu_ops.lamb_apply_optimizer_assign(input0, input1, input2, input3, mul0_x, mul1_x, mul2_x,
                                                     mul3_x, add2_y, steps, do_use_weight, weight_decay_rate, name)
    return update, nextv, nextm

def lamb_apply_weight_assign(input0, input1, input2, input3, input4, name=None):
    if context.executing_eagerly():
      raise RuntimeError("tf.lamb_apply_weight_assign() is not compatible with "
                        "eager execution.")
    result = gen_npu_ops.lamb_apply_weight_assign(input0, input1, input2, input3, input4, name)
    return result


class PReLU(Layer):

    def __init__(self,
                 alpha_initializer='zeros',
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.shared_axes = shared_axes

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        if sum(param_shape) == len(param_shape):
            param_shape = [1, ]
        self.alpha = self.add_weight(
            shape=param_shape,
            name='alpha',
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint)
        self.input_spec = InputSpec(ndim=len(input_shape), axes={})
        self.built = True

    def call(self, inputs):
        return prelu(inputs, self.alpha)