#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

"""NPU implemented dropout"""

import numbers
from npu_device.npu_device import gen_npu_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops.nn_ops import _get_noise_shape
from tensorflow.python.ops import nn_ops
import tensorflow as tf
import tensorflow.compat.v2 as tf2

_GRAPH_MODE = "graph"
_EAGER_MODE = "eager"


def npu_dropout_v2(x, rate, noise_shape=None, seed=None, name=None):
    """Computes Npu dropout.

    Args:
        x: A tensor with type is float.
        rate: A tensor, float, rate of every element reserved.
        noise_shape: A 1-D tensor, with type int32, shape of keep/drop what random
            generated.
        seed: Random seed.
        name: Layer name.

    Returns:
        A tensor.
    """
    keep_prob = 1.0 - rate
    if noise_shape:
        if hasattr(npu_dropout_v2, "__OriginCall__"):
            npu_dropout_v2.__OriginCall__(x,
                                          noise_shape=noise_shape,
                                          seed=seed,
                                          name=name,
                                          rate=(1.0 - keep_prob))
        else:
            raise Exception("dropout_v2 has no __OriginCall__ attr")
    if context.executing_eagerly():
        raise RuntimeError("npu_ops.dropout() is not compatible with "
                           "eager execution.")
    x = ops.convert_to_tensor(x, name="x")
    if not x.dtype.is_floating:
        raise ValueError("x must be a floating point tensor."
                         " Got a %s tensor instead." % x.dtype)
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1.0:
        raise ValueError("keep_prob must be a float value or a scalar tensor in the "
                         "range (0, 1], got %g" % keep_prob)
    if isinstance(keep_prob, float) and keep_prob == 1.0:
        return x
    seed, seed2 = random_seed.get_seed(seed)
    noise_shape = _get_noise_shape(x, noise_shape)
    gen_out = gen_npu_ops.drop_out_gen_mask(noise_shape, keep_prob, seed, seed2, name)
    result = gen_npu_ops.drop_out_do_mask(x, gen_out, keep_prob, name)
    return result


def dropout_api_convert():
    """ Replace dropout API """
    from npu_device.train import npu_convert
    # Replace Api function which define in __init__
    tf.nn.dropout = npu_convert.dropout_convert(npu_dropout_v2, version=['2.6.2'], mode=['graph'])(tf.nn.dropout)
    tf2.nn.dropout = npu_convert.dropout_convert(npu_dropout_v2, version=['2.6.2'], mode=['graph'])(tf2.nn.dropout)
    # Replace Api function in nn_ops.py
    nn_ops.dropout_v2 = npu_convert.dropout_convert(npu_dropout_v2, version=['2.6.2'], mode=['graph'])(
        nn_ops.dropout_v2)


def dropout_domask_grad(op, grad):
    """NPU implemented gradient for dropout"""
    result = gen_npu_ops.drop_out_do_mask(grad, op.inputs[1], op.inputs[2])
    return [result, None, None]


grad_registry_list = ops.gradient_registry.list()
if "DropOutDoMask" not in grad_registry_list:
    ops.RegisterGradient("DropOutDoMask")(dropout_domask_grad)
