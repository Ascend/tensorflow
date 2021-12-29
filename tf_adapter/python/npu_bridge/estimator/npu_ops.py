#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

"""Ops for collective operations implemented using hccl."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

from tensorflow.python.ops.nn_ops import _get_noise_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.eager import context

from npu_bridge.helper import helper
from npu_bridge.estimator.npu.npu_common import NPUBasics

gen_npu_ops = helper.get_gen_ops()

DEFAULT_GRAPH_SEED = 87654321
_MAXINT32 = 2 ** 31 - 1


def NPUInit(name=None):
    """Initiate NPU"""
    if context.executing_eagerly():
        raise RuntimeError("tf.NPUInit() is not compatible with "
                           "eager execution.")

    return gen_npu_ops.npu_init(name=name)


def NPUShutdown(name=None):
    """Shutdown a distributed NPU system for use with TensorFlow.

    Args:
        name: Name of ops.

    Returns:
        The npu init ops which will open the NPU system using `Session.run`.
    """
    if context.executing_eagerly():
        raise RuntimeError("tf.NPUShutdown() is not compatible with "
                           "eager execution.")

    return gen_npu_ops.npu_shutdown(name=name)


def initialize_system(name=None):
    """Initializes a distributed NPU system for use with TensorFlow.

    Args:
        name: Name of ops.

    Returns:
        The npu init ops which will open the NPU system using `Session.run`.
    """
    return NPUInit(name)


def shutdown_system(name=None):
    """Shuts down a running NPU system."""

    return NPUShutdown(name)


def LARS(inputs_w, inputs_g, weight_decay, hyperpara=0.001, epsilon=0.00001, name=None):
    """NPU implemented LARS"""
    if context.executing_eagerly():
        raise RuntimeError("tf.LARS() is not compatible with "
                           "eager execution.")

    return gen_npu_ops.lars(inputs_w=inputs_w, inputs_g=inputs_g, weight_decay=weight_decay, hyperpara=hyperpara,
                            epsilon=epsilon, name=name)


def LARSV2(input_weight,
           input_grad,
           weight_decay,
           learning_rate,
           hyperpara=0.001,
           epsilon=0.00001,
           use_clip=False,
           name=None):
    """NPU implemented LARSV2"""
    if context.executing_eagerly():
        raise RuntimeError("tf.LARSV2() is not compatible with "
                           "eager execution.")

    return gen_npu_ops.lars_v2(input_weight=input_weight,
                               input_grad=input_grad,
                               weight_decay=weight_decay,
                               learning_rate=learning_rate,
                               hyperpara=hyperpara,
                               epsilon=epsilon,
                               use_clip=use_clip,
                               name=name)


def outfeed_dequeue_op(channel_name, output_types, output_shapes, name=None):
    """Operator for outfeed dequeue"""
    return gen_npu_ops.outfeed_dequeue_op(channel_name=channel_name, output_types=output_types,
                                          output_shapes=output_shapes, name=name)


def outfeed_enqueue_op(channel_name, inputs, name=None):
    """Operator for outfeed enqueue"""
    return gen_npu_ops.outfeed_enqueue_op(inputs=inputs, channel_name=channel_name, name=name)


def stop_outfeed_dequeue_op(channel_name, name=None):
    """Operator for stoping outfeed dequeue"""
    return gen_npu_ops.stop_outfeed_dequeue_op(channel_name, name)


def _truncate_seed(seed):
    return seed % _MAXINT32  # Truncate to fit into 32-bit integer


def dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
    """The gradient for `gelu`.

    Args:
        x: A tensor with type is float.
        keep_prob: A tensor, float, rate of every element reserved.
        noise_shape: A 1-D tensor, with type int32, shape of keep/drop what random
            generated.
        seed: Random seed.
        name: Layer name.

    Returns:
        A tensor.
    """
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


@ops.RegisterGradient("DropOutDoMask")
def _DropOutDoMaskGrad(op, grad):
    result = gen_npu_ops.drop_out_do_mask(grad, op.inputs[1], op.inputs[2])
    return [result, None, None]


def basic_lstm_cell(x, h, c, w, b, keep_prob, forget_bias, state_is_tuple,
                    activation, name=None):
    """NPU implemented lstm cell"""
    if context.executing_eagerly():
        raise RuntimeError("tf.basic_lstm_cell() is not compatible with "
                           "eager execution.")
    x = ops.convert_to_tensor(x, name="x")
    h = ops.convert_to_tensor(h, name="h")
    c = ops.convert_to_tensor(c, name="c")
    w = ops.convert_to_tensor(w, name="w")
    b = ops.convert_to_tensor(b, name="b")
    result = gen_npu_ops.basic_lstm_cell(x, h, c, w, b, keep_prob, forget_bias, state_is_tuple,
                                         activation, name)
    return result


@ops.RegisterGradient("BasicLSTMCell")
def basic_lstm_cell_grad(op, dct, dht, dit, djt, dft, dot, dtanhct):
    """NPU implemented gradient for lstm cell"""
    dgate, dct_1 = gen_npu_ops.basic_lstm_cell_c_state_grad(op.inputs[2], dht, dct, op.outputs[2], op.outputs[3],
                                                            op.outputs[4], op.outputs[5], op.outputs[6],
                                                            forget_bias=op.get_attr("forget_bias"),
                                                            activation=op.get_attr("activation"))
    dw, db = gen_npu_ops.basic_lstm_cell_weight_grad(op.inputs[0], op.inputs[1], dgate)
    dxt, dht = gen_npu_ops.basic_lstm_cell_input_grad(dgate, op.inputs[3], keep_prob=op.get_attr("keep_prob"))

    return [dxt, dht, dct_1, dw, db]


def adam_apply_one_assign(input0, input1, input2, input3, input4,
                          mul0_x, mul1_x, mul2_x, mul3_x, add2_y, name=None):
    """NPU implemented adam_apply_one_assign"""
    if context.executing_eagerly():
        raise RuntimeError("tf.adam_apply_one_assign() is not compatible with "
                           "eager execution.")
    result = gen_npu_ops.adam_apply_one_assign(input0, input1, input2, input3, input4,
                                               mul0_x, mul1_x, mul2_x, mul3_x, add2_y, name)
    return result


def adam_apply_one_with_decay_assign(input0, input1, input2, input3, input4,
                                     mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y, name=None):
    """NPU implemented adam_apply_one_with_decay_assign"""
    if context.executing_eagerly():
        raise RuntimeError("tf.adam_apply_one_with_decay_assign() is not compatible with "
                           "eager execution.")
    result = gen_npu_ops.adam_apply_one_with_decay_assign(input0, input1, input2, input3, input4,
                                                          mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y, name)
    return result


@ops.RegisterGradient("DynamicGruV2")
def dynamic_gru_v2_grad(op, dy, doutput_h, dupdate, dreset, dnew, dhidden_new):
    """NPU implemented dynamic_gru_v2"""
    (x, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, init_h) = op.inputs
    (y, output_h, update, reset, new, hidden_new) = op.outputs
    (dw_input, dw_hidden, db_input, db_hidden, dx, dh_prev) = gen_npu_ops.dynamic_gru_v2_grad(x, weight_input,
                                                                                              weight_hidden, y, init_h,
                                                                                              output_h, dy, doutput_h,
                                                                                              update, reset, new,
                                                                                              hidden_new,
                                                                                              direction=op.get_attr(
                                                                                                  "direction"),
                                                                                              cell_depth=op.get_attr(
                                                                                                  "cell_depth"),
                                                                                              keep_prob=op.get_attr(
                                                                                                  "keep_prob"),
                                                                                              cell_clip=op.get_attr(
                                                                                                  "cell_clip"),
                                                                                              num_proj=op.get_attr(
                                                                                                  "num_proj"),
                                                                                              time_major=op.get_attr(
                                                                                                  "time_major"),
                                                                                              gate_order=op.get_attr(
                                                                                                  "gate_order"),
                                                                                              reset_after=op.get_attr(
                                                                                                  "reset_after"))

    return (dx, dw_input, dw_hidden, db_input, db_hidden, seq_length, dh_prev)


@ops.RegisterGradient("DynamicRnn")
def dynamic_rnn_grad(op, dy, dh, dc, di, dj, df, do, dtanhc):
    """NPU implemented dynamic_rnn_grad"""
    (x, w, b, seq_length, init_h, init_c) = op.inputs
    (y, output_h, output_c, i, j, f, o, tanhc) = op.outputs
    (dw, db, dx, dh_prev, dc_prev) = gen_npu_ops.dynamic_rnn_grad(x, w, b, y, init_h[-1], init_c[-1], output_h,
                                                                  output_c, dy, dh[-1], dc[-1], i, j, f, o, tanhc,
                                                                  cell_type=op.get_attr("cell_type"),
                                                                  direction=op.get_attr("direction"),
                                                                  cell_depth=op.get_attr("cell_depth"),
                                                                  use_peephole=op.get_attr("use_peephole"),
                                                                  keep_prob=op.get_attr("keep_prob"),
                                                                  cell_clip=op.get_attr("cell_clip"),
                                                                  num_proj=op.get_attr("num_proj"),
                                                                  time_major=op.get_attr("time_major"),
                                                                  forget_bias=op.get_attr("forget_bias"))

    return (dx, dw, db, seq_length, dh_prev, dc_prev)


@ops.RegisterGradient("DynamicRnnV2")
def dynamic_rnn_v2_grad(op, dy, dh, dc, di, dj, df, do, dtanhc):
    """NPU implemented dynamic_rnn_v2_grad"""
    (x, w, b, init_h, init_c) = op.inputs
    (y, output_h, output_c, i, j, f, o, tanhc) = op.outputs
    (dw, db, dx, dh_prev, dc_prev) = gen_npu_ops.dynamic_rnn_grad(x, w, b, y, init_h[-1], init_c[-1], output_h,
                                                                  output_c, dy, dh[-1], dc[-1], i, j, f, o, tanhc,
                                                                  cell_type=op.get_attr("cell_type"),
                                                                  direction=op.get_attr("direction"),
                                                                  cell_depth=op.get_attr("cell_depth"),
                                                                  use_peephole=op.get_attr("use_peephole"),
                                                                  keep_prob=op.get_attr("keep_prob"),
                                                                  cell_clip=op.get_attr("cell_clip"),
                                                                  num_proj=op.get_attr("num_proj"),
                                                                  time_major=op.get_attr("time_major"),
                                                                  forget_bias=op.get_attr("forget_bias"))

    return (dx, dw, db, dh_prev, dc_prev)


def scatter_elements(data, indices, updates, axis=0, name=None):
    """Scatter data based on indices"""
    data = ops.convert_to_tensor(data, name="data")
    indices = ops.convert_to_tensor(indices, name="indices")
    updates = ops.convert_to_tensor(updates, name="updates")
    y = gen_npu_ops.scatter_elements(data, indices, updates, axis, name)
    return y


def k_means_centroids(x, y, sum_square_y, sum_square_x, use_actual_distance=False, name=None):
    """k_means_centroids.

    Args:
        x: A tensor with type is float.
        y: A tensor with type is float.
        sum_square_y: A tensor with type is float.
        sum_square_x: A tensor with type is float or None.
        use_actual_distance: Whether to output accurate Loss
        name: name.

    Returns:
        A tensor.
    """
    if context.executing_eagerly():
        raise RuntimeError("tf.k_means_centroids() is not compatible with "
                           "eager execution.")
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y")
    sum_square_y = ops.convert_to_tensor(sum_square_y, name="sum_square_y")
    if sum_square_x is not None:
        sum_square_x = ops.convert_to_tensor(sum_square_x, name="sum_square_x")
        use_actual_distance = True
    else:
        use_actual_distance = False

    if use_actual_distance:
        result = gen_npu_ops.k_means_centroids(x, y, sum_square_y, sum_square_x, use_actual_distance, name)
    else:
        result = gen_npu_ops.k_means_centroids_v2(x, y, sum_square_y, use_actual_distance, name)
    return result


def npu_onnx_graph_op(inputs, tout, model_path, name=None):
    """NPU implemented onnx graph operator"""
    output = gen_npu_ops.npu_onnx_graph_op(inputs, tout, model_path, name)
    return output
