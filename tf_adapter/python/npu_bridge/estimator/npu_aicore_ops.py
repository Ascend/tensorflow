#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""All bert ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  numbers
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.eager import context
from tensorflow.python.framework import device
from npu_bridge.estimator.npu.npu_common import NPUBasics

from npu_bridge.helper import helper
npu_aicore_ops = helper.get_gen_ops();

DEFAULT_GRAPH_SEED = 87654321
_MAXINT32 = 2**31 - 1

@ops.RegisterGradient("FastGelu")
def _fast_gelu_grad(op, grad):
  """The gradient for `fast_gelu`.

  Args:
      op: The `fast_gelu` `Operation` that we are differentiating, which we can use
          to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `fast_gelu` op.

  Returns:
      Gradients with respect to the input of `fast_gelu`.
  """
  return [npu_aicore_ops.fast_gelu_grad(grad, op.inputs[0])]  # List of one Tensor, since we have one input


def centralization(x, axes, name=None):
    """
    centralization op
        return x - reduce_mean(x, axes)
    """
    x = ops.convert_to_tensor(x, name="x")
    result = npu_aicore_ops.centralization(x, axes, name=name)
    return result

@ops.RegisterGradient("PRelu")
def prelu_grad(op, grad):
    dx, da = npu_aicore_ops.p_relu_grad(grad, op.inputs[0], op.inputs[1])
    return [dx, da]

def prelu(x, weight):
    return npu_aicore_ops.p_relu(x, weight)

def _truncate_seed(seed):
      return seed % _MAXINT32  # Truncate to fit into 32-bit integer

# go/tf-wildcard-import
def get_seed(op_seed):
  global_seed = ops.get_default_graph().seed

  if global_seed is not None:
    if op_seed is None:
      op_seed = ops.get_default_graph()._last_id

    seeds = _truncate_seed(global_seed), _truncate_seed(op_seed)
  else:
    if op_seed is not None:
      seeds = DEFAULT_GRAPH_SEED, _truncate_seed(op_seed)
    else:
      seeds = None, None
  # Avoid (0, 0) as the C++ ops interpret it as nondeterminism, which would
  # be unexpected since Python docs say nondeterminism is (None, None).
  if seeds == (0, 0):
    return (0, _MAXINT32)
  return seeds

def _get_noise_shape(x, noise_shape):
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return array_ops.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = tensor_shape.as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tensor_shape.TensorShape(new_dims)

  return noise_shape

def dropout_v3(x, keep_prob, noise_shape=None, seed=None, name=None):
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
    x = ops.convert_to_tensor(x, name="x")
    if not x.dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    if isinstance(keep_prob, float) and keep_prob == 1:
      return x
    seed, seed2 = get_seed(seed)
    noise_shape = _get_noise_shape(x, noise_shape)
    gen_out = npu_aicore_ops.drop_out_gen_mask_v3(noise_shape, keep_prob, seed, seed2, name)
    result = npu_aicore_ops.drop_out_do_mask_v3(x, gen_out, keep_prob, name)
    return result

@ops.RegisterGradient("DropOutDoMaskV3")
def _DropOutDoMaskV3Grad(op, grad):
    result = npu_aicore_ops.drop_out_do_mask_v3(grad, op.inputs[1],  op.inputs[2])
    return [result, None, None]

def nonzero(x, transpose=False, output_type=dtypes.int64, name=None):
    """
    nonezero op
    Return the indices of the elementes that are non-zero.
    Return a tuple of arrays,one for each dimension of a ,containing the indices of the non-zero elementes in that dimension.
    The values in a are always tested and returned in row-major ,C-style order.

    """
    x = ops.convert_to_tensor(x, name="x")
    result = npu_aicore_ops.non_zero(x, transpose, output_type, name=name)
    return result


def nonzerowithvalue(x, transpose=False, output_type=dtypes.int64, name=None):
    """
    nonezero op
    Return the indices of the elementes that are non-zero.
    Return a tuple of arrays,one for each dimension of a ,containing the indices of the non-zero elementes in that dimension.
    The values in a are always tested and returned in row-major ,C-style order.

    """
    x = ops.convert_to_tensor(x, name="x")
    result = npu_aicore_ops.non_zero_with_value(x, transpose, output_type, name=name)
    return result
# go/tf-wildcard-import


def layer_norm(x, gamma, beta, begin_norm_axis=0, begin_params_axis=0, epsilon=0.0000001, name=None):
    """ LayerNorm operator interface implementation

    Args:
        x: A input tensor with type is float16 or float32.
        gamma: scaling operation to normalized tensor.
        beta: add offset to normalized tensor.
        begin_norm_axis: A optional attribute, the type is int32. Defaults to 0.
        begin_params_axis: A optional attribute, the type is int32. Defaults to 0.
        epsilon: A optional attribute, the type is int32. Defaults to 0.0000001.
        name: Layer name.

    Returns:
        A tensor.
    """
    res, mean, variance = npu_aicore_ops.fused_layer_norm(x, gamma, beta, begin_norm_axis,
                                                          begin_params_axis, epsilon, name)

    return [res, mean, variance]


@ops.RegisterGradient("FusedLayerNorm")
def _layer_norm_grad(op, *grad):
    pd_x, pd_gamma, pd_beta = npu_aicore_ops.fused_layer_norm_grad(grad[0], op.inputs[0], op.outputs[2], op.outputs[1],
                                                                   op.inputs[1])

    return [pd_x, pd_gamma, pd_beta]
