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

"""All bert ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops.nn_ops import _get_noise_shape
from tensorflow.python.framework import dtypes

from npu_bridge.helper import helper
from npu_bridge.estimator.npu.npu_common import NPUBasics

npu_aicore_ops = helper.get_gen_ops()

DEFAULT_GRAPH_SEED = 87654321
_MAXINT32 = 2 ** 31 - 1


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


def fast_gelu_v2(x, name=None):
    """ fast_gelu_v2 operator interface implementation

    Args:
        x: A input tensor with type is float16 or float32.

    Returns:
        A tensor.
    """
    return npu_aicore_ops.fast_gelu_v2(x, name)


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
    """Gradient for prelu"""
    dx, da = npu_aicore_ops.p_relu_grad(grad, op.inputs[0], op.inputs[1])
    return [dx, da]


def prelu(x, weight):
    """prelu op"""
    return npu_aicore_ops.p_relu(x, weight)


def _truncate_seed(seed):
    return seed % _MAXINT32  # Truncate to fit into 32-bit integer


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
        raise ValueError("x must be a floating point tensor."
                         " Got a %s tensor instead." % x.dtype)
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1.0:
        raise ValueError("keep_prob must be a float value or a scalar tensor in the "
                         "range (0, 1], got %g" % keep_prob)
    if isinstance(keep_prob, float) and keep_prob == 1.0:
        return x
    seed, seed2 = random_seed.get_seed(seed)
    noise_shape = _get_noise_shape(x, noise_shape)
    gen_out = npu_aicore_ops.drop_out_gen_mask_v3(noise_shape, keep_prob, seed, seed2, name)
    result = npu_aicore_ops.drop_out_do_mask_v3(x, gen_out, keep_prob, name)
    return result


@ops.RegisterGradient("DropOutDoMaskV3")
def _DropOutDoMaskV3Grad(op, grad):
    result = npu_aicore_ops.drop_out_do_mask_v3(grad, op.inputs[1], op.inputs[2])
    return [result, None, None]


def dropout_v4(x, keep_prob, noise_shape=None, seed=None, output_dtype=dtypes.bool, name=None):
    """The gradient for `gelu`.

    Args:
        x: A tensor with type is float.
        keep_prob: A tensor, float, rate of every element reserved.
        noise_shape: A 1-D tensor, with type int32, shape of keep/drop what random
            generated.
        seed: Random seed.
        output_dtype: dtype of output tensor, default is bool.
        name: Layer name.

    Returns:
        A tensor.
    """
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
    gen_out = npu_aicore_ops.drop_out_gen_mask_v4(noise_shape, keep_prob, seed, seed2, output_dtype, name)
    result = npu_aicore_ops.drop_out_do_mask_v3(x, gen_out, keep_prob, name)
    return result


def lru_cache_v2(index_list, data, cache, tag, is_last_call, pre_route_count, name=None):
    """
    LRUCacheV2 op

    """
    is_last_call = ops.convert_to_tensor(is_last_call, name="is_last_call")
    data, cache, tag, index_offset_list, not_in_cache_index_list, not_in_cache_number = npu_aicore_ops.lru_cache_v2(
        index_list, data, cache, tag, is_last_call, pre_route_count, name=name)
    return [data, cache, tag, index_offset_list, not_in_cache_index_list, not_in_cache_number]


def nonzero(x, transpose=False, output_type=dtypes.int64, name=None):
    """
    nonezero op
    Return the indices of the elementes that are non-zero.
    Return a tuple of arrays,one for each dimension of a ,containing the indices of the non-zero elementes
    in that dimension. The values in a are always tested and returned in row-major ,C-style order.

    """
    x = ops.convert_to_tensor(x, name="x")
    result = npu_aicore_ops.non_zero(x, transpose, output_type, name=name)
    return result


def nonzerowithvalue(x, transpose=False, output_type=dtypes.int64, name=None):
    """
    nonezero op
    Return the indices of the elementes that are non-zero.
    Return a tuple of arrays,one for each dimension of a ,containing the indices of the non-zero elementes
    in that dimension. The values in a are always tested and returned in row-major ,C-style order.

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


def prod_env_mat_a(coord, types, natoms, box, mesh, davg, dstd, rcut_a=0.0,
                   rcut_r=0.0, rcut_r_smth=0.0, sel_a=None, sel_r=None, name=None):
    """
    prod_env_mat_a op
    Return the indices of the elementes that are non-zero.
    Return a tuple of arrays,one for each dimension of a ,containing the indices of the non-zero elementes
    in that dimension. The values in a are always tested and returned in row-major ,C-style order.

    """
    sel_a = [] if sel_a is None else sel_a
    sel_r = [] if sel_r is None else sel_r
    coord = ops.convert_to_tensor(coord, name="coord")
    types = ops.convert_to_tensor(types, name="type")
    natoms = ops.convert_to_tensor(natoms, name="natoms")
    box = ops.convert_to_tensor(box, name="box")
    mesh = ops.convert_to_tensor(mesh, name="mesh")
    davg = ops.convert_to_tensor(davg, name="davg")
    dstd = ops.convert_to_tensor(dstd, name="dstd")
    result = npu_aicore_ops.prod_env_mat_a(coord, types, natoms, box, mesh, davg, dstd, rcut_a, rcut_r,
                                           rcut_r_smth, sel_a, sel_r, name)

    return result


def prodvirialsea(net_deriv, in_deriv, rij, nlist, natoms, n_a_sel=0, n_r_sel=0, name=None):
    """
    ProdVirialSeA op
    """
    net_deriv = ops.convert_to_tensor(net_deriv, name="net_deriv")
    in_deriv = ops.convert_to_tensor(in_deriv, name="in_deriv")
    rij = ops.convert_to_tensor(rij, name="rij")
    nlist = ops.convert_to_tensor(nlist, name="nlist")
    natoms = ops.convert_to_tensor(natoms, name="natoms")
    result = npu_aicore_ops.prod_virial_se_a(net_deriv, in_deriv, rij, nlist, natoms, n_a_sel, n_r_sel,
                                             name=name)
    return result


def prodforcesea(net_deriv, in_deriv, nlist, natoms, n_a_sel=0, n_r_sel=0, name=None):
    """
    ProdForceSeA op
    """
    net_deriv = ops.convert_to_tensor(net_deriv, name="net_deriv")
    in_deriv = ops.convert_to_tensor(in_deriv, name="in_deriv")
    nlist = ops.convert_to_tensor(nlist, name="nlist")
    natoms = ops.convert_to_tensor(natoms, name="natoms")
    result = npu_aicore_ops.prod_force_se_a(net_deriv, in_deriv, nlist, natoms, n_a_sel, n_r_sel,
                                             name=name)
    return result


def tabulatefusionsea(table, table_info, em_x, em, last_layer_size=128, name=None):
    """
    TabulateFusionSeA op
    """
    table = ops.convert_to_tensor(table, name="table")
    table_info = ops.convert_to_tensor(table_info, name="table_info")
    em_x = ops.convert_to_tensor(em_x, name="em_x")
    em = ops.convert_to_tensor(em, name="em")
    result = npu_aicore_ops.tabulate_fusion_se_a(table, table_info, em_x, em, last_layer_size, name=name)
    return result


def tabulatefusionseagrad(table, table_info, em_x, em, dy_dem_x, dy_dem, name=None):
    """
    TabulateFusionSeAGrad op
    """
    table = ops.convert_to_tensor(table, name="table")
    table_info = ops.convert_to_tensor(table_info, name="table_info")
    em_x = ops.convert_to_tensor(em_x, name="em_x")
    em = ops.convert_to_tensor(em, name="em")
    dy_dem_x = ops.convert_to_tensor(dy_dem_x, name="dy_dem_x")
    dy_dem = ops.convert_to_tensor(dy_dem, name="dy_dem")
    result = npu_aicore_ops.tabulate_fusion_se_a_grad(table, table_info, em_x, em, dy_dem_x, dy_dem, name=name)
    return result


def tabulatefusion(table, table_info, em_x, em, last_layer_size=128, name=None):
    """
    TabulateFusion op
    """
    table = ops.convert_to_tensor(table, name="table")
    table_info = ops.convert_to_tensor(table_info, name="table_info")
    em_x = ops.convert_to_tensor(em_x, name="em_x")
    em = ops.convert_to_tensor(em, name="em")
    result = npu_aicore_ops.tabulate_fusion(table, table_info, em_x, em, last_layer_size, name=name)
    return result


def tabulatefusiongrad(table, table_info, em_x, em, dy_dem_x, dy_dem, name=None):
    """
    TabulateFusionGrad op
    """
    table = ops.convert_to_tensor(table, name="table")
    table_info = ops.convert_to_tensor(table_info, name="table_info")
    em_x = ops.convert_to_tensor(em_x, name="em_x")
    em = ops.convert_to_tensor(em, name="em")
    dy_dem_x = ops.convert_to_tensor(dy_dem_x, name="dy_dem_x")
    dy_dem = ops.convert_to_tensor(dy_dem, name="dy_dem")
    result = npu_aicore_ops.tabulate_fusion_grad(table, table_info, em_x, em, dy_dem_x, dy_dem, name=name)
    return result
