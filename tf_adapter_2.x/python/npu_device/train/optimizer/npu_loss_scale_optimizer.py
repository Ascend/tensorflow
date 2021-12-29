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

"""Functions used for NPU loss scale"""

import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.util import nest
from tensorflow.python.framework import smart_cond

from tensorflow.python.keras.mixed_precision.loss_scale_optimizer import _UnwrapPreventer
from tensorflow.python.keras.mixed_precision.loss_scale_optimizer import _op_in_graph_mode

from npu_device.npu_device import global_npu_ctx
from npu_device import gen_npu_ops
from npu_device.npu_device import npu_compat_function
from npu_device.distribute.hccl import all_reduce


@npu_compat_function
def _npu_finite_status_after_executed(executed_ops):
    if not isinstance(executed_ops, (tuple, list)):
        executed_ops = [executed_ops]
    with ops.get_default_graph()._attr_scope(
            {"_npu_loss_scale": attr_value_pb2.AttrValue(b=True)}):
        with tf.control_dependencies(executed_ops):
            current_status = gen_npu_ops.npu_alloc_float_status()
        assign_float_status = gen_npu_ops.npu_get_float_status(current_status)
        finite_status = gen_npu_ops.npu_clear_float_status(assign_float_status)
        if global_npu_ctx() and global_npu_ctx().workers_num > 1:
            with tf.control_dependencies([assign_float_status]):
                reduced_status = all_reduce(current_status, 'sum', fusion=0)
            return tf.reduce_all(tf.equal(reduced_status, finite_status))
        return tf.reduce_all(tf.equal(current_status, finite_status))


def _npu_compat_loss_scale_update(m, grads):
    grads = nest.flatten(grads)
    is_finite = _npu_finite_status_after_executed(grads)

    def update_if_finite_grads():
        def incr_loss_scale():
            incr_result_finite = tf.less(m.current_loss_scale, 3.4e+38 / m.multiplier)
            update_if_finite_fn = tf.cond(incr_result_finite,
                                          lambda: _op_in_graph_mode(
                                              m.current_loss_scale.assign(m.current_loss_scale * m.multiplier)),
                                          tf.no_op)
            return tf.group(update_if_finite_fn, m.counter.assign(0))

        return tf.cond(m.counter + 1 >= m.growth_steps,
                       incr_loss_scale,
                       lambda: _op_in_graph_mode(m.counter.assign_add(1)))

    def update_if_not_finite_grads():
        new_loss_scale = tf.maximum(m.current_loss_scale / m.multiplier, 1)
        return tf.group(m.counter.assign(0), m.current_loss_scale.assign(new_loss_scale))

    update_op = tf.cond(is_finite,
                        update_if_finite_grads,
                        update_if_not_finite_grads)
    should_apply_gradients = is_finite
    return update_op, should_apply_gradients


class NpuLossScaleOptimizer(tf.keras.mixed_precision.LossScaleOptimizer):
    """NPU implemented loss scale optimizer"""
    _HAS_AGGREGATE_GRAD = True

    def __init__(self, inner_optimizer, dynamic=True, initial_scale=None, dynamic_growth_steps=None):
        super().__init__(inner_optimizer, dynamic, initial_scale, dynamic_growth_steps)
        self._last_step_finite = variable_scope.variable(
            initial_value=False,
            name="npu_last_step_finite",
            dtype=tf.bool,
            trainable=False,
            use_resource=True,
            synchronization=variables.VariableSynchronization.AUTO,
            aggregation=variables.VariableAggregation.NONE
        )

    @property
    def last_step_finite(self):
        """Return property"""
        return self._last_step_finite

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        """Apply gradients on variables"""
        if global_npu_ctx() is None:
            super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

        grads_and_vars = tuple(grads_and_vars)  # grads_and_vars origin type is zip and can only be iter once
        grads = [g for g, _ in grads_and_vars]

        def apply_fn():
            wrapped_vars = _UnwrapPreventer([v for _, v in grads_and_vars])
            return self._apply_gradients(grads, wrapped_vars, name, experimental_aggregate_gradients)

        def do_not_apply_fn():
            return self._optimizer.iterations.assign_add(1, read_value=False)

        if self.dynamic:
            loss_scale_update_op, should_apply_grads = _npu_compat_loss_scale_update(self._loss_scale, grads)
        else:
            loss_scale_update_op = tf.no_op()
            should_apply_grads = _npu_finite_status_after_executed(grads)

        self._last_step_finite.assign(should_apply_grads)
        maybe_apply_op = smart_cond.smart_cond(should_apply_grads, apply_fn, do_not_apply_fn)
        return tf.group(maybe_apply_op, loss_scale_update_op)
