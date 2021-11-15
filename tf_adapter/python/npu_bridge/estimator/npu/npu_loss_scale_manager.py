#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

"""LossScaleManager classes for mixed precision training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import state_ops
from tensorflow.contrib.mixed_precision.python import loss_scale_manager as lsm_lib


class FixedLossScaleManager(lsm_lib.FixedLossScaleManager):
    """Loss scale manager with a fixed loss scale.
    """

    def __init__(self, loss_scale, enable_overflow_check=True):
        """Creates the fixed loss scale manager.
        """
        if loss_scale < 1:
            raise ValueError("loss scale must be at least 1.")
        self._loss_scale = ops.convert_to_tensor(loss_scale, dtype=dtypes.float32, name="loss_scale")
        self._enable_overflow_check = enable_overflow_check
        super(FixedLossScaleManager, self).__init__(loss_scale=loss_scale)

    def get_enable_overflow_check(self):
        return self._enable_overflow_check


class ExponentialUpdateLossScaleManager(lsm_lib.ExponentialUpdateLossScaleManager):
    """Loss scale manager uses an exponential update strategy.
    """

    def __init__(self,
                 init_loss_scale,
                 incr_every_n_steps,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2,
                 decr_ratio=0.8):
        """Constructor of exponential-update loss scale manager.
        """
        super(ExponentialUpdateLossScaleManager, self).__init__(
            init_loss_scale=init_loss_scale,
            incr_every_n_steps=incr_every_n_steps,
            decr_every_n_nan_or_inf=decr_every_n_nan_or_inf,
            incr_ratio=incr_ratio,
            decr_ratio=decr_ratio)

    def update_loss_scale(self, finite_grads):
        """Updates loss scale based on if gradients are finite in current step."""

        def update_if_finite_grads():
            def incr_loss_scale():
                incr_result_finite = gen_math_ops.less(self._loss_scale, (3.4e+38) / self._incr_ratio)
                new_loss_scale = control_flow_ops.cond(
                    incr_result_finite,
                    lambda: self._loss_scale * self._incr_ratio,
                    lambda: self._loss_scale)
                update_op = state_ops.assign(self._loss_scale, new_loss_scale)
                return control_flow_ops.group(update_op, self._reset_stats())

            is_incr_good_steps = self._num_good_steps + 1 >= self._incr_every_n_steps
            return control_flow_ops.cond(is_incr_good_steps, incr_loss_scale,
                                         lambda: state_ops.assign_add(self._num_good_steps, 1).op)

        def update_if_not_finite_grads():
            def decr_loss_scale():
                new_loss_scale = gen_math_ops.maximum(1., self._loss_scale * self._decr_ratio)
                update_op = state_ops.assign(self._loss_scale, new_loss_scale)
                return control_flow_ops.group(update_op, self._reset_stats())

            def only_update_steps():
                return control_flow_ops.group(state_ops.assign_add(self._num_bad_steps, 1),
                                              state_ops.assign(self._num_good_steps, 0))

            is_incr_bad_steps = self._num_bad_steps + 1 >= self._decr_every_n_nan_or_inf
            return control_flow_ops.cond(is_incr_bad_steps, decr_loss_scale, only_update_steps)

        return control_flow_ops.cond(finite_grads, update_if_finite_grads, update_if_not_finite_grads)
