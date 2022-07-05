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

"""Optimizer for mixed precision training for Davinci NPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.contrib.mixed_precision.python import loss_scale_optimizer as lso
from npu_bridge.hccl import hccl_ops
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.helper import helper

gen_npu_ops = helper.get_gen_ops()


class NPULossScaleOptimizer(lso.LossScaleOptimizer):
    """NPU implemented loss scale optimizer"""
    def __init__(self, opt, loss_scale_manager, is_distributed=False):
        """Construct a loss scaling optimizer.
        """
        self._opt = opt
        self._loss_scale_manager = loss_scale_manager
        self._float_status = tf.constant([0.0], dtype=tf.float32)
        self._is_distributed = is_distributed
        self._name = "NPULossScaleOptimizer{}".format(type(optimizer).__name__)
        super(NPULossScaleOptimizer, self).__init__(opt=opt, loss_scale_manager=loss_scale_manager)

    def __getattr__(self, attr):
        return getattr(self._opt, attr)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients. See base class `tf.compat.v1.train.Optimizer`."""
        if self._enable_overflow_check():
            with tf.name_scope(self._name):
                grads = []
                for (g, _) in grads_and_vars:
                    if g is not None:
                        grads.append(g)
                with tf.get_default_graph().control_dependencies(grads):
                    local_float_status = gen_npu_ops.npu_get_float_status_v2()
                with tf.get_default_graph().control_dependencies([local_float_status]):
                    cleared_float_status = gen_npu_ops.npu_clear_float_status_v2()

            if self._is_distributed:
                aggregated_float_status = hccl_ops.allreduce([local_float_status], "sum", fusion=0)
                with tf.get_default_graph().control_dependencies([cleared_float_status]):
                    op = tf.equal(aggregated_float_status, 0)
                    is_overall_finite = math_ops.reduce_all(op,
                                                            name="overflow_status_reduce_all")
            else:
                with tf.get_default_graph().control_dependencies([cleared_float_status]):
                    op_ = tf.equal(0, local_float_status)
                is_overall_finite = math_ops.reduce_all(op_,
                                                        name="overflow_status_reduce_all")
        else:
            is_overall_finite = tf.constant(True, dtype=tf.bool)

        def true_apply_grads_fn():
            return self._opt.apply_gradients(grads_and_vars, global_step, name)

        update_variables = control_flow_ops.cond(is_overall_finite,
                                                 true_apply_grads_fn,
                                                 gen_control_flow_ops.no_op)

        # Potentially adjust gradient scale in case of finite gradients.
        return control_flow_ops.group(update_variables,
                                      self._loss_scale_manager.update_loss_scale(is_overall_finite))

    def _enable_overflow_check(self):
        if isinstance(self._loss_scale_manager, FixedLossScaleManager) or \
                issubclass(type(self._loss_scale_manager), FixedLossScaleManager):
            return self._loss_scale_manager.get_enable_overflow_check()
        return True
