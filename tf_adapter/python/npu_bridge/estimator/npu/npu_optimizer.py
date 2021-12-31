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

"""
Optimizer that implements distributed gradient reduction for NPU.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras import optimizers
from tensorflow.python.platform import tf_logging as logging
from npu_bridge.estimator.npu.npu_common import NPUBasics
from npu_bridge.estimator.npu import util
from npu_bridge.hccl import hccl_ops
from npu_bridge.helper import helper

gen_npu_ops = helper.get_gen_ops()


def allreduce(tensor, average=True):
    """
    Perform an allreduce on a tf.Tensor or tf.IndexedSlices.

    Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
        The shape of the input must be identical across all ranks.
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.

    This function performs a bandwidth-optimal ring allreduce on the input
    tensor. If the input is an tf.IndexedSlices, the function instead does an
    allgather on the values and the indices, effectively doing an allreduce on
    the represented tensor.
    """
    basic = NPUBasics("")
    size = basic.size()
    # the tensor is the instance of tf.IndexedSlices
    if isinstance(tensor, tf.IndexedSlices):
        # For IndexedSlices, do two allgathers intead of an allreduce.
        logging.debug("HcomAllgather...")
        values = hccl_ops.allgather(tensor.values, size)
        indices = hccl_ops.allgather(tensor.indices, size)

        if values is None:
            raise ValueError('the result of tf.HcomAllgather([tensor.values]) is empty')
        if indices is None:
            raise ValueError('the result of tf.HcomAllgather([tensor.indices]) is empty')

        # To make this operation into an average, divide all gathered values by the size.
        rank_size = tf.cast(size, tensor.values.dtype)
        new_values = tf.div(values, rank_size) if average else values

        return tf.IndexedSlices(new_values, indices, dense_shape=tensor.dense_shape)

    logging.debug("HcomAllReduce...")
    summed_tensor = hccl_ops.allreduce(tensor, "sum")

    if summed_tensor is None:  # and summed_tensor:
        raise ValueError('the result of tf.DavinciAllreduce([tensor]) is empty')

    rank_size = tf.cast(size, dtype=tensor.dtype)
    new_tensor = tf.div(summed_tensor, rank_size) if average else summed_tensor

    return new_tensor


def reduce(tensor, root_rank, average=True, fusion=0, fusion_id=-1):
    """NPU implemented reduce"""
    basic = NPUBasics("")
    size = basic.size()
    # the tensor is the instance of tf.IndexedSlices
    if isinstance(tensor, tf.IndexedSlices):
        # For IndexedSlices, do two allgathers intead of a reduce.
        logging.debug("HcomAllgather...")
        values = hccl_ops.allgather(tensor.values, size)
        indices = hccl_ops.allgather(tensor.indices, size)

        if values is None:
            raise ValueError('the result of tf.HcomAllgather([tensor.values]) is empty')
        if indices is None:
            raise ValueError('the result of tf.HcomAllgather([tensor.indices]) is empty')

        # To make this operation into an average, divide all gathered values by the size.
        rank_size = tf.cast(size, tensor.values.dtype)
        new_values = tf.div(values, rank_size) if average else values

        return tf.IndexedSlices(new_values, indices, dense_shape=tensor.dense_shape)

    logging.debug("HcomReduce...")
    local_rank_id = os.getenv('DEVICE_ID')
    if local_rank_id is None or int(local_rank_id) < 0:
        raise ValueError('Please set the correct RANK_ID value, current RANK_ID is:', local_rank_id)

    summed_tensor = hccl_ops.reduce(tensor, "sum", root_rank, fusion, fusion_id)
    if summed_tensor is None:  # and summed_tensor:
        raise ValueError('the result of tf.DavinciReduce([tensor]) is empty')
    if root_rank != int(local_rank_id):
        return summed_tensor
    rank_size = tf.cast(size, dtype=tensor.dtype)
    new_tensor = tf.div(summed_tensor, rank_size) if average else summed_tensor
    return new_tensor


class NPUOptimizer(optimizer.Optimizer):
    """An optimizer that wraps another tf.Optimizer that can using an allreduce to
    average gradient values before applying gradients to model weights when
    'is_distributed' is True. And applies loss scaling in backprop when 'is_loss_scale'
    is True. 'is_tailing_optimization' is used to determine whether to enable
    communication tailing optimization to improve training performance,
    this setting only takes effect when 'is_distributed' is True.
    """

    def __init__(self, opt, loss_scale_manager=None, is_distributed=False, is_loss_scale=False,
                 is_tailing_optimization=False, name=None):
        """Construct a loss scaling optimizer.

        Args:
            opt: The actual optimizer that will be used to compute and apply the
                gradients. Must be an implementation of the
                `tf.compat.v1.train.Optimizer` interface.
            loss_scale_manager: A LossScaleManager object.
        """
        self._opt = opt
        self._loss_scale_manager = loss_scale_manager
        self._float_status = tf.constant([0.0], dtype=tf.float32)
        self._is_distributed = is_distributed
        self._is_loss_scale = is_loss_scale
        self._is_tailing_optimization = is_tailing_optimization
        self._is_overall_finite = None
        if is_loss_scale and loss_scale_manager is None:
            raise ValueError("is_loss_scale is True, loss_scale_manager can not be None")
        if name is None:
            name = "NPUOptimizer{}".format(type(opt).__name__)
        self._name = name
        super(NPUOptimizer, self).__init__(name=self._name, use_locking=False)

    def compute_gradients(self,
                          loss,
                          var_list=None,
                          gate_gradients=optimizer.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        """Compute gradients. See base class `tf.compat.v1.train.Optimizer`."""
        if self._is_loss_scale:
            loss_scale = self._loss_scale_manager.get_loss_scale()
            loss_value = loss() if callable(loss) else loss
            scaled_loss = loss_value * math_ops.cast(loss_scale, loss_value.dtype.base_dtype)
            self._float_status = gen_npu_ops.npu_alloc_float_status()
        else:
            scaled_loss = loss

        logging.debug("compute_gradients...")
        gradients = self._opt.compute_gradients(
            scaled_loss,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)
        if not self._is_distributed:
            if self._is_loss_scale:
                return self._down_scale(gradients, loss_scale)
            return gradients

        averaged_gradients = self._averaged_gradients(gradients)
        if self._is_loss_scale:
            return self._down_scale(averaged_gradients, loss_scale)
        return averaged_gradients

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients on variables"""
        if self._is_loss_scale:
            if not self._is_tailing_optimization:
                grads = (g for (g, _) in grads_and_vars)
                self._reduce_all(grads)

            def true_apply_grads_fn():
                return self._opt.apply_gradients(grads_and_vars, global_step, name)

            update_variables = control_flow_ops.cond(self._is_overall_finite,
                                                     true_apply_grads_fn,
                                                     gen_control_flow_ops.no_op)

            # Potentially adjust gradient scale in case of finite gradients.
            return control_flow_ops.group(
                update_variables,
                self._loss_scale_manager.update_loss_scale(self._is_overall_finite))
        return self._opt.apply_gradients(grads_and_vars, global_step, name)

    def _down_scale(self, grads_vars, loss_scale):
        grads_and_vars = []
        reciprocal_loss_scale = gen_math_ops.reciprocal(loss_scale)
        for grads, variables in grads_vars:
            if grads is not None:
                grads_and_vars.append((grads * math_ops.cast(reciprocal_loss_scale, grads.dtype.base_dtype), variables))
            else:
                grads_and_vars.append((grads, variables))
        return grads_and_vars

    def _reduce_all(self, grads):
        with tf.get_default_graph().control_dependencies(grads):
            local_float_status = gen_npu_ops.npu_get_float_status(self._float_status)
            cleared_float_status = gen_npu_ops.npu_clear_float_status(local_float_status)

        if self._is_distributed:
            with tf.get_default_graph().control_dependencies([local_float_status]):
                aggregated_float_status = hccl_ops.allreduce([self._float_status], "sum", fusion=0)
                self._is_overall_finite = math_ops.reduce_all(tf.equal(aggregated_float_status,
                                                                       cleared_float_status))
        else:
            self._is_overall_finite = math_ops.reduce_all(tf.equal(self._float_status,
                                                                   cleared_float_status))

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._opt.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._opt.variables(*args, **kwargs)

    def _averaged_gradients(self, gradients):
        averaged_gradients = []
        grads = []
        with tf.name_scope(self._name + "_Allreduce"):
            for grad, var in gradients:
                grads.append(grad)
                if self._is_loss_scale and (len(grads) == len(gradients)) and self._is_tailing_optimization:
                    self._reduce_all(grads)
                    with tf.get_default_graph().control_dependencies([self._is_overall_finite]):
                        avg_grad = allreduce(grad, True) if grad is not None else None
                        averaged_gradients.append((avg_grad, var))
                else:
                    avg_grad = allreduce(grad, True) if grad is not None else None
                    averaged_gradients.append((avg_grad, var))
        return averaged_gradients


class NPUDistributedOptimizer(tf.train.Optimizer):
    """
    An optimizer that wraps another tf.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.
    """

    def __init__(self, tf_optimizer,
                 is_weight_update_sharding=False,
                 name=None):
        """
        Construct a new DistributedOptimizer, which uses another optimizer
        under the hood for computing single-process gradient values and
        applying gradient updates after the gradient values have been averaged
        across all the hcom ranks.

        Args:
            tf_optimizer: Optimizer to use for computing gradients and applying updates.
            name: Optional name prefix for the operations created when applying
                gradients. Defaults to "Distributed" followed by the provided
                optimizer type.
                See Optimizer.__init__ for more info.
        """
        if name is None:
            name = "Distributed{}".format(type(tf_optimizer).__name__)
        self._optimizer = tf_optimizer
        self._is_weight_update_sharding = is_weight_update_sharding
        super(NPUDistributedOptimizer, self).__init__(name=name, use_locking=False)

    def compute_gradients(self, *args, **kwargs):
        """
        Compute gradients of all trainable variables.
        See Optimizer.compute_gradients() for more info.
        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        logging.debug("compute_gradients...")
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        rank_size = os.getenv('RANK_SIZE')
        if rank_size is None or int(rank_size) <= 1:
            return gradients

        averaged_gradients = []
        if self._is_weight_update_sharding and int(rank_size) <= len(gradients):
            averaged_gradients = self._averaged_gradients_if_weight_update_sharding(gradients, rank_size)
        elif self._is_weight_update_sharding and int(rank_size) > len(gradients):
            raise ValueError("The number of gradients is less than rank_size, "
                             "so weight_update_sharding cannot be executed")
        else:
            with tf.name_scope(self._name + "_Allreduce"):
                for grad, var in gradients:
                    avg_grad = allreduce(grad, True) if grad is not None else None
                    averaged_gradients.append((avg_grad, var))
        return averaged_gradients

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients on variables"""
        rank_size = os.getenv('RANK_SIZE')
        if rank_size is None or int(rank_size) <= 1:
            return self._optimizer.apply_gradients(grads_and_vars, global_step, name)

        if self._is_weight_update_sharding:
            return self._apply_gradients_if_weight_update_sharding(grads_and_vars, global_step, name)
        return self._optimizer.apply_gradients(grads_and_vars, global_step, name)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)

    def _apply_gradients_if_weight_update_sharding(self, grads_and_vars,
                                                   global_step, name):
        op_list = []
        local_rank_id = os.getenv('DEVICE_ID')
        if local_rank_id is None or int(local_rank_id) < 0:
            raise ValueError('Please set the correct RANK_ID value, current RANK_ID is:', local_rank_id)
        local_grads_and_vars = []
        for grad, var in grads_and_vars:
            rank_id = util.get_gid_by_weight(var)
            if rank_id >= 0 and rank_id == int(local_rank_id):
                local_grads_and_vars.append((grad, var))
        apply_res = self._optimizer.apply_gradients(local_grads_and_vars, global_step, name)
        with tf.get_default_graph().control_dependencies([apply_res]):
            with tf.name_scope(self._name + "_Broadcast_Weight_Update_Sharding"):
                for grad, var in grads_and_vars:
                    rank_id = util.get_gid_by_weight(var)
                    with tf.get_default_graph().control_dependencies(op_list):
                        outputs = hccl_ops.broadcast([var], rank_id, 0)
                    if outputs is not None:
                        op_list.append(outputs[0].op)
        for grad, var in grads_and_vars:
            rank_id = util.get_gid_by_weight(var)
            if rank_id >= 0 and rank_id != int(local_rank_id):
                op_list.append(grad)
        op_list.append(apply_res)
        return tf.group(op_list)

    def _averaged_gradients_if_weight_update_sharding(self, gradients, rank_size):
        averaged_gradients = []
        local_rank_id = os.getenv('DEVICE_ID')
        if local_rank_id is None or int(local_rank_id) < 0:
            raise ValueError('Please set the correct RANK_ID value, current RANK_ID is:', local_rank_id)
        util.add_grads_and_vars(gradients, int(rank_size))
        with tf.name_scope(self._name + "_Reduce_Weight_Update_Sharding"):
            for grad, var in gradients:
                rank_id = util.get_gid_by_grad(grad)
                avg_grad = reduce(grad, rank_id, True, 2, rank_id) if grad is not None else None
                averaged_gradients.append((avg_grad, var))
        return averaged_gradients


class KerasDistributeOptimizer(optimizer_v2.OptimizerV2):
    """
    An optimizer that wraps another keras Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.
    """

    def __init__(self, tf_optimizer, name="NpuKerasOptimizer", **kwargs):
        """
        Construct a new KerasDistributeOptimizer, which uses another optimizer
        under the hood for computing single-process gradient values and
        applying gradient updates after the gradient values have been averaged
        across all the hcom ranks.

        Args:
            tf_optimizer: Optimizer to use for get_updates gradients.
        """
        super(KerasDistributeOptimizer, self).__init__(name, **kwargs)
        self._optimizer = tf_optimizer
        old_get_gradient = self._optimizer.get_gradients

        def new_get_gradient(loss, params):
            grads = old_get_gradient(loss, params)
            rank_size = os.getenv('RANK_SIZE', '1')
            if rank_size is None or int(rank_size) <= 1:
                return grads
            averaged_grads = []
            with tf.name_scope(name + "_Allreduce"):
                for grad in grads:
                    avg_grad = allreduce(grad, True) if grad is not None else None
                    averaged_grads.append(avg_grad)
            return averaged_grads

        self._optimizer.get_gradients = new_get_gradient

    def get_updates(self, loss, params):
        """Get updated loss and parameters"""
        return self._optimizer.get_updates(loss, params)

    def _compute_gradients(self, loss, var_list, grad_loss=None):
        gradients = self._optimizer._compute_gradients(loss, var_list, grad_loss)
        rank_size = os.getenv('RANK_SIZE', '1')
        if rank_size is None or int(rank_size) <= 1:
            return gradients
        averaged_grads = []
        with tf.name_scope(self._name + "_Allreduce"):
            for grad, var in gradients:
                avg_grad = allreduce(grad, True) if grad is not None else None
                averaged_grads.append((avg_grad, var))
        return averaged_grads

    def apply_gradients(self, grads_and_vars, name=None):
        """Apply gradients on variables"""
        return self._optimizer.apply_gradients(grads_and_vars, name)

    def get_config(self):
        """Get optimizer configuration"""
        config = {
            "optimizer": self._optimizer
        }
        base_config = super(KerasDistributeOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def npu_distributed_optimizer_wrapper(tf_optimizer):
    """
    An optimizer that wraps Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.
    """
    if isinstance(tf_optimizer, str):
        tf_optimizer = optimizers.get(tf_optimizer)
    rank_size = os.getenv('RANK_SIZE')
    if hasattr(tf_optimizer, "compute_gradients"):
        org_compute_gradients = tf_optimizer.compute_gradients

        def _npu_compute_gradients(*args, **kwargs):
            """
            In DistributedOptimizer, compute_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = org_compute_gradients(*args, **kwargs)
            if rank_size is None or int(rank_size) <= 1:
                return gradients
            averaged_gradients = []
            with tf.name_scope("Npu_Distributed_optimizer_Allreduce"):
                for grad, var in gradients:
                    avg_grad = allreduce(grad, True) if grad is not None else None
                    averaged_gradients.append((avg_grad, var))
            return averaged_gradients

        tf_optimizer.compute_gradients = _npu_compute_gradients

    if hasattr(tf_optimizer, "get_gradients"):
        org_get_gradients = tf_optimizer.get_gradients

        def _npu_get_gradients(loss, params):
            grads = org_get_gradients(loss, params)
            if rank_size is None or int(rank_size) <= 1:
                return grads
            averaged_grads = []
            with tf.name_scope("Npu_Distributed_optimizer_get_grads_Allreduce"):
                for grad in grads:
                    avg_grad = allreduce(grad, True) if grad is not None else None
                    averaged_grads.append(avg_grad)
            return averaged_grads

        tf_optimizer.get_gradients = _npu_get_gradients

    if hasattr(tf_optimizer, "_compute_gradients"):
        org_compute_gradients = tf_optimizer._compute_gradients

        def _npu_compute_gradients(loss, var_list, grad_loss=None):
            gradients = org_compute_gradients(loss, var_list, grad_loss)
            if rank_size is None or int(rank_size) <= 1:
                return gradients
            averaged_grads = []
            with tf.name_scope("Npu_Distributed_optimizer_compute_grads_Allreduce"):
                for grad, var in gradients:
                    avg_grad = allreduce(grad, True) if grad is not None else None
                    averaged_grads.append((avg_grad, var))
            return averaged_grads

        tf_optimizer._compute_gradients = _npu_compute_gradients

    return tf_optimizer


def _npu_allreduce(values, reduction="mean", fusion=1, fusion_id=-1, group="hccl_world_group"):
    mean_reduce = False
    if reduction == "mean":
        mean_reduce = True
        reduction = "sum"

    reduced_values = []
    size = int(os.getenv("RANK_SIZE", "1"))
    for value in values:
        if isinstance(value, tf.IndexedSlices):
            # For IndexedSlices, do two allgathers intead of an allreduce.
            tensor_values = hccl_ops.allgather(value.values, size, group)
            tensor_indices = hccl_ops.allgather(value.indices, size, group)

            if tensor_values is None:
                raise ValueError('the result of tf.HcomAllgather([value.values]) is empty')
            if tensor_indices is None:
                raise ValueError('the result of tf.HcomAllgather([value.indices]) is empty')

            # To make this operation into an average, divide all gathered values by the size.
            rank_size = tf.cast(size, value.values.dtype)
            new_values = tf.div(tensor_values, rank_size) if mean_reduce else tensor_values

            reduced_values.append(tf.IndexedSlices(new_values, tensor_indices, dense_shape=value.dense_shape))
        else:
            summed_tensor = hccl_ops.allreduce(value, reduction, fusion, fusion_id, group)
            if summed_tensor is None:
                raise ValueError('the result of tf.DavinciAllreduce([tensor]) is empty')

            rank_size = tf.cast(size, dtype=value.dtype)
            reduced_values.append(tf.div(summed_tensor, rank_size) if mean_reduce else summed_tensor)
    return reduced_values


def npu_allreduce(values, reduction="mean", fusion=1, fusion_id=-1, group="hccl_world_group"):
    """NPU implemented allreduce"""
    if isinstance(values, (list, tuple,)):
        return _npu_allreduce(values, reduction, fusion, fusion_id, group)
    return _npu_allreduce([values], reduction, fusion, fusion_id, group)[0]
