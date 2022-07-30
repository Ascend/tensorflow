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

"""NPU hccl functions"""

from npu_device.distribute import hccl_ops
from npu_device.distribute.npu_callbacks import NPUBroadcastGlobalVariablesCallback
from npu_device.npu_device import global_npu_ctx
from npu_device.npu_device import npu_compat_function
from npu_device.utils.scope import npu_gradients_scope
from npu_device.utils.scope import npu_optimizer_scope

import tensorflow as tf
from absl import logging


def shard_and_rebatch_dataset(dataset, global_bs):
    """Generate shard of dataset and rebatch it"""
    if global_npu_ctx() is None or global_npu_ctx().workers_num <= 1:
        return dataset, global_bs
    if global_bs % global_npu_ctx().workers_num != 0:
        raise ValueError('Batch size must be divisible by num npus: {}'.format(global_npu_ctx().workers_num))

    batch_size = int(global_bs) / global_npu_ctx().workers_num
    dataset = dataset.shard(global_npu_ctx().workers_num, global_npu_ctx().worker_id)

    return dataset, int(batch_size)


@npu_compat_function
def _all_reduce(values, reduction, fusion, fusion_id, group):
    workers_num = global_npu_ctx().workers_num

    mean_reduce = False
    if reduction == 'mean':
        mean_reduce = True
        reduction = 'sum'

    reduced_values = []
    for value in values:
        reduced_value = hccl_ops.allreduce(value, reduction, fusion, fusion_id, group)
        is_float = reduced_value.dtype in (tf.float16, tf.float32, tf.float64)
        if is_float:
            typed_workers_num = tf.cast(1.0 / float(workers_num), reduced_value.dtype)
        else:
            typed_workers_num = tf.cast(workers_num, reduced_value.dtype)
        with tf.control_dependencies([tf.group(*values)]):
            if mean_reduce:
                if is_float:
                    reduced_values.append(tf.multiply(reduced_value, typed_workers_num))
                else:
                    reduced_values.append(tf.divide(reduced_value, typed_workers_num))
            else:
                reduced_values.append(tf.multiply(reduced_value, tf.cast(1, reduced_value.dtype)))
    return reduced_values


def all_reduce(values, reduction="mean", fusion=1, fusion_id=-1, group="hccl_world_group"):
    """NPU implemented all_reduce"""
    if global_npu_ctx() is None or not global_npu_ctx().is_cluster_worker():
        logging.info("Skip all reduce as current process is not npu cluster worker")
        return values

    if isinstance(values, (list, tuple,)):
        is_list_value = True
    else:
        is_list_value = False
        values = [values]
    reduced_values = _all_reduce([v for v in values if v is not None], reduction, fusion, fusion_id, group)
    results = [None if v is None else reduced_values.pop(0) for v in values]
    return results if is_list_value else results[0]


@npu_compat_function
def _broadcast(values, root_rank, fusion, fusion_id, group):
    for value in values:
        value.assign(hccl_ops.broadcast([value], root_rank, fusion, fusion_id, group)[0])


def broadcast(values, root_rank=0, fusion=2, fusion_id=0, group="hccl_world_group"):
    """Broadcast value among cluster"""
    if global_npu_ctx() is None or not global_npu_ctx().is_cluster_worker():
        logging.info("Skip broadcast as current process is not npu cluster worker")
        return
    if isinstance(values, (list, tuple,)):
        _broadcast(values, root_rank, fusion, fusion_id, group)
    else:
        _broadcast([values], root_rank, fusion, fusion_id, group)


def npu_distributed_keras_optimizer_wrapper(optimizer, reduce_reduction="mean", fusion=1, fusion_id=-1,
                                            group="hccl_world_group"):
    """NPU implemented keras optimizer"""
    optimizer = tf.keras.optimizers.get(optimizer)
    org_apply_gradients = optimizer.apply_gradients
    org_get_gradient = optimizer.get_gradients
    org_compute_gradients = optimizer._compute_gradients

    def _npu_distribute_apply_gradients(grads_and_vars, *args, **kwargs):
        grads, variables = zip(*grads_and_vars)
        with npu_optimizer_scope():
            return org_apply_gradients(zip(all_reduce(grads, reduce_reduction, fusion, fusion_id, group), variables),
                                       *args, **kwargs)

    def _npu_get_gradients(*args, **kwargs):
        with npu_gradients_scope():
            return org_get_gradient(*args, **kwargs)

    def _npu_compute_gradients(*args, **kwargs):
        with npu_gradients_scope():
            return org_compute_gradients(*args, **kwargs)

    optimizer.apply_gradients = _npu_distribute_apply_gradients
    optimizer.get_gradient = _npu_get_gradients
    optimizer._compute_gradients = _npu_compute_gradients
    return optimizer


def npu_callbacks_append(callbacks_list=()):
    """Appand NPU callback functions"""
    if not isinstance(callbacks_list, list):
        callbacks_list = []
    callbacks_list.append(NPUBroadcastGlobalVariablesCallback(0))
    return callbacks_list