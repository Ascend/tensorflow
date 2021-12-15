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
# Description: Common depends and micro defines for and only for data preprocess module

import os
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from npu_device.distribute import hccl
from npu_device.npu_device import npu_compat_function
from absl import logging

from npu_device.npu_device import global_npu_ctx


class GroupingVars():
    def __init__(self, variables, rank_size):
        self._vars = []
        for var in variables:
            if var is not None:
                item = self._GradDivisionItem(var)
                self._vars.append(item)
        self._fair_division(rank_size)

    def _fair_division(self, number):
        if number > len(self._vars) or number < 0:
            raise ValueError("'number' is greater than the number of vars or 'number' is less than 0. ")
        elif number == len(self._vars):
            for i in range(len(self._vars)):
                self._vars[i].root_rank_id = i
            return

        left_vars = list(self._vars)

        def get_average(size):
            large_number_list = []
            average_size = 0
            res = 0
            for item in left_vars:
                res += item.size
            if size == 1:
                return res
            while True:
                find_large_number = False
                left = size - len(large_number_list)
                if left <= 1:
                    return res
                average_size = res // left
                for item in left_vars:
                    if item not in large_number_list and item.root_rank_id < 0 and item.size > res - item.size:
                        find_large_number = True
                        res -= item.size
                        large_number_list.append(item)
                if not find_large_number:
                    break
            return average_size

        j = -1
        while True:
            j = j + 1
            total_number = number - j
            if total_number == 0:
                break
            while left_vars[0].root_rank_id >= 0:
                left_vars.pop(0)
            average_size = get_average(total_number)
            current_group_size = 0
            for i in range(0, len(left_vars) - total_number + 1):
                if current_group_size + left_vars[i].size <= average_size:
                    left_vars[i].root_rank_id = j
                    current_group_size += left_vars[i].size
                else:
                    if current_group_size <= 0:
                        left_vars[i].root_rank_id = j
                    elif (current_group_size + left_vars[i].size - average_size) <= (average_size - current_group_size):
                        left_vars[i].root_rank_id = j
                    break
        return

    class _GradDivisionItem():
        def __init__(self, var):
            self.var = var
            self.size = self.__get_size()
            self.root_rank_id = -1

        def __get_size(self):
            size = 1
            var_shape = self.var.shape
            if len(var_shape) <= 0:
                return 0
            for i in range(len(var_shape)):
                size = size * int(var_shape[i])
            size = size * self.var.dtype.size
            return size

    def get_gid_by_var(self, var):
        gid = -1
        for item in self._vars:
            if item.var is var:
                gid = item.root_rank_id
        return gid


@npu_compat_function
def grouping_gradients_apply(apply_func, grads_and_vars, *args, **kwargs):
    if global_npu_ctx() is None or not global_npu_ctx().is_cluster_worker():
        return apply_func(grads_and_vars, *args, **kwargs)

    grads_and_vars = tuple(grads_and_vars)  # grads_and_vars origin type is zip and can only be iter once

    op_list = []

    local_rank_id = global_npu_ctx().worker_id
    variables = []
    for _, var in grads_and_vars:
        variables.append(var)
    grouping_vars = GroupingVars(variables, global_npu_ctx().workers_num)
    local_grads_and_vars = []
    for grad, var in grads_and_vars:
        rank_id = grouping_vars.get_gid_by_var(var)
        if rank_id >= 0 and rank_id == local_rank_id:
            local_grads_and_vars.append((grad, var))
    apply_res = apply_func(local_grads_and_vars, *args, **kwargs)
    with ops.get_default_graph()._attr_scope(
            {"_weight_update_grouping": attr_value_pb2.AttrValue(b=True)}):
        for i in range(len(variables)):
            var = variables[i]
            rank_id = grouping_vars.get_gid_by_var(var)
            hccl.broadcast([var], rank_id, 0)
    for grad, var in grads_and_vars:
        rank_id = grouping_vars.get_gid_by_var(var)
        if rank_id >= 0 and rank_id != local_rank_id:
            op_list.append(grad)
    op_list.append(apply_res)
    return tf.group(op_list)


@npu_compat_function
def grouping_broadcast(variables):
    if global_npu_ctx() is None or not global_npu_ctx().is_cluster_worker():
        logging.info("Skip grouping broadcast as current process is not npu cluster worker")
        return variables
    grouping_vars = GroupingVars(variables, global_npu_ctx().workers_num)
    for var in variables:
        rank_id = grouping_vars.get_gid_by_var(var)
        hccl.broadcast([var], rank_id, 0)
