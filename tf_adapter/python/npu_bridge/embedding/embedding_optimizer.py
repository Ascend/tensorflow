#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import adam
from tensorflow.python.training import adagrad
from tensorflow.python.training import training_ops
from tensorflow.python.training import training_util
from npu_bridge.embedding.embedding_resource import NpuEmbeddingResource
from npu_bridge.npu_cpu.npu_cpu_ops import gen_npu_cpu_ops

_GLOBAL_STEP_VALUE = 1


class AdamOptimizer(adam.AdamOptimizer):
    @property
    def embedding_dims(self):
        return self._embedding_dims

    @embedding_dims.setter
    def embedding_dims(self, val):
        self._embedding_dims = val

    def _resource_apply_sparse(self, grad, var, indices):
        if isinstance(var, NpuEmbeddingResource):
            beta1_power, beta2_power = self._get_beta_accumulators()
            return gen_npu_cpu_ops.embedding_apply_adam(var.handle, beta1_power, beta2_power,
                                                        math_ops.cast(self._lr_t, grad.dtype),
                                                        math_ops.cast(self._beta1_t, grad.dtype),
                                                        math_ops.cast(self._beta2_t, grad.dtype),
                                                        math_ops.cast(self._epsilon_t, grad.dtype),
                                                        grad,
                                                        indices,
                                                        ops.convert_to_tensor(_GLOBAL_STEP_VALUE),
                                                        self._embedding_dims)
        else:
            return self._apply_sparse_shared(grad, var, indices, self._resource_scatter_add)

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

        for v in var_list:
            if not isinstance(v, NpuEmbeddingResource):
                self._zeros_slot(v, "m", self._name)
                self._zeros_slot(v, "v", self._name)


class AdagradOptimizer(adagrad.AdagradOptimizer):
    @property
    def embedding_dims(self):
        return self._embedding_dims

    @embedding_dims.setter
    def embedding_dims(self, val):
        self._embedding_dims = val

    def _resource_apply_sparse(self, grad, var, indices):
        if isinstance(var, NpuEmbeddingResource):
            return gen_npu_cpu_ops.embedding_apply_ada_grad(var.handle,
                                                           math_ops.cast(self._learning_rate_tensor, grad.dtype),
                                                           grad,
                                                           indices,
                                                           ops.convert_to_tensor(_GLOBAL_STEP_VALUE),
                                                           self._embedding_dims)
        else:
            return self.training_ops.resource_sparse_apply_adagrad(var.handle, grad.handle,
                                                                   math_ops.cast(self._learning_rate_tensor,
                                                                                 grad.dtype),
                                                                   grad, indices,
                                                                   use_locking=self._use_locking)

    def _create_slots(self, var_list):
        for v in var_list:
            if not isinstance(v, NpuEmbeddingResource):
                dtype = v.dtype.base_dtype
                if v.get_shape().is_fully_defined():
                    init = init_ops.constant_initializer(self._initial_accumulator_value,
                                                         dtype=dtype)
                else:
                    init = self._init_constant_op(v, dtype)
                self._get_or_make_slot_with_initializer(v, init, v.get_shape(), dtype,
                                                        "accumulator", self._name)
