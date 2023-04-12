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
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.training import optimizer
from tensorflow.python.training import adam
from tensorflow.python.training import adagrad
from tensorflow.python.training import training_ops
from tensorflow.python.training import training_util
from npu_bridge.embedding.embedding_resource import NpuEmbeddingResource
from npu_bridge.npu_cpu.npu_cpu_ops import gen_npu_cpu_ops

_GLOBAL_STEP_VALUE = 1
_ADAMW_BEAT1_POWER_VALUE = 0.9
_ADAMW_BEAT2_POWER_VALUE = 0.99


class AdamOptimizer(adam.AdamOptimizer):
    @property
    def embedding_dims(self):
        return self._embedding_dims

    @embedding_dims.setter
    def embedding_dims(self, val):
        self._embedding_dims = val

    @property
    def max_nums(self):
        return self._max_nums

    @max_nums.setter
    def max_nums(self, val):
        self._max_nums = val

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        epsilon = self._call_if_callable(self._epsilon)
        self._beta1_t_list = []
        self._beta2_t_list = []
        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

    def _resource_apply_sparse(self, grad, var, indices):
        if isinstance(var, NpuEmbeddingResource):
            beta1 = self._call_if_callable(self._beta1)
            beta2 = self._call_if_callable(self._beta2)
            self._beta1_t = ops.convert_to_tensor(beta1, name="beta1" + str(self.table_idx))
            self._beta2_t = ops.convert_to_tensor(beta2, name="beta2" + str(self.table_idx))
            self._beta1_t_list.append(self._beta1_t)
            self._beta2_t_list.append(self._beta2_t)
            beta1_power, beta2_power = self._get_beta_accumulators()
            self.table_idx += 1
            result = gen_npu_cpu_ops.embedding_apply_adam(var.handle, beta1_power, beta2_power,
                                                          math_ops.cast(self._lr_t, grad.dtype),
                                                          math_ops.cast(self._beta1_t, grad.dtype),
                                                          math_ops.cast(self._beta2_t, grad.dtype),
                                                          math_ops.cast(self._epsilon_t, grad.dtype),
                                                          grad,
                                                          indices,
                                                          ops.convert_to_tensor(_GLOBAL_STEP_VALUE),
                                                          self._embedding_dims)
            result.op._set_attr("_embedding_dim", attr_value_pb2.AttrValue(i=self._embedding_dims))
            result.op._set_attr("_max_num", attr_value_pb2.AttrValue(i=self._max_nums))
            return result
        else:
            return self._apply_sparse_shared(grad, var, indices, self._resource_scatter_add)

    def _create_slots(self, var_list):
        self.table_num = 1
        self.table_idx = 0
        self._var_idx = var_list[0].name
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power" + str(self._var_idx), colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power" + str(self._var_idx), colocate_with=first_var)

        for v in var_list:
            if not isinstance(v, NpuEmbeddingResource):
                self._zeros_slot(v, "m", self._name)
                self._zeros_slot(v, "v", self._name)

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
        return (self._get_non_slot_variable("beta1_power" + str(self._var_idx), graph=graph),
                self._get_non_slot_variable("beta2_power" + str(self._var_idx), graph=graph))

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        self.table_num = 0
        self.table_idx = 0
        finish_output = []
        with ops.control_dependencies(update_ops):
            beta1_power_list = []
            beta2_power_list = []
            for k in update_ops:
                beta1_power, beta2_power = self._get_beta_accumulators()
                beta1_power_list.append(beta1_power)
                beta2_power_list.append(beta2_power)
                self.table_idx += 1
        for idx in range(len(update_ops)):
            beta1_power = beta1_power_list[idx]
            beta2_power = beta2_power_list[idx]
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t_list[idx], use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t_list[idx], use_locking=self._use_locking)
            new_update_op = []
            new_update_op.append(update_ops[idx])
            finish_output.append(control_flow_ops.group(
                *new_update_op + [update_beta1, update_beta2], name=name_scope + str(idx)))
        return finish_output


class AdagradOptimizer(adagrad.AdagradOptimizer):
    @property
    def embedding_dims(self):
        return self._embedding_dims

    @embedding_dims.setter
    def embedding_dims(self, val):
        self._embedding_dims = val

    @property
    def max_nums(self):
        return self._max_nums

    @max_nums.setter
    def max_nums(self, val):
        self._max_nums = val

    def _resource_apply_sparse(self, grad, var, indices):
        if isinstance(var, NpuEmbeddingResource):
            result = gen_npu_cpu_ops.embedding_apply_ada_grad(var.handle,
                                                              math_ops.cast(self._learning_rate_tensor, grad.dtype),
                                                              grad,
                                                              indices,
                                                              ops.convert_to_tensor(_GLOBAL_STEP_VALUE),
                                                              self._embedding_dims)
            result.op._set_attr("_embedding_dim", attr_value_pb2.AttrValue(i=self._embedding_dims))
            result.op._set_attr("_max_num", attr_value_pb2.AttrValue(i=self._max_nums))
            return result
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


class AdamWOptimizer(optimizer.Optimizer):
    """A basic adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate=0.01,
                 weight_decay=0.004,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 max_grad_norm=0.5,
                 amsgrad: bool = False,
                 maximize: bool = False,
                 name="AdamWOptimizer"):
        """Construct a AdamW optimizer."""
        super(AdamWOptimizer, self).__init__(False, name)
        if (learning_rate is None) or (weight_decay is None) or (beta_1 is None) or (beta_2 is None):
            raise ValueError("learning_rate, weight decay, beta_1 and beta_2 can not be None.")
        if (epsilon is None) or (amsgrad is None) or (maximize is None):
            raise ValueError("epsilon, amsgrad and maximize can not be None.")
        if (max_grad_norm is None) and amsgrad:
            raise ValueError("if amsgrad is True, max_grad_norm can not be None.")
        self._weight_decay = weight_decay
        self._lr = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self._max_grad_norm = max_grad_norm
        self._amsgrad = amsgrad
        self._maximize = maximize

        # Tensor versions of the constructor arguments, created in _prepare()
        self._weight_decay_t = None
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._max_grad_norm_t = None
        self._beta1_power_t = None
        self._beta2_power_t = None

    @property
    def embedding_dims(self):
        return self._embedding_dims

    @embedding_dims.setter
    def embedding_dims(self, val):
        self._embedding_dims = val

    @property
    def max_nums(self):
        return self._max_nums

    @max_nums.setter
    def max_nums(self, val):
        self._max_nums = val

    def _prepare(self):
        beta1_power = self._call_if_callable(_ADAMW_BEAT1_POWER_VALUE)
        beta2_power = self._call_if_callable(_ADAMW_BEAT2_POWER_VALUE)
        lr = self._call_if_callable(self._lr)
        weight_decay = self._call_if_callable(self._weight_decay)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)
        max_grad_norm = self._call_if_callable(self._max_grad_norm)

        self._beta1_power_t = ops.convert_to_tensor(beta1_power, name="beta1_power")
        self._beta2_power_t = ops.convert_to_tensor(beta2_power, name="beta2_power")
        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._weight_decay_t = ops.convert_to_tensor(weight_decay, name="weight_decay")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
        self._max_grad_norm_t = ops.convert_to_tensor(max_grad_norm, name="max_grad_norm")

    def _resource_apply_sparse(self, grad, var, indices):
        if isinstance(var, NpuEmbeddingResource):
            result = gen_npu_cpu_ops.embedding_apply_adam_w(var.handle,
                                                            beta1_power=
                                                            math_ops.cast(self._beta1_power_t, grad.dtype),
                                                            beta2_power=
                                                            math_ops.cast(self._beta1_power_t, grad.dtype),
                                                            lr=math_ops.cast(self._lr_t, grad.dtype),
                                                            weight_decay=
                                                            math_ops.cast(self._weight_decay_t, grad.dtype),
                                                            beta1=math_ops.cast(self._beta1_t, grad.dtype),
                                                            beta2=math_ops.cast(self._beta2_t, grad.dtype),
                                                            epsilon=math_ops.cast(self._epsilon_t, grad.dtype),
                                                            grad=grad,
                                                            keys=indices,
                                                            max_grad_norm=
                                                            math_ops.cast(self._max_grad_norm_t, grad.dtype),
                                                            amsgrad=self._amsgrad,
                                                            maximize=self._maximize,
                                                            embedding_dim=self._embedding_dims)
            result.op._set_attr("_embedding_dim", attr_value_pb2.AttrValue(i=self._embedding_dims))
            result.op._set_attr("_max_num", attr_value_pb2.AttrValue(i=self._max_nums))
            return result
        else:
            raise TypeError("Variable is not NpuEmbeddingResource type, please check.")
