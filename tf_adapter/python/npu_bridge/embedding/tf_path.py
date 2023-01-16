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

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer as embeddingOptimizer
from npu_bridge.embedding.embedding_resource import NpuEmbeddingResource


class _NpuEmbeddingResourceProcessor(embeddingOptimizer._OptimizableVariable):
    """Processor for dense NpuEmbeddingResourceProcessor."""

    def __init__(self, v):
        self._v = v

    def target(self):
        return self._v

    def update_op(self, optimizer, g):
        return optimizer._resource_apply_sparse(g.values, self._v, g.indices)


def _get_processor(v):
    """The processor of v."""
    if context.executing_eagerly():
        if isinstance(v, ops.Tensor):
            return embeddingOptimizer._TensorProcessor(v)
        else:
            return embeddingOptimizer._DenseResourceVariableProcessor(v)
    if isinstance(v, NpuEmbeddingResource):
        return _NpuEmbeddingResourceProcessor(v)
    if resource_variable_ops.is_resource_variable(v) and not v._in_graph_mode:  # pylint: disable=protected-access
        # True if and only if `v` was initialized eagerly.
        return embeddingOptimizer._DenseResourceVariableProcessor(v)
    if v.op.type == "VarHandleOp":
        return embeddingOptimizer._DenseResourceVariableProcessor(v)
    if isinstance(v, variables.Variable):
        return embeddingOptimizer._RefVariableProcessor(v)
    if isinstance(v, ops.Tensor):
        return embeddingOptimizer._TensorProcessor(v)

    raise NotImplementedError("Trying to optimize unsupported type ", v)


def path_on_tf():
    embeddingOptimizer._get_processor = _get_processor


