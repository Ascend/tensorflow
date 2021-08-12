#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
Config the non npu compilation scope for NPU in mix compute mode.
"""
import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.util import compat
from npu_bridge.estimator.npu.npu_config import NpuExecutePlacement

@contextlib.contextmanager
def without_npu_compile_scope():
    """
    Enable the non npu compilation of operators within the scope.
    """
    attrs = {
        "_without_npu_compile": attr_value_pb2.AttrValue(b=True)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def npu_variable_scope(placement=NpuExecutePlacement.ALL):
    """
    Enable the node in the scope adding _variable_placement attr.
    """
    if placement not in NpuExecutePlacement:
        raise ValueError("placement vaule must be in NpuExecutePlacement's vaule")
    attrs = {
        "_variable_placement": attr_value_pb2.AttrValue(s=compat.as_bytes(placement.value))
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def keep_dtype_scope():
    """
    Specify which layers retain the original precision.
    """
    attrs = {
        "_keep_dtype": attr_value_pb2.AttrValue(i=1)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def npu_stage_scope(stage):
    """
    Enable the node in the scope adding _stage_level attr.
    """
    attrs = {
        "_stage_level": attr_value_pb2.AttrValue(i=stage)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield

@contextlib.contextmanager
def npu_mem_type_scope():
    """
    Enable the node in the scope adding _output_memory_type attr.
    """
    attrs = {
        "_output_memory_type": attr_value_pb2.AttrValue(i=1)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


@contextlib.contextmanager
def npu_weight_prefetch_scope(buffer_pool_id=0, buffer_pool_size=536870912):
    """
    Enable the PREFETCH node in the scope to use buffer pool memory.
    buffer_pool_id: Specifies the id of buffer pool to enable,
                    it is a integer, default is 0;
    buffer_pool_size: Specifies the size of this buffer pool in bytes,
                      default is 512MB.

    Use constraints:
    1. BufferPoolMemory is only supported for PREFETCH node with single
       input and single output;
    2. Buffer pool size of the same ID must be the same;
    3. The size of the buffer pool should be able to meet the requirements
       of the PREFETCH node with the largest memory (note that alignment
       and complement are included, for example, 512 bytes alignment of
       the HCOM node with an additional 512 bytes before and after each);
    4. Prefetch is not supported if it is located in a subgraph or
       in a control flow branch.
    """
    attrs = {
        "_buffer_pool_id": attr_value_pb2.AttrValue(i=buffer_pool_id),
        "_buffer_pool_size": attr_value_pb2.AttrValue(i=buffer_pool_size)
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield

@contextlib.contextmanager
def op_specified_engine_scope(engine_name, kernel_lib_name):
    """
    Enable the node in the scope adding _specified_engine_name and _specified_kernel_lib_name attr.
    """
    attrs = {
        "_specified_engine_name": attr_value_pb2.AttrValue(s=compat.as_bytes(engine_name)),
        "_specified_kernel_lib_name": attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_lib_name))
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield
