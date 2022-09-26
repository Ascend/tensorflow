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

"""NPU callback functions"""

import os
import weakref
from tensorflow.python import keras
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf
from npu_device.distribute import hccl

# mapping id(var) to weakref.ref(var), which id(var) return memory address of var
broadcast_registry = weakref.WeakValueDictionary()


def broadcast_helper(variables, root_rank=0):
    """Broadcast trainable variables, and register to avoid repetitive processing"""
    candicates = []
    for var in variables:
        if hasattr(var, '_cast_dtype') and getattr(var, "_cast_dtype") != var.dtype:
            continue
        if id(var) not in broadcast_registry:
            candicates.append(var)
    if candicates:
        hccl.broadcast(candicates)
        for value in candicates:
            broadcast_registry[id(value)] = value


def broadcast_keras_model(model, root_rank=0):
    """Broadcast trainable variables of keras Model"""
    if not isinstance(model, tf.keras.Model):
        return model
    if model.built:
        broadcast_helper(model.trainable_variables, root_rank)
    return model


class NPUBroadcastGlobalVariablesCallback(keras.callbacks.Callback):
    """
    Keras Callback that will broadcast all global variables from root rank
    to all other processes during initialization.
    This is necessary to ensure consistent initialization of all workers when
    training is started with random weights or restored from a checkpoint.
    """

    def __init__(self, root_rank):
        """
        Construct a new BroadcastGlobalVariablesCallback that will broadcast all
        global variables from root rank to all other processes during initialization.
        Args:
            root_rank: Rank that will send data, other ranks will receive data.
        """
        super(NPUBroadcastGlobalVariablesCallback, self).__init__()
        self.root_rank = root_rank
        self.broadcast_done = False

        def on_batch_begin(self, batch, logs=None):
            """This function is called when every batch begins"""
            if self.broadcast_done:
                return

            rank_size = os.getenv("RANK_SIZE", "1")
            if int(rank_size) > 1:
                broadcast_helper(self.model.trainable_variables, self.root_rank)

            self.broadcast_done = True


class NpuBroadcastScopeContext(object):
    """
    A context manager, used to record and broadcast trainable varibles in its
    scope automatically.
    """
    _is_in_scope = False

    def __init__(self, org_scope_ctx):
        self.org_scope_ctx = org_scope_ctx
        self.enter_scope_again_count = 0

        def _variable_broadcast_creator(next_creator, **kwargs):
            var = next_creator(**kwargs)
            if var.trainable:
                broadcast_helper([var])
            return var

        # all var created in scope ctx will auto-broadcast
        self.broadcast_scope = tf.variable_creator_scope(
            _variable_broadcast_creator)

    def __enter__(self):
        if not NpuBroadcastScopeContext._is_in_scope:
            NpuBroadcastScopeContext._is_in_scope = True
            if self.broadcast_scope:
                self.broadcast_scope.__enter__()
        else:
            self.enter_scope_again_count += 1 # handle with reentry
        if self.org_scope_ctx:
            self.org_scope_ctx.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        if self.org_scope_ctx:
            self.org_scope_ctx.__exit__(exception_type, exception_value, traceback)
        if self.enter_scope_again_count > 0:
            self.enter_scope_again_count -= 1
            return
        if self.broadcast_scope:
            self.broadcast_scope.__exit__(exception_type, exception_value, traceback)
        NpuBroadcastScopeContext._is_in_scope = False


def npu_broadcast_scope_wrapper(strategy):
    """ wrap strategy.scope with NpuBroadcastScopeContext """
    if not isinstance(strategy, tf.distribute.Strategy):
        return strategy

    org_scope = strategy.scope
    
    def _npu_broadcast_scope():
        org_scope_ctx = org_scope()
        return NpuBroadcastScopeContext(org_scope_ctx)

    if not hasattr(strategy, '_npu_scope_wrapped'):
        strategy.scope = _npu_broadcast_scope
        setattr(strategy, '_npu_scope_wrapped', True)
    return strategy