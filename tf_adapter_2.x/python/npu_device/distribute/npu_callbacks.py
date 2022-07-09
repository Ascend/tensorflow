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
from tensorflow.python import keras
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf
from npu_device.distribute import hccl

broadcast_registry = set()


def broadcast_keras_model(model, root_rank=0):
    """Broadcast trainable variables of keras Model"""
    if not isinstance(model, tf.keras.Model):
        return model
    candicates = []
    for var in model.trainable_variables:
        if id(var) not in broadcast_registry:
            candicates.append(var)
    if candicates:
        hccl.broadcast(candicates)
        broadcast_registry.update([id(value) for value in candicates])
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
                broadcast_keras_model(self.model, self.root_rank)

            self.broadcast_done = True
