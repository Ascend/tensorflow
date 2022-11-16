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
from tensorflow.python.keras import backend
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from npu_bridge.hccl import hccl_ops
from npu_bridge.estimator.npu import util as util_lib


def broadcast_global_variables(root_rank):
    """Used to broadcast global variables"""
    variables = backend._get_variables(backend.get_graph())
    candidate_vars = []
    for v in variables:
        if getattr(v, "_keras_initialized", False):
            candidate_vars.append(v)
    op_list = []
    if candidate_vars:
        for var in candidate_vars:
            inputs = [var]
            outputs = hccl_ops.broadcast(tensor=inputs, root_rank=root_rank)
            if outputs is not None:
                op_list.append(outputs[0].op)
                op_list.append(state_ops.assign(var, outputs[0]))
    return control_flow_ops.group(op_list)


class BroadcastGlobalVariablesCallbackImpl:
    """NPU implemented global variable broadcast callback function"""
    def __init__(self, root_rank, *args):
        super(BroadcastGlobalVariablesCallbackImpl, self).__init__(*args)
        self.root_rank = root_rank
        self.broadcast_done = False

    def on_batch_begin(self, batch, logs=None):
        """This function is called when every batch begins"""
        if self.broadcast_done:
            return

        rank_size = util_lib.get_ranksize()
        if int(rank_size) > 1:
            bcast_op = broadcast_global_variables(self.root_rank)
            backend.get_session().run(bcast_op)

        self.broadcast_done = True


class NPUBroadcastGlobalVariablesCallback(BroadcastGlobalVariablesCallbackImpl, keras.callbacks.Callback):
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
        super(NPUBroadcastGlobalVariablesCallback, self).__init__(root_rank)
