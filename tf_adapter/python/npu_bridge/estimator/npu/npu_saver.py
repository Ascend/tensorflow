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

"""Definition for NPU saver"""

import tensorflow as tf
from tensorflow.python.training.saver import BulkSaverBuilder
from tensorflow.python.training.saver import Saver
from tensorflow.python.eager import context
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from npu_bridge.estimator.npu import util
from npu_bridge.hccl import hccl_ops


class NPUBulkSaverBuilder(BulkSaverBuilder):
    """Class to build NPU builker saver"""
    def _build_internal(self,
                        names_to_saveables,
                        reshape=False,
                        sharded=False,
                        max_to_keep=5,
                        keep_checkpoint_every_n_hours=10000.0,
                        op_name=None,
                        restore_sequentially=False,
                        filename="model",
                        build_save=True,
                        build_restore=True):
        """build() with option to only perform save and restore."""
        if not context.executing_eagerly() and (not build_save or
                                                not build_restore):
            raise ValueError("save and restore operations need to be built together "
                             " when eager execution is not enabled.")

        saveables = saveable_object_util.validate_and_slice_inputs(
            names_to_saveables)
        if max_to_keep is None:
            max_to_keep = 0

        with ops.name_scope(op_name, "save",
                            [saveable.op for saveable in saveables]) as name:
            # Add a placeholder string tensor for the filename.
            filename_tensor = array_ops.placeholder_with_default(
                filename or "model", shape=(), name="filename")
            # Keep the name "Const" for backwards compatibility.
            filename_tensor = array_ops.placeholder_with_default(
                filename_tensor, shape=(), name="Const")

            # Add the save ops.
            if sharded:
                per_device = self._GroupByDevices(saveables)
                if build_save:
                    op_list = []
                    with tf.name_scope("Save_Weight_Update_Sharding"):
                        grad_and_var_items = util.get_all_grad_item()
                        for item in grad_and_var_items:
                            if item.var in names_to_saveables:
                                rank_id = item.root_rank_id
                                if rank_id >= 0:
                                    with tf.get_default_graph().control_dependencies(op_list):
                                        out_var = hccl_ops.broadcast([item.var], rank_id, 2, rank_id)
                                    op_list.append(out_var[0].op)
                    if len(op_list) > 0:
                        with tf.get_default_graph().control_dependencies(op_list):
                            save_tensor = self._AddShardedSaveOps(filename_tensor, per_device)
                    else:
                        save_tensor = self._AddShardedSaveOps(filename_tensor, per_device)
                if build_restore:
                    restore_op = self._AddShardedRestoreOps(filename_tensor, per_device,
                                                            restore_sequentially, reshape)
            else:
                if build_save:
                    op_list = []
                    with tf.name_scope("Save_Weight_Update_Sharding"):
                        grad_and_var_items = util.get_all_grad_item()
                        for item in grad_and_var_items:
                            if item.var in names_to_saveables:
                                rank_id = item.root_rank_id
                                if rank_id >= 0:
                                    with tf.get_default_graph().control_dependencies(op_list):
                                        out_var = hccl_ops.broadcast([item.var], rank_id, 2, rank_id)
                                    op_list.append(out_var[0].op)
                    if len(op_list) > 0:
                        with tf.get_default_graph().control_dependencies(op_list):
                            save_tensor = self._AddSaveOps(filename_tensor, saveables)
                    else:
                        save_tensor = self._AddSaveOps(filename_tensor, saveables)
                if build_restore:
                    restore_op = self._AddRestoreOps(filename_tensor, saveables,
                                                     restore_sequentially, reshape)

        # In the following use case, it's possible to have restore_ops be called
        # something else:
        # - Build inference graph and export a meta_graph.
        # - Import the inference meta_graph
        # - Extend the inference graph to a train graph.
        # - Export a new meta_graph.
        # Now the second restore_op will be called "restore_all_1".
        # As such, comment out the assert for now until we know whether supporting
        # such usage model makes sense.
        #
        # assert restore_op.name.endswith("restore_all"), restore_op.name
        if context.executing_eagerly():
            # Store the tensor values to the tensor_names.
            save_tensor_name = save_tensor.numpy() if build_save else ""
            return saver_pb2.SaverDef(
                filename_tensor_name=filename_tensor.numpy(),
                save_tensor_name=save_tensor_name,
                restore_op_name="",
                max_to_keep=max_to_keep,
                sharded=sharded,
                keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
                version=self._write_version)
        else:
            graph = ops.get_default_graph()
            # Do some sanity checking on collections containing
            # PartitionedVariables. If a saved collection has a PartitionedVariable,
            # the GraphDef needs to include concat ops to get the value (or there'll
            # be a lookup error on load).
            check_collection_list = graph.get_all_collection_keys()
            for collection_type in check_collection_list:
                for element in graph.get_collection(collection_type):
                    if isinstance(element, variables.PartitionedVariable):
                        try:
                            graph.get_operation_by_name(element.name)
                        except KeyError:
                            # Create a concat op for this PartitionedVariable. The user may
                            # not need it, but we'll try looking it up on MetaGraph restore
                            # since it's in a collection.
                            element.as_tensor()
            return saver_pb2.SaverDef(
                filename_tensor_name=filename_tensor.name,
                save_tensor_name=save_tensor.name,
                restore_op_name=restore_op.name,
                max_to_keep=max_to_keep,
                sharded=sharded,
                keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
                version=self._write_version)


class NPUSaver(Saver):
    """NPU saver for saving checkpoints"""
    def __init__(self):
        self._builder = None

    def _build(self, checkpoint_path, build_save, build_restore):
        if not self.saver_def or context.executing_eagerly():
            if self._builder is None:
                self._builder = NPUBulkSaverBuilder(self._write_version)
        super()._build(checkpoint_path=checkpoint_path, build_save=build_save, build_restore=build_restore)
