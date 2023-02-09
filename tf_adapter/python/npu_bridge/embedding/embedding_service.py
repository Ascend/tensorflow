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

import json
import contextlib
import os
import math
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.core.framework import attr_value_pb2
from npu_bridge.npu_cpu.npu_cpu_ops import gen_npu_cpu_ops
from npu_bridge.embedding.embedding_resource import NpuEmbeddingResource
from npu_bridge.embedding import embedding_optimizer


@contextlib.contextmanager
def specified_ps_engine_scope():
    """
    Enable the non npu compilation of operators within the scope.
    """
    attrs = {
        "_process_node_engine_id": attr_value_pb2.AttrValue(s=tf.compat.as_bytes("PS"))
    }
    with ops.get_default_graph()._attr_scope(attrs):
        yield


class ESWorker:
    """ Embedding service class. """

    def __init__(self, config_from_param=None):
        env_dist = os.environ
        cluster_config_from_env = env_dist.get("ESCLUSTER_CONFIG_PATH")
        if cluster_config_from_env is None:
            if config_from_param is None:
                raise ValueError("EsClusterConfig and env variable are both null.")
            es_cluster_config = config_from_param
        else:
            es_cluster_config = cluster_config_from_env
        with open(es_cluster_config, encoding='utf-8') as a:
            es_cluster_config_json = json.load(a)
            self._es_cluster_conf = json.dumps(es_cluster_config_json)
            self._ps_num = int(es_cluster_config_json["psNum"])
            self._embedding_dim = -1
            self._max_num = -1
            self._ps_ids = []
            self._ps_ids_list = es_cluster_config_json["psCluster"]
            self._init_embedding_hash_maps = {}
            self._init_partition_maps = {}
            self._table_to_embedding_dim = {}
            for each_ps in self._ps_ids_list:
                self._ps_ids.append(each_ps["id"])
        self._train_mode = True
        self._train_level = False
        self._optimizer = None
        self.slot_vars_num = None
        self._initializer = None
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["es_cluster_config"].s = tf.compat.as_bytes(self._es_cluster_conf)
        self.es_all_config = config

    # 提供embedding init功能
    # @param vocabulary_size int 类型
    # @param file_path string 类型
    # @param file_name string 类型
    # @param table_id int32 类型
    # @param max_batch_size int32 类型
    # @param optimizer 类型
    # @param initializer string 类型
    # @param embedding_dim int32 类型
    # @param only_var bool 类型
    # @param mode string 类型
    # @param partition_num int 类型
    def embedding_init(self, vocabulary_size, file_path, file_name, table_id, max_batch_size, optimizer=None,
                       initializer=None, embedding_dim=1, only_var=False, mode="bin", partition_num=65537):
        """ Operator for init embedding table. """
        self._embedding_dim = embedding_dim
        self._max_num = max_batch_size
        self._table_to_embedding_dim[table_id] = embedding_dim
        self._initializer = initializer
        bucket_size = math.ceil(vocabulary_size / self._ps_num)
        if optimizer is None:
            self._train_mode = False
            self.slot_vars_num = 0
        else:
            if (not isinstance(optimizer, embedding_optimizer.AdamOptimizer) and
                    not isinstance(optimizer, embedding_optimizer.AdagradOptimizer)):
                raise ValueError(
                    "optimizer should be embedding_optimizer.AdamOptimizer or embedding_optimizer.AdagradOptimizer")
            self._optimizer = optimizer
            self._optimizer._embedding_dims = embedding_dim
            # adam include m and v, 2 slots; adagrad include accumulator, 1 slot
            self.slot_vars_num = 2 if isinstance(self._optimizer, embedding_optimizer.AdamOptimizer) else 1
        if file_path is None or file_name is None:
            self._train_level = True
        with specified_ps_engine_scope():
            self._init_partition_maps[table_id] = \
                gen_npu_cpu_ops.init_partition_map(ps_num=ops.convert_to_tensor(self._ps_num),
                                                   ps_ids=ops.convert_to_tensor(self._ps_ids),
                                                   partition_num=partition_num)
            self._init_partition_maps.get(table_id)._set_attr("_execute_times", attr_value_pb2.AttrValue(i=1))
            self._init_partition_maps.get(table_id)._set_attr("_embedding_dim",
                                                              attr_value_pb2.AttrValue(i=self._embedding_dim))
            self._init_partition_maps.get(table_id)._set_attr("_max_num", attr_value_pb2.AttrValue(i=self._max_num))
            self._init_partition_maps.get(table_id)._set_attr("_deploy_inject_config",
                                                              attr_value_pb2.AttrValue(
                                                                  s=tf.compat.as_bytes(self._es_cluster_conf)))
            return self._init_hashmap_and_table_import(bucket_size, file_path, file_name, table_id,
                                                       initializer, embedding_dim, only_var, mode)

    # 提供embedding lookup功能
    # @param table_id int32 类型
    # @param input_ids int64 类型
    # @return values float32 类型
    def embedding_lookup(self, table_id, input_ids):
        """ Operator for look up in embedding table. """
        if self._train_mode:
            seed1, seed2 = random_seed.get_seed(None)
            result = gen_npu_cpu_ops.embedding_table_find_and_init(table_id=ops.convert_to_tensor(table_id),
                                                                   keys=input_ids,
                                                                   embedding_dim=
                                                                   self._table_to_embedding_dim.get(table_id),
                                                                   random_alg=self._initializer,
                                                                   seed=seed1, seed2=seed2,
                                                                   value_total_len=
                                                                   self._table_to_embedding_dim.get(table_id) *
                                                                   (self.slot_vars_num + 1)
                                                                   )
        else:
            result = gen_npu_cpu_ops.embedding_table_find(table_id=ops.convert_to_tensor(table_id),
                                                          keys=input_ids,
                                                          embedding_dim=self._table_to_embedding_dim.get(table_id))
        result.op._set_attr("_embedding_dim", attr_value_pb2.AttrValue(i=self._embedding_dim))
        result.op._set_attr("_max_num", attr_value_pb2.AttrValue(i=self._max_num))
        result.op._set_attr("_deploy_inject_config",
                            attr_value_pb2.AttrValue(s=tf.compat.as_bytes(self._es_cluster_conf)))
        return result

    # 提供embedding update功能
    # @param loss 类型
    # @param params float32 类型
    # @param table_ids int32 类型
    # @param input_ids_list int64 类型
    def embedding_update(self, loss, params, table_ids, input_ids_list):
        """ Operator for update in embedding table. """
        if (not isinstance(params, (list, tuple)) and not isinstance(table_ids, (list, tuple))
                and not isinstance(input_ids_list, (list, tuple))):
            params = [params]
            table_ids = [table_ids]
            input_ids_list = [input_ids_list]
        if (len(params) != len(table_ids)) or (len(params) != len(input_ids_list)) \
                or (len(table_ids) != len(input_ids_list)):
            raise ValueError("The length of params, table_ids, input_ids_list should be equal.")
        embedding_grads = tf.gradients(loss, params)
        params_grads = []
        for i in range(len(embedding_grads)):
            params_grads.append(tf.IndexedSlices(embedding_grads[i], input_ids_list[i], dense_shape=params[i].shape))
        with specified_ps_engine_scope():
            var_refs = [NpuEmbeddingResource(table_id) for table_id in table_ids]
            update_op = self._optimizer.apply_gradients(list(zip(params_grads, var_refs)))
            return update_op

    # 提供训练好的embedding values save功能
    # @param file_path string 类型
    # @param file_name string 类型
    # @param table_id int32 类型
    # @param mode string 类型
    def embedding_save(self, file_path, file_name, table_id, mode="bin"):
        """ Operator for save values in embedding table. """
        with specified_ps_engine_scope():
            embedding_dim = self._table_to_embedding_dim.get(table_id)
            return gen_npu_cpu_ops.embedding_table_export(file_path, file_name, ops.convert_to_tensor(-1), table_id,
                                                          embedding_dim, embedding_dim, True, mode)

    # 提供训练好的embedding values + 调优参数 save功能
    # @param file_path string 类型
    # @param file_name string 类型
    # @param table_id int32 类型
    # @param mode string 类型
    def embedding_ckpt_save(self, file_path, file_name, table_id, mode="bin"):
        """ Operator for save values and optimizer params in embedding table. """
        with specified_ps_engine_scope():
            embedding_dim = self._table_to_embedding_dim.get(table_id)
            return gen_npu_cpu_ops.embedding_table_export(file_path, file_name, ops.convert_to_tensor(-1), table_id,
                                                          embedding_dim, embedding_dim * (self.slot_vars_num + 1),
                                                          False, mode)

    def _init_hashmap_and_table_import(self, bucket_size, file_path, file_name, table_id,
                                       initializer, embedding_dim, only_var, mode):
        with tf.control_dependencies([self._init_partition_maps.get(table_id)]):
            if self._train_mode:
                if self._train_level:
                    seed1, seed2 = random_seed.get_seed(None)
                    self._init_embedding_hash_maps[table_id] = \
                        gen_npu_cpu_ops.init_embedding_hashmap(table_id=ops.convert_to_tensor(table_id),
                                                               bucket_size=bucket_size,
                                                               value_total_len=embedding_dim * (self.slot_vars_num + 1),
                                                               embedding_dim=embedding_dim,
                                                               random_alg=initializer, seed=seed1, seed2=seed2)
                else:
                    self._init_embedding_hash_maps[table_id] = \
                        gen_npu_cpu_ops.init_embedding_hashmap(table_id=ops.convert_to_tensor(table_id),
                                                               bucket_size=bucket_size,
                                                               value_total_len=embedding_dim * (self.slot_vars_num + 1),
                                                               embedding_dim=embedding_dim,
                                                               random_alg=None, seed=None, seed2=None)
            else:
                self._init_embedding_hash_maps[table_id] = \
                    gen_npu_cpu_ops.init_embedding_hashmap(table_id=ops.convert_to_tensor(table_id),
                                                           bucket_size=bucket_size,
                                                           value_total_len=embedding_dim,
                                                           embedding_dim=embedding_dim,
                                                           random_alg=None, seed=None, seed2=None)
        return self._init_or_restore(file_path, file_name, table_id, embedding_dim, only_var, mode)

    def _init_or_restore(self, file_path, file_name, table_id, embedding_dim, only_var, mode):
        if self._train_mode and self._train_level:
            return tf.group(
                [tf.initializers.variables(self._optimizer.variables()), self._init_embedding_hash_maps.get(table_id)])
        # restore embedding table
        with tf.control_dependencies([self._init_embedding_hash_maps.get(table_id)]):
            embedding_table_import = gen_npu_cpu_ops.embedding_table_import(
                file_path=ops.convert_to_tensor(file_path),
                file_name=ops.convert_to_tensor(file_name),
                # ps_id will be changed in executor, so can not be optimized in graph
                ps_id=ops.convert_to_tensor(-1),
                table_id=ops.convert_to_tensor(table_id),
                embedding_dim=embedding_dim,
                value_total_len=embedding_dim * (self.slot_vars_num + 1),
                only_var_flag=only_var,
                file_type=mode)
        return tf.group([embedding_table_import])
