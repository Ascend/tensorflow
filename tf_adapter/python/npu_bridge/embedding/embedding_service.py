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
from npu_bridge.embedding.embedding_table_map_policy import NoneTableMapPolicy, AutoMergeTableMapPolicy

_INT32_MAX_VALUE = 2147483647


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
            self._table_to_max_num = {}
            self._table_to_optimizer = {}
            self._table_to_initializer = {}
            self._table_to_slot_var_num = {}
            for each_ps in self._ps_ids_list:
                self._ps_ids.append(each_ps["id"])
        self._train_mode = True
        self._train_level = False
        self._optimizer = None
        self.slot_vars_num = None
        self._initializer = None
        self._init_flag = False
        self._table_has_init = []
        self.user_defined_table_infos = []
        self.table_map_policy = None
        self.table_create_infos = []
        self.total_variable_table = []
        self.total_embedding_count = 0
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
                       initializer=None, embedding_dim=-1, only_var=False, mode="bin", partition_num=65537):
        """ Operator for init embedding table. """
        if vocabulary_size is None or table_id is None or max_batch_size is None or embedding_dim is None:
            raise ValueError("vocabulary_size or table_id or max_batch_size or embedding_dim is None.")
        if (not isinstance(vocabulary_size, int)) or (not isinstance(table_id, int)) or \
                (not isinstance(max_batch_size, int)) or (not isinstance(embedding_dim, int)):
            raise ValueError("vocabulary_size, table_id, max_batch_size and embedding_dim must be int.")
        if vocabulary_size < 0 or table_id < 0:
            raise ValueError("vocabulary_size and table_id can not be smaller than zero.")
        if vocabulary_size >= _INT32_MAX_VALUE or table_id >= _INT32_MAX_VALUE:
            raise ValueError("vocabulary_size or table_id exceed int32 max value.")
        if embedding_dim <= 0 or partition_num <= 0 or max_batch_size <= 0:
            raise ValueError("embedding_dim, partition_num and max_batch_size must be greater than zero.")
        if table_id in self._table_has_init:
            raise ValueError("this table has already initialized.")
        self._embedding_dim = embedding_dim
        self._max_num = max_batch_size
        self._table_to_embedding_dim[table_id] = embedding_dim
        self._table_to_max_num[table_id] = max_batch_size
        self._initializer = initializer
        self._table_to_initializer[table_id] = initializer
        self._table_has_init.append(table_id)
        bucket_size = math.ceil(vocabulary_size / self._ps_num)
        if optimizer is None:
            if file_path is None or file_name is None or (not tf.gfile.Exists(os.path.join(file_path, file_name))):
                raise ValueError("embedding table file not exist.")
            self._train_mode = False
            self.slot_vars_num = 0
        else:
            if (not isinstance(optimizer, embedding_optimizer.AdamOptimizer) and
                    not isinstance(optimizer, embedding_optimizer.AdagradOptimizer)):
                raise ValueError(
                    "optimizer should be embedding_optimizer.AdamOptimizer or embedding_optimizer.AdagradOptimizer")
            if (initializer is not None) and (initializer is not 'random_uniform') and \
                    (initializer is not 'truncated_normal'):
                raise ValueError("initializer must be random_uniform or truncated_normal.")
            self._optimizer = optimizer
            self._optimizer._embedding_dims = embedding_dim
            self._optimizer._max_nums = max_batch_size
            self._table_to_optimizer[table_id] = self._optimizer
            # adam include m and v, 2 slots; adagrad include accumulator, 1 slot
            self.slot_vars_num = 2 if isinstance(self._optimizer, embedding_optimizer.AdamOptimizer) else 1
        self._table_to_slot_var_num[table_id] = self.slot_vars_num
        if (file_path is None) or (file_name is None) or (not tf.gfile.Exists(os.path.join(file_path, file_name))):
            if initializer is None:
                raise ValueError("In new embedding training, initializer can not be None.")
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
        if (table_id is None) or (input_ids is None):
            raise ValueError("table_id or input_ids must be specified.")
        if not isinstance(table_id, int):
            raise ValueError("type of table_id must be int.")
        if input_ids.dtype != tf.int64:
            raise ValueError("dtype of input_ids must be tf.int64.")
        if table_id < 0:
            raise ValueError("table_id can not be smaller than zero.")
        if not self._init_flag:
            raise ValueError("embedding must init first!")
        if table_id not in self._table_has_init:
            raise ValueError("this table has not yet initialized.")
        if self._train_mode:
            seed1, seed2 = random_seed.get_seed(None)
            result = gen_npu_cpu_ops.embedding_table_find_and_init(table_id=ops.convert_to_tensor(table_id),
                                                                   keys=input_ids,
                                                                   embedding_dim=
                                                                   self._table_to_embedding_dim.get(table_id),
                                                                   random_alg=self._table_to_initializer.get(table_id),
                                                                   seed=seed1, seed2=seed2,
                                                                   value_total_len=
                                                                   self._table_to_embedding_dim.get(table_id) *
                                                                   (self._table_to_slot_var_num.get(table_id) + 1)
                                                                   )
        else:
            result = gen_npu_cpu_ops.embedding_table_find(table_id=ops.convert_to_tensor(table_id),
                                                          keys=input_ids,
                                                          embedding_dim=self._table_to_embedding_dim.get(table_id))
        result.op._set_attr("_embedding_dim", attr_value_pb2.AttrValue(i=self._table_to_embedding_dim.get(table_id)))
        result.op._set_attr("_max_num", attr_value_pb2.AttrValue(i=self._table_to_max_num.get(table_id)))
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
        if (loss is None) or (params is None) or (table_ids is None) or (input_ids_list is None):
            raise ValueError("loss or params or table_ids or input_ids_list is None.")
        if (isinstance(loss, str)) or (isinstance(params, str)) or isinstance(table_ids, str) or \
                isinstance(input_ids_list, str):
            raise ValueError("loss, params, table_ids and input_ids_list can not be str.")
        if not self._init_flag:
            raise ValueError("embedding must init first!")
        if (not isinstance(params, (list, tuple)) and not isinstance(table_ids, (list, tuple))
                and not isinstance(input_ids_list, (list, tuple))):
            params = [params]
            table_ids = [table_ids]
            input_ids_list = [input_ids_list]
        for table_id in table_ids:
            if table_id not in self._table_has_init:
                raise ValueError("this table has not yet initialized.")
        if (len(params) != len(table_ids)) or (len(params) != len(input_ids_list)) \
                or (len(table_ids) != len(input_ids_list)):
            raise ValueError("The length of params, table_ids, input_ids_list should be equal.")
        embedding_grads = tf.gradients(loss, params)
        update_op = []
        with specified_ps_engine_scope():
            for i in range(len(table_ids)):
                params_grads = [tf.IndexedSlices(embedding_grads[i], input_ids_list[i], dense_shape=params[i].shape)]
                var_refs = [NpuEmbeddingResource(table_ids[i])]
                update_op.append(
                    self._table_to_optimizer.get(table_ids[i]).apply_gradients(list(zip(params_grads, var_refs))))
            return update_op

    # 提供训练好的embedding values save功能
    # @param file_path string 类型
    # @param file_name string 类型
    # @param table_id int32 类型
    # @param mode string 类型
    def embedding_save(self, file_path, file_name, table_id, mode="bin"):
        """ Operator for save values in embedding table. """
        if file_path is None or file_name is None or table_id is None:
            raise ValueError("table_id, embedding table file_name and file_path can not be None.")
        if table_id not in self._table_has_init:
            raise ValueError("this table has not yet initialized.")
        if not os.path.exists(file_path):
            os.mkdir(file_path)
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
        if file_path is None or file_name is None or table_id is None:
            raise ValueError("table_id, embedding table file_name and file_path can not be None.")
        if table_id not in self._table_has_init:
            raise ValueError("this table has not yet initialized.")
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        with specified_ps_engine_scope():
            embedding_dim = self._table_to_embedding_dim.get(table_id)
            return gen_npu_cpu_ops.embedding_table_export(file_path, file_name, ops.convert_to_tensor(-1), table_id,
                                                          embedding_dim, embedding_dim *
                                                          (self._table_to_slot_var_num.get(table_id) + 1),
                                                          False, mode)

    def data_parallel_embedding(self, max_vocabulary_size, embedding_dim, multihot_lens, allow_merge=True):
        if (max_vocabulary_size is None) or (embedding_dim is None) or (multihot_lens is None):
            raise ValueError("max_vocabulary_size or embedding_dim or multihot_lens can not be None.")
        if (not isinstance(max_vocabulary_size, int)) or (not isinstance(embedding_dim, int)) or \
                (not isinstance(multihot_lens, int)) or (not isinstance(allow_merge, bool)):
            raise TypeError("max_vocabulary_size, embedding_dim, multihot_lens must be int, allow_merge must be bool.")
        if max_vocabulary_size <= 0 or embedding_dim <= 0 or multihot_lens <= 0:
            raise ValueError("max_vocabulary_size, embedding_dim, multihot_lens must be greater than zero.")
        new_table_info = dict(
            max_vocabulary_size=max_vocabulary_size,
            embedding_dim=embedding_dim,
            multihot_lens=multihot_lens,
            allow_merge=allow_merge
        )
        self.user_defined_table_infos.append(new_table_info)

    def init_table(self, table_map_policy=AutoMergeTableMapPolicy()):
        if (not isinstance(table_map_policy, NoneTableMapPolicy)) and\
                (not isinstance(table_map_policy, AutoMergeTableMapPolicy)):
            raise TypeError("table_map_policy should be NoneTableMapPolicy or AutoMergeTableMapPolicy.")
        if len(self.user_defined_table_infos) == 0:
            raise ValueError("small table has not been created.")
        self.table_map_policy = table_map_policy
        self.table_create_infos = self.table_map_policy.map_table_infos(self.user_defined_table_infos)
        for table_info_ in self.table_create_infos:
            self.total_variable_table.append(tf.Variable(
                tf.random_normal([table_info_['max_vocabulary_size'], table_info_['embedding_dim']], mean=0.0,
                                 stddev=1.0, dtype=tf.float32, seed=1234)
            ))
            self.total_embedding_count += 1

    def embeddings_look_up(self, tf_indices):
        if self.total_embedding_count != len(self.table_create_infos) or self.total_embedding_count == 0:
            raise ValueError("Must init_table() first!")
        if tf_indices is None:
            raise ValueError("tf_indices can not be None.")
        if tf_indices.dtype != tf.int64:
            raise TypeError("dtype of tf_indices must be tf.int64.")
        (in_slot_size_group, slot_to_table, table_to_input_group, \
         table_to_slot, table_to_output_slots) = \
            (self.table_map_policy.in_slot_size_group, self.table_map_policy.slot_to_table, \
             self.table_map_policy.table_to_input_groups, self.table_map_policy.table_to_slot, \
             self.table_map_policy.table_to_output_slots)

        tf_indices_shape_list = tf_indices.get_shape().as_list()
        total_in_slot_num = 0
        for in_slot_size in in_slot_size_group:
            total_in_slot_num += in_slot_size
        if tf_indices_shape_list[1] != total_in_slot_num:
            raise ValueError("size of tf_indices is not the same as all small tables.")

        indices_split = tf.split(tf_indices, in_slot_size_group, axis=1)
        for tid in range(self.total_embedding_count):
            table_to_input_group[tid] = []
        for sid, indices in enumerate(indices_split):
            tid = slot_to_table[sid]
            table_to_input_group[tid].append(indices)

        output_slots = [None for _ in in_slot_size_group]
        for tid, table_input_group in enumerate(table_to_input_group):
            table_input_hash = tf.concat(table_input_group, axis=1)
            table_input_hash_shape = table_input_hash.get_shape().as_list()
            input_hash_after_unique, recovery_matrix =\
                tf.unique(x=tf.reshape(table_input_hash, shape=[table_input_hash_shape[0] * table_input_hash_shape[1]]),
                          out_idx=tf.int64)
            table_input_after_mapping = \
                gen_npu_cpu_ops.embedding_feature_mapping(feature_id=input_hash_after_unique)
            table_to_input_group[tid] = table_input_after_mapping
            table_embedding = tf.nn.embedding_lookup(self.total_variable_table[tid], table_input_after_mapping)
            table_embedding_after_gather = tf.reshape(tf.gather(params=table_embedding, indices=recovery_matrix),
                                                      shape=[table_input_hash_shape[0], table_input_hash_shape[1], -1])
            out_embedding_splited = tf.split(table_embedding_after_gather, table_to_output_slots[tid], axis=1)
            for out_emb, sid in zip(out_embedding_splited, table_to_slot[tid]):
                output_slots[sid] = out_emb
        return output_slots

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
        self._init_flag = True
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
