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


class EmbeddingVariableOption:
    """ option for embedding service table. """

    def __init__(self, filter_option=None,
                 evict_option=None,
                 storage_option=None,
                 feature_freezing_option=None,
                 communication_option=None):
        self.filter_option = filter_option
        self.evict_option = evict_option
        self.storage_option = storage_option
        self.feature_freezing_option = feature_freezing_option
        self.communication_option = communication_option


class CounterFilter:
    """ Counter filter for embedding table. """

    def __init__(self, filter_freq, default_key_or_value, default_key=None, default_value=None):
        self.filter_freq = filter_freq
        self.default_key = default_key
        self.default_value = default_value
        self.default_key_or_value = default_key_or_value


class Initializer:
    """Initializer for embedding service table."""

    def __init__(self, min, max, initializer_mode, constant_value, mu=0.0, sigma=1.0):
        self.min = min
        self.max = max
        self.initializer_mode = initializer_mode
        self.constant_value = constant_value
        self.mu = mu
        self.sigma = sigma


class ESWorker:
    """ Embedding service class. """

    def __init__(self):
        env_dist = os.environ
        cluster_config_from_env = env_dist.get("ESCLUSTER_CONFIG_PATH")
        if cluster_config_from_env is None:
            raise ValueError("EsClusterConfig env are both null.")
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
            self._table_to_counter_filter = {}
            for each_ps in self._ps_ids_list:
                self._ps_ids.append(each_ps["id"])
        self._train_mode = True
        self._train_level = False
        self._optimizer = None
        self.slot_vars_num = None
        self._initializer = None
        self._init_flag = False
        self._table_init = False
        self._table_has_init = []
        self.user_defined_table_infos = []
        self.table_map_policy = None
        self.table_create_infos = []
        self.total_variable_table = []
        self.total_embedding_count = 0
        self._npu_table_to_embedding_dim = {}
        self._use_counter_filter = False
        self._default_key_or_value = True
        self._filter_freq = None
        self._default_key = None
        self._default_value = None

    # 提供 embedding_service table initializer method
    # table_id embedding 表索引, int 类型
    # min 下限值, float 类型
    # max 上限值, float 类型
    # initializer_mode 初始化方式, string 类型
    # constant_value 常量初始化的常量值, float 类型
    # mu 正态分布的均值, float 类型
    # sigma 正态分布的标准差, float 类型
    def initializer(self, table_id, initializer_mode, min=-2.0, max=2.0, constant_value=0.0, mu=0.0, sigma=1.0):
        """Operator for init initializer."""
        if (table_id is None) or (initializer_mode is None):
            raise ValueError("table_id and initializer_mode can not be None.")
        if initializer_mode == 'random_uniform':
            if (min is None) or (max is None) or \
                    (not isinstance(min, (float, int))) or (not isinstance(max, (float, int))):
                raise ValueError("If initializer is random_uniform, min and max can not be None, must be int or float.")
        if initializer_mode == 'truncated_normal':
            if (min is None) or (max is None) or (mu is None) or (sigma is None) or \
                    (not isinstance(min, (float, int))) or (not isinstance(max, (float, int))) or \
                    (not isinstance(mu, (float, int))) or (not isinstance(sigma, (float, int))):
                raise ValueError("If initializer is truncated_normal, min, max, mu and sigma can not be None,"
                                 "and they must be int or float.")
        if initializer_mode == 'constant':
            if (constant_value is None) or (not isinstance(constant_value, (float, int))):
                raise ValueError("If initializer is constant, constant_value can not be None, must be float or int.")
        if (not isinstance(table_id, int)) or (table_id < 0) or (table_id >= _INT32_MAX_VALUE):
            raise ValueError("table_id value is false, must be [0, 2147483647) and int type, please check.")
        if min > max:
            raise ValueError("Initializer min value can not be larger than max value.")
        if (initializer_mode != 'constant') and (initializer_mode != 'random_uniform') and \
                (initializer_mode != 'truncated_normal'):
            raise ValueError("Initializer mode must be random_uniform or truncated normal or constant.")
        self._table_to_initializer[table_id] = Initializer(min=min,
                                                           max=max,
                                                           initializer_mode=initializer_mode,
                                                           constant_value=constant_value,
                                                           mu=mu,
                                                           sigma=sigma)

    # embedding variable option
    # 包括特征准入及淘汰策略，特征存储策略及通信策略等
    # 暂时只使用特征准入option
    def embedding_variable_option(self, filter_option=None, evict_option=None, storage_option=None,
                                  feature_freezing_option=None, communication_option=None):
        if filter_option is None:
            raise ValueError("Now filter_option can't be None.")
        if not isinstance(filter_option, CounterFilter):
            raise TypeError("If filter_option isn't None, it must be CounterFilter type.")
        self._use_counter_filter = True
        return EmbeddingVariableOption(filter_option=filter_option, evict_option=evict_option,
                                       storage_option=storage_option, feature_freezing_option=feature_freezing_option,
                                       communication_option=communication_option)

    # 提供embedding init功能
    # @param vocabulary_size 表的初始大小, int 类型
    # @param table_id, int32 类型
    # @param max_batch_size, int32 类型
    # @param optimizer, 支持EmbeddingAdamOptimizer，EmbeddingAdagradOptimizer，EmbeddingAdamwOptimizer
    # @param initializer, string 类型
    # @param embedding_dim, int32 类型
    def embedding_init(self, vocabulary_size, table_id, max_batch_size, embedding_dim, optimizer=None,
                       initializer=None, ev_option=None):
        """ Operator for init embedding table. """
        if vocabulary_size is None or table_id is None or max_batch_size is None or embedding_dim is None:
            raise ValueError("vocabulary_size or table_id or max_batch_size or embedding_dim is None.")
        if (ev_option is not None) and (not isinstance(ev_option, EmbeddingVariableOption)):
            raise TypeError("ev_option must be EmbeddingVariableOption type.")
        if (not isinstance(vocabulary_size, int)) or (not isinstance(table_id, int)) or \
                (not isinstance(max_batch_size, int)) or (not isinstance(embedding_dim, int)):
            raise ValueError("vocabulary_size, table_id, max_batch_size and embedding_dim must be int.")
        if vocabulary_size < 0 or table_id < 0:
            raise ValueError("vocabulary_size and table_id can not be smaller than zero.")
        if vocabulary_size >= _INT32_MAX_VALUE or table_id >= _INT32_MAX_VALUE:
            raise ValueError("vocabulary_size or table_id exceed int32 max value.")
        if embedding_dim <= 0 or max_batch_size <= 0:
            raise ValueError("embedding_dim and max_batch_size must be greater than zero.")
        if table_id in self._table_has_init:
            raise ValueError("this table has already initialized.")
        self._embedding_dim = embedding_dim
        self._max_num = max_batch_size
        self._table_to_embedding_dim[table_id] = embedding_dim
        self._table_to_max_num[table_id] = max_batch_size
        self._table_has_init.append(table_id)
        if len(self._table_has_init) > 10:
            raise ValueError("Now only 10 embedding tables can be init.")
        bucket_size = math.ceil(vocabulary_size / self._ps_num)
        if (self._table_to_initializer.get(table_id) is None) and (initializer is not None):
            self._table_to_initializer[table_id] = Initializer(min=-2,
                                                               max=2,
                                                               initializer_mode=initializer,
                                                               constant_value=0,
                                                               mu=0.0,
                                                               sigma=1.0)
        if optimizer is None:
            self._train_mode = False
            self.slot_vars_num = 0
        else:
            if (not isinstance(optimizer, embedding_optimizer.AdamOptimizer)) and \
                    (not isinstance(optimizer, embedding_optimizer.AdagradOptimizer)) and \
                    (not isinstance(optimizer, embedding_optimizer.AdamWOptimizer)):
                raise ValueError(
                    "optimizer should be embedding_optimizer AdamOptimizer, AdagradOptimizer or AdamWOptimizer.")
            if (initializer is not None) and (initializer != 'random_uniform') and \
                    (initializer != 'truncated_normal') and (initializer != 'constant'):
                raise ValueError("initializer must be random_uniform or truncated_normal or constant.")
            self._optimizer = optimizer
            self._optimizer._embedding_dims = embedding_dim
            self._optimizer._max_nums = max_batch_size
            self._optimizer._es_cluster_configs = self._es_cluster_conf
            self._table_to_optimizer[table_id] = self._optimizer
            # adam include m and v, 2 slots; adagrad include accumulator, 1 slot
            self.slot_vars_num = 1 if isinstance(self._optimizer, embedding_optimizer.AdagradOptimizer) else 2
            if (initializer is not None) or (self._table_to_initializer.get(table_id) is not None):
                self._train_level = True
        self._table_to_slot_var_num[table_id] = self.slot_vars_num
        with specified_ps_engine_scope():
            self._init_partition_maps[table_id] = \
                gen_npu_cpu_ops.init_partition_map(ps_num=ops.convert_to_tensor(self._ps_num),
                                                   ps_ids=ops.convert_to_tensor(self._ps_ids),
                                                   partition_num=65537)
            self._init_partition_maps.get(table_id)._set_attr("_embedding_dim",
                                                              attr_value_pb2.AttrValue(i=self._embedding_dim))
            self._init_partition_maps.get(table_id)._set_attr("_max_key_num", attr_value_pb2.AttrValue(i=self._max_num))
            return self._init_hashmap_and_table_import(bucket_size, table_id, embedding_dim, ev_option)

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
            if self._table_to_counter_filter.get(table_id) is not None:
                filter_mode = "counter"
                self._filter_freq = self._table_to_counter_filter.get(table_id).filter_freq
                self._default_key_or_value = self._table_to_counter_filter.get(table_id).default_key_or_value
                self._default_key = self._table_to_counter_filter.get(table_id).default_key
                self._default_value = self._table_to_counter_filter.get(table_id).default_value
            else:
                filter_mode = "no_filter"
            result = gen_npu_cpu_ops. \
                embedding_table_find_and_init(table_id=ops.convert_to_tensor(table_id),
                                              keys=input_ids,
                                              embedding_dim=self._table_to_embedding_dim.get(table_id),
                                              initializer_mode=self._table_to_initializer.get(table_id)
                                              .initializer_mode,
                                              constant_value=self._table_to_initializer.get(table_id).constant_value,
                                              min=self._table_to_initializer.get(table_id).min,
                                              max=self._table_to_initializer.get(table_id).max,
                                              mu=self._table_to_initializer.get(table_id).mu,
                                              sigma=self._table_to_initializer.get(table_id).sigma,
                                              seed=seed1,
                                              seed2=seed2,
                                              value_total_len=self._table_to_embedding_dim.get(table_id) *
                                                              (self._table_to_slot_var_num.get(table_id) + 1),
                                              filter_mode=filter_mode,
                                              filter_freq=self._filter_freq,
                                              default_key_or_value=self._default_key_or_value,
                                              default_key=self._default_key,
                                              default_value=self._default_value
                                              )
            self._filter_freq = None
            self._default_key_or_value = True
            self._default_key = None
            self._default_value = None
        else:
            result = gen_npu_cpu_ops.embedding_table_find(table_id=ops.convert_to_tensor(table_id),
                                                          keys=input_ids,
                                                          embedding_dim=self._table_to_embedding_dim.get(table_id))
        result.op._set_attr("_embedding_dim", attr_value_pb2.AttrValue(i=self._table_to_embedding_dim.get(table_id)))
        result.op._set_attr("_max_key_num", attr_value_pb2.AttrValue(i=self._table_to_max_num.get(table_id)))
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

    def counter_filter(self, filter_freq, default_key=None, default_value=None):
        if not isinstance(filter_freq, int):
            raise TypeError("filter_freq must be int, please check.")
        if (default_key is None) and (default_value is None):
            raise ValueError("default_key and default_value can not be both None.")
        if (default_key is not None) and (default_value is not None):
            raise ValueError("default_key and default_value can not be both set.")
        if default_key is None and (not isinstance(default_value, (int, float))):
            raise TypeError("When default_value is not None, it must be float or int, please check.")
        if default_value is None and (not isinstance(default_key, int)):
            raise TypeError("When default_key is not None, it must be int, please check.")
        self._use_counter_filter = True
        if default_key is None:
            return CounterFilter(filter_freq=filter_freq, default_key_or_value=False,
                                 default_key=default_key, default_value=default_value)
        else:
            return CounterFilter(filter_freq=filter_freq, default_key_or_value=True,
                                 default_key=default_key, default_value=default_value)

    def data_parallel_embedding(self, max_vocabulary_size, embedding_dim, multihot_lens, allow_merge=True,
                                initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01, seed=1234)):
        if (max_vocabulary_size is None) or (embedding_dim is None) or (multihot_lens is None):
            raise ValueError("max_vocabulary_size or embedding_dim or multihot_lens can not be None.")
        if (not isinstance(max_vocabulary_size, int)) or (not isinstance(embedding_dim, int)) or \
                (not isinstance(multihot_lens, int)) or (not isinstance(allow_merge, bool)):
            raise TypeError("max_vocabulary_size, embedding_dim, multihot_lens must be int, allow_merge must be bool.")
        if max_vocabulary_size <= 0 or embedding_dim <= 0 or multihot_lens <= 0:
            raise ValueError("max_vocabulary_size, embedding_dim, multihot_lens must be greater than zero.")
        if initializer is None:
            raise ValueError("Initializer can not be None.")
        if initializer is not None and not callable(initializer):
            init_dtype = ops.convert_to_tensor(initializer).dtype.base_dtype
            if init_dtype != tf.float32:
                raise ValueError("Initializer type '%s' and explict dtype tf.float32 don't match." % init_dtype)
        new_table_info = dict(
            max_vocabulary_size=max_vocabulary_size,
            embedding_dim=embedding_dim,
            multihot_lens=multihot_lens,
            allow_merge=allow_merge,
            initializer=initializer
        )
        self.user_defined_table_infos.append(new_table_info)

    def init_table(self, table_map_policy=AutoMergeTableMapPolicy()):
        if (not isinstance(table_map_policy, NoneTableMapPolicy)) and \
                (not isinstance(table_map_policy, AutoMergeTableMapPolicy)):
            raise TypeError("table_map_policy should be NoneTableMapPolicy or AutoMergeTableMapPolicy.")
        if len(self.user_defined_table_infos) == 0:
            raise ValueError("small table has not been created.")
        self.table_map_policy = table_map_policy
        self.table_create_infos = self.table_map_policy.map_table_infos(self.user_defined_table_infos)
        self.total_embedding_count = 0
        self.total_variable_table = []
        for table_info_ in self.table_create_infos:
            self.total_variable_table.append(tf.get_variable('ES' + str(self.total_embedding_count),
                                                             shape=[table_info_['max_vocabulary_size'],
                                                                    table_info_['embedding_dim']],
                                                             initializer=table_info_['initializer'],
                                                             dtype=tf.float32
                                                             ))
            self._npu_table_to_embedding_dim[self.total_embedding_count] = table_info_['embedding_dim']
            self.total_embedding_count += 1
        self.user_defined_table_infos = []

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
            table_embedding = tf.nn.embedding_lookup(self.total_variable_table[tid], table_input_hash)
            out_embedding_splited = tf.split(table_embedding, table_to_output_slots[tid], axis=1)
            for out_emb, sid in zip(out_embedding_splited, table_to_slot[tid]):
                output_slots[sid] = out_emb
        return output_slots

    def save_embedding(self, path, table_id):
        """ Operator for save values in table_id embedding table. """
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if table_id not in self._table_has_init:
            raise ValueError("this table has not yet initialized.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        with specified_ps_engine_scope():
            file_path_tensor = ops.convert_to_tensor(path, name="file_path")
            ps_id_tensor = ops.convert_to_tensor(-1, name="ps_id")
            table_id_tensor = ops.convert_to_tensor([table_id], name="table_id")
            embedding_table_export = \
                gen_npu_cpu_ops.embedding_table_export(file_path=file_path_tensor,
                                                       ps_id=ps_id_tensor,
                                                       table_id=table_id_tensor,
                                                       embedding_dim=[self._table_to_embedding_dim.get(table_id)],
                                                       value_total_len=[self._table_to_embedding_dim.get(table_id)],
                                                       export_mode="all",
                                                       only_var_flag=True,
                                                       file_type="bin")
            return tf.group([embedding_table_export])

    def save_embeddings(self, path):
        """ Operator for save values in all embedding tables. """
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        if not self._table_init:
            raise ValueError("Not any table has been initialized.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            for table_id in self._table_has_init:
                table_id_list.append(table_id)
                embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            file_path_tensor = ops.convert_to_tensor(path, name="file_path")
            ps_id_tensor = ops.convert_to_tensor(-1, name="ps_id")
            table_id_tensor = ops.convert_to_tensor(table_id_list, name="table_id")
            embedding_table_export = \
                gen_npu_cpu_ops.embedding_table_export(file_path=file_path_tensor,
                                                       ps_id=ps_id_tensor,
                                                       table_id=table_id_tensor,
                                                       embedding_dim=embedding_dim_list,
                                                       value_total_len=embedding_dim_list,
                                                       export_mode="all",
                                                       only_var_flag=True,
                                                       file_type="bin")
            return tf.group([embedding_table_export])

    def restore_embedding(self, path, table_id):
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if table_id not in self._table_has_init:
            raise ValueError("this table has not yet initialized.")
        with specified_ps_engine_scope():
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ops.convert_to_tensor(-1),
                                                       file_path=ops.convert_to_tensor(path),
                                                       table_id=ops.convert_to_tensor([table_id]),
                                                       embedding_dim=[self._table_to_embedding_dim.get(table_id)],
                                                       value_total_len=[self._table_to_embedding_dim.get(table_id)],
                                                       only_var_flag=True,
                                                       file_type="bin")
            return tf.group([embedding_table_import])

    def restore_embeddings(self, path):
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._table_init:
            raise ValueError("Not any table has been initialized.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            for table_id in self._table_has_init:
                table_id_list.append(table_id)
                embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ops.convert_to_tensor(-1),
                                                       file_path=ops.convert_to_tensor(path),
                                                       table_id=ops.convert_to_tensor(table_id_list),
                                                       embedding_dim=embedding_dim_list,
                                                       value_total_len=embedding_dim_list,
                                                       only_var_flag=True,
                                                       file_type="bin")
            return tf.group([embedding_table_import])

    def save_checkpoint(self, path, table_id):
        """ Operator for save values and optimizer params in table_id embedding table. """
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if table_id not in self._table_has_init:
            raise ValueError("this table has not yet initialized.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        with specified_ps_engine_scope():
            file_path_tensor = ops.convert_to_tensor(path, name="file_path")
            ps_id_tensor = ops.convert_to_tensor(-1, name="ps_id")
            table_id_tensor = ops.convert_to_tensor([table_id], name="table_id")
            embedding_table_export = \
                gen_npu_cpu_ops.embedding_table_export(file_path=file_path_tensor,
                                                       ps_id=ps_id_tensor,
                                                       table_id=table_id_tensor,
                                                       embedding_dim=[self._table_to_embedding_dim.get(table_id)],
                                                       value_total_len=[self._table_to_embedding_dim.get(table_id) *
                                                                       (self._table_to_slot_var_num.get(table_id) + 1)],
                                                       export_mode="all",
                                                       only_var_flag=False,
                                                       file_type="bin")
            with tf.control_dependencies([embedding_table_export]):
                embedding_compute_var_export = \
                    gen_npu_cpu_ops.embedding_compute_var_export(file_path=file_path_tensor,
                                                                 ps_id=ps_id_tensor,
                                                                 table_id=table_id_tensor)
                return tf.group([embedding_compute_var_export])

    def save_checkpoints(self, path):
        """ Operator for save values and optimizer params in all embedding tables. """
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        if not self._table_init:
            raise ValueError("Not any table has been initialized.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            value_total_len_list = []
            for table_id in self._table_has_init:
                table_id_list.append(table_id)
                embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
                value_total_len_list.append(self._table_to_embedding_dim.get(table_id) *
                                            (self._table_to_slot_var_num.get(table_id) + 1))
            file_path_tensor = ops.convert_to_tensor(path, name="file_path")
            ps_id_tensor = ops.convert_to_tensor(-1, name="ps_id")
            table_id_tensor = ops.convert_to_tensor(table_id_list, name="table_id")
            embedding_table_export = \
                gen_npu_cpu_ops.embedding_table_export(file_path=file_path_tensor,
                                                       ps_id=ps_id_tensor,
                                                       table_id=table_id_tensor,
                                                       embedding_dim=embedding_dim_list,
                                                       value_total_len=value_total_len_list,
                                                       export_mode="all",
                                                       only_var_flag=False,
                                                       file_type="bin")
            with tf.control_dependencies([embedding_table_export]):
                embedding_compute_var_export = \
                    gen_npu_cpu_ops.embedding_compute_var_export(file_path=file_path_tensor,
                                                                 ps_id=ps_id_tensor,
                                                                 table_id=table_id_tensor)
                return tf.group([embedding_compute_var_export])

    def restore_checkpoint(self, path, table_id):
        """ Operator for restore values and optimizer params in table_id embedding table. """
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if table_id not in self._table_has_init:
            raise ValueError("this table has not yet initialized.")
        with specified_ps_engine_scope():
            file_path_tensor = ops.convert_to_tensor(path, name="file_path")
            ps_id_tensor = ops.convert_to_tensor(-1, name="ps_id")
            table_id_tensor = ops.convert_to_tensor([table_id], name="table_id")
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ps_id_tensor,
                                                       file_path=file_path_tensor,
                                                       table_id=table_id_tensor,
                                                       embedding_dim=[self._table_to_embedding_dim.get(table_id)],
                                                       value_total_len=[self._table_to_embedding_dim.get(table_id) *
                                                                       (self._table_to_slot_var_num.get(table_id) + 1)],
                                                       only_var_flag=False,
                                                       file_type="bin")
            with tf.control_dependencies([embedding_table_import]):
                embedding_compute_var_import = \
                    gen_npu_cpu_ops.embedding_compute_var_import(file_path=file_path_tensor,
                                                                 ps_id=ps_id_tensor,
                                                                 table_id=table_id_tensor)
                return tf.group([embedding_compute_var_import])

    def restore_checkpoints(self, path):
        """ Operator for restore values and optimizer params in all embedding tables. """
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._table_init:
            raise ValueError("Not any table has been initialized.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            value_total_len_list = []
            for table_id in self._table_has_init:
                table_id_list.append(table_id)
                embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
                value_total_len_list.append(self._table_to_embedding_dim.get(table_id) *
                                            (self._table_to_slot_var_num.get(table_id) + 1))
            file_path_tensor = ops.convert_to_tensor(path, name="file_path")
            ps_id_tensor = ops.convert_to_tensor(-1, name="ps_id")
            table_id_tensor = ops.convert_to_tensor(table_id_list, name="table_id")
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ps_id_tensor,
                                                       file_path=file_path_tensor,
                                                       table_id=table_id_tensor,
                                                       embedding_dim=embedding_dim_list,
                                                       value_total_len=value_total_len_list,
                                                       only_var_flag=False,
                                                       file_type="bin")
            with tf.control_dependencies([embedding_table_import]):
                embedding_compute_var_import = \
                    gen_npu_cpu_ops.embedding_compute_var_import(file_path=file_path_tensor,
                                                                 ps_id=ps_id_tensor,
                                                                 table_id=table_id_tensor)
                return tf.group([embedding_compute_var_import])

    def save_incremental_embedding(self, path, table_id):
        """ Operator for save incremental values in table_id embedding table. """
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if table_id not in self._table_has_init:
            raise ValueError("this table has not yet initialized.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        with specified_ps_engine_scope():
            file_path_tensor = ops.convert_to_tensor(path, name="file_path")
            ps_id_tensor = ops.convert_to_tensor(-1, name="ps_id")
            table_id_tensor = ops.convert_to_tensor([table_id], name="table_id")
            embedding_table_export = \
                gen_npu_cpu_ops.embedding_table_export(file_path=file_path_tensor,
                                                       ps_id=ps_id_tensor,
                                                       table_id=table_id_tensor,
                                                       embedding_dim=[self._table_to_embedding_dim.get(table_id)],
                                                       value_total_len=[self._table_to_embedding_dim.get(table_id)],
                                                       export_mode="new",
                                                       only_var_flag=True,
                                                       file_type="bin")
            return tf.group([embedding_table_export])

    def save_incremental_embeddings(self, path):
        """ Operator for save incremental values in all embedding tables. """
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        if not self._table_init:
            raise ValueError("Not any table has been initialized.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            for table_id in self._table_has_init:
                table_id_list.append(table_id)
                embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            file_path_tensor = ops.convert_to_tensor(path, name="file_path")
            ps_id_tensor = ops.convert_to_tensor(-1, name="ps_id")
            table_id_tensor = ops.convert_to_tensor(table_id_list, name="table_id")
            embedding_table_export = \
                gen_npu_cpu_ops.embedding_table_export(file_path=file_path_tensor,
                                                       ps_id=ps_id_tensor,
                                                       table_id=table_id_tensor,
                                                       embedding_dim=embedding_dim_list,
                                                       value_total_len=embedding_dim_list,
                                                       export_mode="new",
                                                       only_var_flag=True,
                                                       file_type="bin")
            return tf.group([embedding_table_export])

    def restore_incremental_embedding(self, path, table_id):
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if table_id not in self._table_has_init:
            raise ValueError("this table has not yet initialized.")
        with specified_ps_engine_scope():
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ops.convert_to_tensor(-1),
                                                       file_path=ops.convert_to_tensor(path),
                                                       table_id=ops.convert_to_tensor([table_id]),
                                                       embedding_dim=[self._table_to_embedding_dim.get(table_id)],
                                                       value_total_len=[self._table_to_embedding_dim.get(table_id)],
                                                       only_var_flag=True,
                                                       file_type="bin")
            return tf.group([embedding_table_import])

    def restore_incremental_embeddings(self, path):
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if not self._table_init:
            raise ValueError("Not any table has been initialized.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            for table_id in self._table_has_init:
                table_id_list.append(table_id)
                embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ops.convert_to_tensor(-1),
                                                       file_path=ops.convert_to_tensor(path),
                                                       table_id=ops.convert_to_tensor(table_id_list),
                                                       embedding_dim=embedding_dim_list,
                                                       value_total_len=embedding_dim_list,
                                                       only_var_flag=True,
                                                       file_type="bin")
            return tf.group([embedding_table_import])

    def _init_hashmap_and_table_import(self, bucket_size, table_id, embedding_dim, ev_option):
        if (ev_option is not None) and (ev_option.filter_option is not None):
            filter_mode = "counter"
            self._table_to_counter_filter[table_id] = ev_option.filter_option
        else:
            filter_mode = "no_filter"
        with tf.control_dependencies([self._init_partition_maps.get(table_id)]):
            if self._train_mode:
                if self._train_level:
                    seed1, seed2 = random_seed.get_seed(None)
                    self._init_embedding_hash_maps[table_id] = \
                        gen_npu_cpu_ops.init_embedding_hashmap(table_id=ops.convert_to_tensor(table_id),
                                                               bucket_size=bucket_size,
                                                               value_total_len=embedding_dim * (self.slot_vars_num + 1),
                                                               embedding_dim=embedding_dim,
                                                               initializer_mode=
                                                               self._table_to_initializer.get(table_id)
                                                               .initializer_mode,
                                                               constant_value=
                                                               self._table_to_initializer.get(table_id).constant_value,
                                                               min=self._table_to_initializer.get(table_id).min,
                                                               max=self._table_to_initializer.get(table_id).max,
                                                               mu=self._table_to_initializer.get(table_id).mu,
                                                               sigma=self._table_to_initializer.get(table_id).sigma,
                                                               seed=seed1, seed2=seed2, filter_mode=filter_mode)
                else:
                    self._init_embedding_hash_maps[table_id] = \
                        gen_npu_cpu_ops.init_embedding_hashmap(table_id=ops.convert_to_tensor(table_id),
                                                               bucket_size=bucket_size,
                                                               value_total_len=embedding_dim * (self.slot_vars_num + 1),
                                                               embedding_dim=embedding_dim,
                                                               initializer_mode=None, constant_value=None,
                                                               min=None, max=None, mu=None, sigma=None,
                                                               seed=None, seed2=None, filter_mode=filter_mode)
            else:
                self._init_embedding_hash_maps[table_id] = \
                    gen_npu_cpu_ops.init_embedding_hashmap(table_id=ops.convert_to_tensor(table_id),
                                                           bucket_size=bucket_size,
                                                           value_total_len=embedding_dim,
                                                           embedding_dim=embedding_dim,
                                                           initializer_mode=None, constant_value=None,
                                                           min=None, max=None, mu=None, sigma=None,
                                                           seed=None, seed2=None, filter_mode=filter_mode)
        self._init_flag = True
        self._table_init = True
        if self._train_mode:
            return tf.group(
                [tf.initializers.variables(self._optimizer.variables()), self._init_embedding_hash_maps.get(table_id)])
        else:
            return tf.group([self._init_embedding_hash_maps.get(table_id)])
