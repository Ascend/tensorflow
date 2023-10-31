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
import typing
import re
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


class EsInitializer:
    """Initializer for embedding service table."""

    def __init__(self, initializer_mode, min=-0.01, max=0.01, constant_value=1.0, mu=0.0, sigma=1.0, seed=0):
        self.initializer_mode = initializer_mode
        self.min = min
        self.max = max
        self.constant_value = constant_value
        self.mu = mu
        self.sigma = sigma
        self.seed = seed


# 提供 embedding_service table initializer method
# min 下限值, float 类型
# max 上限值, float 类型
# initializer_mode 初始化方式, string 类型
# constant_value 常量初始化的常量值, float 类型
# mu 正态分布的均值, float 类型
# sigma 正态分布的标准差, float 类型
def es_initializer(initializer_mode, min=-2.0, max=2.0, constant_value=0.0, mu=0.0, sigma=1.0, seed=0):
    """Operator for init initializer."""
    if initializer_mode is None:
        raise ValueError("initializer_mode can not be None.")
    if initializer_mode == 'random_uniform':
        if (min is None) or (max is None) or \
                (not isinstance(min, (float, int))) or (not isinstance(max, (float, int))):
            raise ValueError("If initializer is random_uniform, min and max can not be None, must be int or float.")
    if initializer_mode == 'truncated_normal':
        if (min is None) or (max is None) or (mu is None) or (sigma is None):
            raise ValueError("If initializer is truncated_normal, min, max, mu and sigma can not be None")
        if (not isinstance(min, (float, int))) or (not isinstance(max, (float, int))) or \
                (not isinstance(mu, (float, int))) or (not isinstance(sigma, (float, int))):
            raise ValueError("If initializer is truncated_normal, min, max, mu and sigma must be int or float.")
    if initializer_mode == 'constant':
        if (constant_value is None) or (not isinstance(constant_value, (float, int))):
            raise ValueError("If initializer is constant, constant_value can not be None, must be float or int.")
    if min > max:
        raise ValueError("Initializer min value can not be larger than max value.")
    if (initializer_mode != 'constant') and (initializer_mode != 'random_uniform') and \
            (initializer_mode != 'truncated_normal'):
        raise ValueError("Initializer mode must be random_uniform or truncated normal or constant.")
    return EsInitializer(initializer_mode=initializer_mode,
                         min=min,
                         max=max,
                         constant_value=constant_value,
                         mu=mu,
                         sigma=sigma,
                         seed=seed)


class ESWorker:
    """ Embedding service class. """

    def __init__(self):
        env_dist = os.environ
        cluster_config_from_env = env_dist.get("ESCLUSTER_CONFIG_PATH")
        if cluster_config_from_env is None:
            raise ValueError("EsClusterConfig env is null.")
        es_cluster_config = cluster_config_from_env
        with open(es_cluster_config, encoding='utf-8') as a:
            es_cluster_config_json = json.load(a)
            self._es_cluster_conf = json.dumps(es_cluster_config_json)
            self._ps_num = int(es_cluster_config_json["psNum"])
            self._ps_ids = []
            self._ps_ids_list = es_cluster_config_json["psCluster"]
            for each_ps in self._ps_ids_list:
                self._ps_ids.append(each_ps["id"])

        if self._ps_num > 15:
            raise ValueError("PS num can not exceed 15, please check config params.")

        self._init_embedding_hash_maps = {}
        self._init_partition_maps = {}
        # storage each ps table's params
        self._table_to_embedding_dim = {}
        self._table_to_max_num = {}
        self._table_to_optimizer = {}
        self._table_to_initializer = {}
        self._table_to_slot_var_num = {}
        self._table_to_counter_filter = {}
        self._train_mode = True
        self._train_level = False
        self._optimizer = None
        self._init_table_flag = False

        self._small_table_name_list = []
        self._ps_table_count = 0
        self._table_name_to_id = {}
        self._table_id_to_name = {}
        self._table_id_to_initializer = {}

        self._ps_table_id_list = []
        # storage lookup: table_id list, lookup result list, lookup key list
        self._ps_lookup_index = 0
        self._ps_table_has_lookup = []
        self._ps_table_lookup_key = []
        self._ps_table_lookup_result = []
        # storage all inited table names
        self._table_name_has_init = []
        # only storage all inited PS table names
        self._ps_table_name_list = []
        # now only use for adagrad accum
        self._ps_table_id_to_optimizer_mode = {}
        self._ps_table_id_to_optimizer_params = {}

        # use for small table merge
        self.user_defined_table_infos = []
        self.table_map_policy = None
        self.table_create_infos = []
        self.total_variable_table = []
        self.total_embedding_count = 0
        self._npu_table_to_embedding_dim = {}
        self._need_table_merge = False
        self._only_merge_to_one_table = True
        # use for counter filter
        self._table_use_counter_filter = {}

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
        self._table_id_to_initializer[table_id] = EsInitializer(min=min,
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

    # new version
    # 提供embedding init功能
    # @param vocabulary_size 表的初始大小, int 类型
    # @param table_id, int32 类型
    # @param max_batch_size, int32 类型
    # @param optimizer, 支持EmbeddingAdamOptimizer，EmbeddingAdagradOptimizer，EmbeddingAdamwOptimizer
    # @param initializer, string 类型
    # @param embedding_dim, int32 类型
    def get_embedding_variable(self, name, init_vocabulary_size, embedding_dim, key_dtype=tf.int64,
                               value_dtype=tf.float32, partitioner=None,
                               initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01, seed=1234),
                               embedding_type="PS", ev_option=None, max_feature_count=None, multihot_lens=None,
                               optimizer=None, allow_merge=True):
        """ Operator for get embedding variable according to embedding type. """
        if (name is None) or (init_vocabulary_size is None) or (embedding_dim is None):
            raise ValueError("table name, init_vocabulary_size and embedding_dim can not be None.")
        if not isinstance(name, str):
            raise TypeError("embedding table name must be string.")
        regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
        if regex.search(name) is not None:
            raise ValueError("table name contains illegal character.")
        if (not isinstance(init_vocabulary_size, int)) or (not isinstance(embedding_dim, int)):
            raise ValueError("init_vocabulary_size and embedding_dim must be int.")
        if init_vocabulary_size < 0:
            raise ValueError("init_vocabulary_size can not be smaller than zero.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")
        if (embedding_type != "PS") and (embedding_type != "data_parallel"):
            raise TypeError("embedding_type must be PS or data_parallel")

        if embedding_type == "data_parallel":
            if name not in self._small_table_name_list:
                self._small_table_name_list.append(name)
            else:
                raise ValueError("This small table has been initialized.")
            if (init_vocabulary_size is None) or (embedding_dim is None) or (multihot_lens is None):
                raise ValueError("max_vocabulary_size or embedding_dim or multihot_lens can not be None.")
            if (key_dtype is None) or (value_dtype is None):
                raise ValueError("key_dtype and value_dtype can not be None.")
            if (key_dtype is not tf.int64) or (value_dtype is not tf.float32):
                raise TypeError("key_dtype only support tf.int64, value_dtype only support tf.float32 now.")
            if (not isinstance(init_vocabulary_size, int)) or (not isinstance(embedding_dim, int)) or \
                    (not isinstance(multihot_lens, int)) or (not isinstance(allow_merge, bool)):
                raise TypeError("init_vocabulary_size, embedding_dim, multihot_lens must be int,"
                                "allow_merge must be bool.")
            if init_vocabulary_size <= 0 or embedding_dim <= 0 or multihot_lens <= 0:
                raise ValueError("init_vocabulary_size, embedding_dim, multihot_lens must be greater than zero.")
            if initializer is None:
                raise ValueError("Initializer can not be None.")
            if allow_merge:
                self._need_table_merge = True
            if isinstance(initializer, EsInitializer):
                if initializer.initializer_mode == "random_uniform":
                    self._table_id_to_initializer[table_id] = \
                        tf.random_uniform_initializer(minval=initializer.min, maxval=initializer.max,
                                                      seed=initializer.seed, dtype=value_dtype)
                elif initializer.initializer_mode == "truncated_normal":
                    self._table_id_to_initializer[table_id] = \
                        tf.truncated_normal_initializer(stddev=initializer.stddev, mean=initializer.mean,
                                                        seed=initializer.seed, dtype=value_dtype)
                elif initializer.initializer_mode == "constant":
                    self._table_id_to_initializer[table_id] = \
                        tf.constant_initializer(value=initializer.value, dtype=value_dtype)
            elif not callable(initializer):
                if ops.convert_to_tensor(initializer).dtype.base_dtype != tf.float32:
                    raise ValueError("Initializer type '%s' and explict dtype tf.float32 don't match." % init_dtype)
            new_small_table_info = dict(
                max_vocabulary_size=init_vocabulary_size,
                embedding_dim=embedding_dim,
                multihot_lens=multihot_lens,
                allow_merge=allow_merge,
                initializer=initializer)
            self.user_defined_table_infos.append(new_small_table_info)
            return new_small_table_info

        elif embedding_type == "PS":
            if max_feature_count is None:
                raise ValueError("For ps table, max_feature_count can not be None.")
            if (ev_option is not None) and (not isinstance(ev_option, EmbeddingVariableOption)):
                raise TypeError("For ps table, ev_option must be EmbeddingVariableOption type.")
            if not isinstance(max_feature_count, int):
                raise ValueError("For ps table, max_feature_count must be int.")
            if init_vocabulary_size >= _INT32_MAX_VALUE:
                raise ValueError("init_vocabulary_size exceeds int32 max value.")
            if max_feature_count <= 0:
                raise ValueError("For ps table, max_feature_count must be greater than zero.")
            if name not in self._table_name_has_init:
                table_id = self._ps_table_count
                self._table_name_to_id[name] = table_id
                self._table_id_to_name[table_id] = name
                self._ps_table_count += 1
                self._table_name_has_init.append(name)
            else:
                raise ValueError("This table has been initialized.")
            self._ps_lookup_index = self._ps_table_count
            self._table_to_embedding_dim[table_id] = embedding_dim
            self._table_to_max_num[table_id] = max_feature_count
            # storage the table id for embedding PS table
            self._ps_table_id_list.append(table_id)
            self._ps_table_name_list.append(name)
            if len(self._ps_table_id_list) > 10:
                raise ValueError("Now only 10 PS embedding tables can be init.")
            bucket_size = math.ceil(init_vocabulary_size / self._ps_num)
            if optimizer is None:
                self._train_mode = False
                self._table_to_slot_var_num[table_id] = 0
            else:
                if (not isinstance(optimizer, embedding_optimizer.AdamOptimizer)) and \
                        (not isinstance(optimizer, embedding_optimizer.AdagradOptimizer)) and \
                        (not isinstance(optimizer, embedding_optimizer.AdamWOptimizer)):
                    raise ValueError(
                        "optimizer should be embedding_optimizer AdamOptimizer, AdagradOptimizer or AdamWOptimizer.")
                if initializer is not None:
                    if isinstance(initializer, EsInitializer):
                        self._table_id_to_initializer[table_id] = initializer
                    elif isinstance(initializer, tf.initializers.truncated_normal):
                        if initializer.dtype != tf.float32:
                            raise TypeError("initializer dtype error.")
                        self._table_id_to_initializer[table_id] = \
                            EsInitializer(initializer_mode="truncated_normal", mu=initializer.mean,
                                          sigma=initializer.stddev, seed=initializer.seed)
                    elif isinstance(initializer, tf.initializers.random_uniform):
                        if initializer.dtype != tf.float32:
                            raise TypeError("initializer dtype error.")
                        self._table_id_to_initializer[table_id] = \
                            EsInitializer(initializer_mode="random_uniform", min=initializer.minval,
                                          max=initializer.maxval, seed=initializer.seed)
                    elif isinstance(initializer, tf.initializers.constant):
                        if initializer.dtype != tf.float32:
                            raise TypeError("initializer dtype error.")
                        self._table_id_to_initializer[table_id] = \
                            EsInitializer(initializer_mode="constant", constant_value=initializer.value)
                    else:
                        raise TypeError("initializer must be EsInitializer or tensorflow initializer, and only support"
                                        "random_uniform, truncated_normal and constant value.")
                self._optimizer = optimizer
                self._optimizer._embedding_dims = embedding_dim
                self._optimizer._max_nums = max_feature_count
                self._optimizer._es_cluster_configs = self._es_cluster_conf
                self._table_to_optimizer[table_id] = self._optimizer
                self._ps_table_id_to_optimizer_params[table_id] = []
                # adam, adamw include m and v, 2 slots; adagrad include accumulator, 1 slot
                if isinstance(self._optimizer, embedding_optimizer.AdagradOptimizer):
                    self._table_to_slot_var_num[table_id] = 1
                else:
                    self._table_to_slot_var_num[table_id] = 2
                # new train or continue train from a checkpoint
                if initializer is not None:
                    self._train_level = True
            with specified_ps_engine_scope():
                self._init_partition_maps[table_id] = \
                    gen_npu_cpu_ops.init_partition_map(ps_num=ops.convert_to_tensor(self._ps_num),
                                                       ps_ids=ops.convert_to_tensor(self._ps_ids),
                                                       partition_num=65537)
                self._init_partition_maps.get(table_id)._set_attr("_embedding_dim",
                                                                  attr_value_pb2.AttrValue(i=embedding_dim))
                self._init_partition_maps.get(table_id)._set_attr("_max_key_num",
                                                                  attr_value_pb2.AttrValue(i=max_feature_count))
                return self._init_hashmap_and_table_import(bucket_size, table_id, embedding_dim, ev_option)

    # old version
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
        if table_id in self._ps_table_id_list:
            raise ValueError("this table has already initialized.")

        self._table_to_embedding_dim[table_id] = embedding_dim
        self._table_to_max_num[table_id] = max_batch_size
        self._table_id_to_name[table_id] = str(table_id)
        self._ps_table_id_list.append(table_id)
        self._ps_table_name_list.append(str(table_id))
        if len(self._ps_table_id_list) > 10:
            raise ValueError("Now only 10 embedding tables can be init.")
        bucket_size = math.ceil(vocabulary_size / self._ps_num)
        if (self._table_id_to_initializer.get(table_id) is None) and (initializer is not None):
            self._table_id_to_initializer[table_id] = EsInitializer(min=-2,
                                                                    max=2,
                                                                    initializer_mode=initializer,
                                                                    constant_value=0,
                                                                    mu=0.0,
                                                                    sigma=1.0)
        if optimizer is None:
            self._train_mode = False
            self._table_to_slot_var_num[table_id] = 0
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
            self._ps_table_id_to_optimizer_params[table_id] = []
            # adam include m and v, 2 slots; adagrad include accumulator, 1 slot
            if isinstance(self._optimizer, embedding_optimizer.AdagradOptimizer):
                self._table_to_slot_var_num[table_id] = 1
            else:
                self._table_to_slot_var_num[table_id] = 2
            if (initializer is not None) or (self._table_to_initializer.get(table_id) is not None):
                self._train_level = True

        with specified_ps_engine_scope():
            self._init_partition_maps[table_id] = \
                gen_npu_cpu_ops.init_partition_map(ps_num=ops.convert_to_tensor(self._ps_num),
                                                   ps_ids=ops.convert_to_tensor(self._ps_ids),
                                                   partition_num=65537)
            self._init_partition_maps.get(table_id)._set_attr("_embedding_dim",
                                                              attr_value_pb2.AttrValue(i=embedding_dim))
            self._init_partition_maps.get(table_id)._set_attr("_max_key_num",
                                                              attr_value_pb2.AttrValue(i=max_batch_size))
            return self._init_hashmap_and_table_import(bucket_size, table_id, embedding_dim, ev_option)

    # new version
    # 提供embedding lookup功能
    # @param name str 类型
    # @param ids int64 类型
    # @return values float32 类型
    def embedding_lookup(self, name: str, ids: typing.Any, key_num_input=None, unique_indices=None):
        """ Operator for look up in embedding table. """
        if (name is None) or (ids is None):
            raise ValueError("table name or ids must be specified.")
        if not isinstance(name, str):
            raise TypeError("embedding table name must be string.")
        regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
        if regex.search(name) is not None:
            raise ValueError("table name contains illegal character.")
        if ids.dtype != tf.int64:
            raise ValueError("dtype of ids must be tf.int64.")
        if not self._init_table_flag:
            raise ValueError("embedding table must init first!")

        table_id = self._table_name_to_id.get(name)
        if table_id not in self._ps_table_id_list:
            raise ValueError("this ps table has not yet initialized.")

        if self._table_to_counter_filter.get(table_id) is not None:
            filter_mode = "counter"
            self._filter_freq = self._table_to_counter_filter.get(table_id).filter_freq
            self._default_key_or_value = self._table_to_counter_filter.get(table_id).default_key_or_value
            self._default_key = self._table_to_counter_filter.get(table_id).default_key
            self._default_value = self._table_to_counter_filter.get(table_id).default_value
        else:
            filter_mode = "no_filter"
            self._default_value = -1

        use_host_unique = False
        if (key_num_input is not None) and (unique_indices is not None):
            use_host_unique = True

        if self._train_mode:
            if use_host_unique:
                result = gen_npu_cpu_ops. \
                    fused_remote_lookup_with_unique(table_id=ops.convert_to_tensor(table_id),
                                                    keys=ids,
                                                    key_num_input=key_num_input,
                                                    unique_indices=unique_indices,
                                                    embedding_dim=self._table_to_embedding_dim.get(table_id),
                                                    initializer_mode=self._table_id_to_initializer.get(table_id)
                                                    .initializer_mode,
                                                    constant_value=self._table_id_to_initializer.get(table_id)
                                                    .constant_value,
                                                    min=self._table_id_to_initializer.get(table_id).min,
                                                    max=self._table_id_to_initializer.get(table_id).max,
                                                    mu=self._table_id_to_initializer.get(table_id).mu,
                                                    sigma=self._table_id_to_initializer.get(table_id).sigma,
                                                    seed=self._table_id_to_initializer.get(table_id).seed,
                                                    seed2=self._table_id_to_initializer.get(table_id).seed,
                                                    value_total_len=self._table_to_embedding_dim
                                                    .get(table_id) * (self._table_to_slot_var_num.get(table_id) + 1),
                                                    filter_mode=filter_mode,
                                                    filter_freq=self._filter_freq,
                                                    default_key_or_value=self._default_key_or_value,
                                                    default_key=self._default_key,
                                                    default_value=self._default_value,
                                                    optimizer_mode=self._ps_table_id_to_optimizer_mode.get(table_id),
                                                    optimizer_params=self._ps_table_id_to_optimizer_params.get(table_id)
                                                    )
            else:
                result = gen_npu_cpu_ops. \
                    embedding_table_find_and_init(table_id=ops.convert_to_tensor(table_id),
                                                  keys=ids,
                                                  embedding_dim=self._table_to_embedding_dim.get(table_id),
                                                  initializer_mode=self._table_id_to_initializer.get(table_id)
                                                  .initializer_mode,
                                                  constant_value=self._table_id_to_initializer.get(table_id)
                                                  .constant_value,
                                                  min=self._table_id_to_initializer.get(table_id).min,
                                                  max=self._table_id_to_initializer.get(table_id).max,
                                                  mu=self._table_id_to_initializer.get(table_id).mu,
                                                  sigma=self._table_id_to_initializer.get(table_id).sigma,
                                                  seed=self._table_id_to_initializer.get(table_id).seed,
                                                  seed2=self._table_id_to_initializer.get(table_id).seed,
                                                  value_total_len=self._table_to_embedding_dim.get(table_id) *
                                                                  (self._table_to_slot_var_num.get(table_id) + 1),
                                                  filter_mode=filter_mode,
                                                  filter_freq=self._filter_freq,
                                                  default_key_or_value=self._default_key_or_value,
                                                  default_key=self._default_key,
                                                  default_value=self._default_value,
                                                  optimizer_mode=self._ps_table_id_to_optimizer_mode.get(table_id),
                                                  optimizer_params=self._ps_table_id_to_optimizer_params.get(table_id)
                                                  )

        else:
            result = gen_npu_cpu_ops.embedding_table_find(table_id=ops.convert_to_tensor(table_id),
                                                          keys=ids,
                                                          embedding_dim=self._table_to_embedding_dim.get(table_id),
                                                          default_value=self._default_value)
        self._filter_freq = None
        self._default_key_or_value = True
        self._default_key = None
        self._default_value = None
        result.op._set_attr("_embedding_dim", attr_value_pb2.AttrValue(i=self._table_to_embedding_dim.get(table_id)))
        result.op._set_attr("_max_key_num", attr_value_pb2.AttrValue(i=self._table_to_max_num.get(table_id)))
        result.op._set_attr("_use_counter_filter",
                            attr_value_pb2.AttrValue(i=self._table_use_counter_filter.get(table_id)))
        if self._ps_lookup_index != 0:
            self._ps_table_has_lookup.append(table_id)
            self._ps_table_lookup_key.append(ids)
            self._ps_table_lookup_result.append(result)
            self._ps_lookup_index = self._ps_lookup_index - 1
        return result

    # old version
    # 提供embedding lookup功能
    # @param table_id int32 类型
    # @param input_ids int64 类型
    # @return values float32 类型
    def embedding_lookup_v1(self, table_id: int, input_ids: typing.Any):
        """ Operator for look up in embedding table. """
        if (table_id is None) or (input_ids is None):
            raise ValueError("table_id or input_ids must be specified.")
        if not isinstance(table_id, int):
            raise ValueError("type of table_id must be int.")
        if input_ids.dtype != tf.int64:
            raise ValueError("dtype of input_ids must be tf.int64.")
        if table_id < 0:
            raise ValueError("table_id can not be smaller than zero.")
        if not self._init_table_flag:
            raise ValueError("embedding must init first!")
        if table_id not in self._ps_table_id_list:
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
                                              initializer_mode=self._table_id_to_initializer.get(table_id)
                                              .initializer_mode,
                                              constant_value=self._table_id_to_initializer.get(table_id).constant_value,
                                              min=self._table_id_to_initializer.get(table_id).min,
                                              max=self._table_id_to_initializer.get(table_id).max,
                                              mu=self._table_id_to_initializer.get(table_id).mu,
                                              sigma=self._table_id_to_initializer.get(table_id).sigma,
                                              seed=seed1,
                                              seed2=seed2,
                                              value_total_len=self._table_to_embedding_dim.get(table_id) *
                                                              (self._table_to_slot_var_num.get(table_id) + 1),
                                              filter_mode=filter_mode,
                                              filter_freq=self._filter_freq,
                                              default_key_or_value=self._default_key_or_value,
                                              default_key=self._default_key,
                                              default_value=self._default_value,
                                              optimizer_mode=self._ps_table_id_to_optimizer_mode.get(table_id),
                                              optimizer_params=self._ps_table_id_to_optimizer_params.get(table_id)
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
        result.op._set_attr("_use_counter_filter",
                            attr_value_pb2.AttrValue(i=self._table_use_counter_filter.get(table_id)))
        return result

    # new version
    # 提供embedding update功能
    # @param loss 类型
    def embedding_update(self, loss):
        """ Operator for update in embedding table. """
        params = self._ps_table_lookup_result
        input_ids_list = self._ps_table_lookup_key
        table_ids = self._ps_table_has_lookup
        if (loss is None) or (params is None) or (table_ids is None) or (input_ids_list is None):
            raise ValueError("loss or params or table_ids or input_ids_list is None.")
        if (isinstance(loss, str)) or (isinstance(params, str)) or isinstance(table_ids, str) or \
                isinstance(input_ids_list, str):
            raise ValueError("loss, params, table_ids and input_ids_list can not be str.")
        if not self._init_table_flag:
            raise ValueError("embedding must init first!")
        if (not isinstance(params, (list, tuple)) and not isinstance(table_ids, (list, tuple))
                and not isinstance(input_ids_list, (list, tuple))):
            params = [params]
            table_ids = [table_ids]
            input_ids_list = [input_ids_list]
        for table_id in table_ids:
            if table_id not in self._ps_table_id_list:
                raise ValueError("this table has not yet initialized.")
        if (len(params) != len(table_ids)) or (len(params) != len(input_ids_list)) \
                or (len(table_ids) != len(input_ids_list)):
            raise ValueError("The length of params, table_ids, input_ids_list should be equal.")
        embedding_grads = tf.gradients(loss, params)
        update_op = []
        self._ps_table_lookup_result = []
        self._ps_table_lookup_key = []
        self._ps_table_has_lookup = []
        with specified_ps_engine_scope():
            for i in range(len(table_ids)):
                params_grads = [tf.IndexedSlices(embedding_grads[i], input_ids_list[i], dense_shape=params[i].shape)]
                var_refs = [NpuEmbeddingResource(table_ids[i])]
                update_op.append(
                    self._table_to_optimizer.get(table_ids[i]).apply_gradients(list(zip(params_grads, var_refs))))
            return update_op

    # old version
    # 提供embedding update功能
    # @param loss 类型
    # @param params float32 类型
    # @param table_ids int32 类型
    # @param input_ids_list int64 类型
    def embedding_update_v1(self, loss, params, table_ids, input_ids_list):
        """ Operator for update in embedding table. """
        if (loss is None) or (params is None) or (table_ids is None) or (input_ids_list is None):
            raise ValueError("loss or params or table_ids or input_ids_list is None.")
        if (isinstance(loss, str)) or (isinstance(params, str)) or isinstance(table_ids, str) or \
                isinstance(input_ids_list, str):
            raise ValueError("loss, params, table_ids and input_ids_list can not be str.")
        if not self._init_table_flag:
            raise ValueError("embedding must init first!")
        if (not isinstance(params, (list, tuple)) and not isinstance(table_ids, (list, tuple))
                and not isinstance(input_ids_list, (list, tuple))):
            params = [params]
            table_ids = [table_ids]
            input_ids_list = [input_ids_list]
        for table_id in table_ids:
            if table_id not in self._ps_table_id_list:
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
        if filter_freq < 0:
            raise ValueError("filter_freq must can not be smaller than 0.")
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
        self._small_table_name_list = []

    # new version
    def embeddings_lookup(self, ids_list):
        if self.total_embedding_count != len(self.table_create_infos) or self.total_embedding_count == 0:
            raise ValueError("Must init_table() first!")
        if ids_list is None:
            raise ValueError("ids_list can not be None.")
        if ids_list.dtype != tf.int64:
            raise TypeError("dtype of ids_list must be tf.int64.")
        (in_slot_size_group, slot_to_table, table_to_input_group, \
         table_to_slot, table_to_output_slots) = \
            (self.table_map_policy.in_slot_size_group, self.table_map_policy.slot_to_table, \
             self.table_map_policy.table_to_input_groups, self.table_map_policy.table_to_slot, \
             self.table_map_policy.table_to_output_slots)

        ids_list_shape_list = ids_list.get_shape().as_list()
        total_in_slot_num = 0
        for in_slot_size in in_slot_size_group:
            total_in_slot_num += in_slot_size
        if ids_list_shape_list[1] != total_in_slot_num:
            raise ValueError("size of ids_list is not the same as all small tables.")

        if self.total_embedding_count == 1:
            output_slots = [None for _ in in_slot_size_group]
            tid = 0
            table_embedding = tf.nn.embedding_lookup(self.total_variable_table[tid], ids_list)
            out_embedding_splited = tf.split(table_embedding, table_to_output_slots[0], axis=1)
            for out_emb, sid in zip(out_embedding_splited, table_to_slot[0]):
                output_slots[sid] = out_emb
            return output_slots

        indices_split = tf.split(ids_list, in_slot_size_group, axis=1)
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

    # old version
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

    def save_embedding(self, name: str, path: str):
        """ Operator for save values in table_id embedding table. """
        if path is None or name is None:
            raise ValueError("table name, embedding table path can not be None.")
        if not isinstance(name, str):
            raise TypeError("embedding table name must be string.")
        regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
        if regex.search(name) is not None:
            raise ValueError("table name contains illegal character.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if name not in self._ps_table_name_list:
            raise ValueError("this table has not yet initialized.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        table_id = self._table_name_to_id.get(name)
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
                                                       file_type="bin",
                                                       table_name=[name])
            return tf.group([embedding_table_export])

    def save_embeddings(self, path: str):
        """ Operator for save values in all embedding tables. """
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            for table_id in self._ps_table_id_list:
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
                                                       file_type="bin",
                                                       table_name=self._ps_table_name_list)
            return tf.group([embedding_table_export])

    def restore_embedding(self, name: str, path: str):
        if path is None or name is None:
            raise ValueError("table name, embedding table path can not be None.")
        if not isinstance(name, str):
            raise TypeError("embedding table name must be string.")
        regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
        if regex.search(name) is not None:
            raise ValueError("table name contains illegal character.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if name not in self._ps_table_name_list:
            raise ValueError("this table has not yet initialized.")
        table_id = self._table_name_to_id.get(name)
        with specified_ps_engine_scope():
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ops.convert_to_tensor(-1),
                                                       file_path=ops.convert_to_tensor(path),
                                                       table_id=ops.convert_to_tensor([table_id]),
                                                       embedding_dim=[self._table_to_embedding_dim.get(table_id)],
                                                       value_total_len=[self._table_to_embedding_dim.get(table_id)],
                                                       only_var_flag=True,
                                                       file_type="bin",
                                                       table_name=[name])
            return tf.group([embedding_table_import])

    def restore_embeddings(self, path: str):
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            for table_id in self._ps_table_id_list:
                table_id_list.append(table_id)
                embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ops.convert_to_tensor(-1),
                                                       file_path=ops.convert_to_tensor(path),
                                                       table_id=ops.convert_to_tensor(table_id_list),
                                                       embedding_dim=embedding_dim_list,
                                                       value_total_len=embedding_dim_list,
                                                       only_var_flag=True,
                                                       file_type="bin",
                                                       table_name=self._ps_table_name_list)
            return tf.group([embedding_table_import])

    def save_checkpoint(self, name: str, path: str, save_filtered_features=False):
        """ Operator for save values and optimizer params in table_id embedding table. """
        if path is None or name is None:
            raise ValueError("table name, embedding table path can not be None.")
        if not isinstance(name, str):
            raise TypeError("embedding table name must be string.")
        regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
        if regex.search(name) is not None:
            raise ValueError("table name contains illegal character.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if name not in self._ps_table_name_list:
            raise ValueError("this table has not yet initialized.")
        if not isinstance(save_filtered_features, bool):
            raise TypeError("save_filtered_features must be bool.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        table_id = self._table_name_to_id.get(name)
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
                                                                        (self._table_to_slot_var_num.get(
                                                                            table_id) + 1)],
                                                       export_mode="all",
                                                       only_var_flag=False,
                                                       file_type="bin",
                                                       table_name=[name],
                                                       filter_export_flag=save_filtered_features)
            with tf.control_dependencies([embedding_table_export]):
                embedding_compute_var_export = \
                    gen_npu_cpu_ops.embedding_compute_var_export(file_path=file_path_tensor,
                                                                 ps_id=ps_id_tensor,
                                                                 table_id=table_id_tensor,
                                                                 table_name=[name])
                return tf.group([embedding_compute_var_export])

    def save_checkpoints(self, path: str, save_filtered_features=False):
        """ Operator for save values and optimizer params in all embedding tables. """
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if not isinstance(save_filtered_features, bool):
            raise TypeError("save_filtered_features must be bool.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            value_total_len_list = []
            for table_id in self._ps_table_id_list:
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
                                                       file_type="bin",
                                                       table_name=self._ps_table_name_list,
                                                       filter_export_flag=save_filtered_features)
            with tf.control_dependencies([embedding_table_export]):
                embedding_compute_var_export = \
                    gen_npu_cpu_ops.embedding_compute_var_export(file_path=file_path_tensor,
                                                                 ps_id=ps_id_tensor,
                                                                 table_id=table_id_tensor,
                                                                 table_name=self._ps_table_name_list)
                return tf.group([embedding_compute_var_export])

    def restore_checkpoint(self, name: str, path: str):
        """ Operator for restore values and optimizer params in table_id embedding table. """
        if path is None or name is None:
            raise ValueError("name, embedding table path can not be None.")
        if not isinstance(name, str):
            raise TypeError("embedding table name must be string.")
        regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
        if regex.search(name) is not None:
            raise ValueError("table name contains illegal character.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if name not in self._ps_table_name_list:
            raise ValueError("this table has not yet initialized.")
        table_id = self._table_name_to_id.get(name)
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
                                                                        (self._table_to_slot_var_num.get(
                                                                            table_id) + 1)],
                                                       only_var_flag=False,
                                                       file_type="bin",
                                                       table_name=[name])
            with tf.control_dependencies([embedding_table_import]):
                embedding_compute_var_import = \
                    gen_npu_cpu_ops.embedding_compute_var_import(file_path=file_path_tensor,
                                                                 ps_id=ps_id_tensor,
                                                                 table_id=table_id_tensor,
                                                                 table_name=[name])
                return tf.group([embedding_compute_var_import])

    def restore_checkpoints(self, path: str):
        """ Operator for restore values and optimizer params in all embedding tables. """
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            value_total_len_list = []
            for table_id in self._ps_table_id_list:
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
                                                       file_type="bin",
                                                       table_name=self._ps_table_name_list)
            with tf.control_dependencies([embedding_table_import]):
                embedding_compute_var_import = \
                    gen_npu_cpu_ops.embedding_compute_var_import(file_path=file_path_tensor,
                                                                 ps_id=ps_id_tensor,
                                                                 table_id=table_id_tensor,
                                                                 table_name=self._ps_table_name_list)
                return tf.group([embedding_compute_var_import])

    def save_incremental_embedding(self, name: str, path: str):
        """ Operator for save incremental values in table_id embedding table. """
        if path is None or name is None:
            raise ValueError("table name, embedding table path can not be None.")
        if not isinstance(name, str):
            raise TypeError("embedding table name must be string.")
        regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
        if regex.search(name) is not None:
            raise ValueError("table name contains illegal character.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if name not in self._ps_table_name_list:
            raise ValueError("this table has not yet initialized.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        table_id = self._table_name_to_id.get(name)
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
                                                       file_type="bin",
                                                       table_name=[name])
            return tf.group([embedding_table_export])

    def save_incremental_embeddings(self, path: str):
        """ Operator for save incremental values in all embedding tables. """
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        env_dist = os.environ
        rank_id_from_env = env_dist.get("RANK_ID")
        if rank_id_from_env != "0":
            raise ValueError("Device must be rank_id 0.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            for table_id in self._ps_table_id_list:
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
                                                       file_type="bin",
                                                       table_name=self._ps_table_name_list)
            return tf.group([embedding_table_export])

    def restore_incremental_embedding(self, name: str, path: str):
        if path is None or name is None:
            raise ValueError("table name, embedding table path can not be None.")
        if not isinstance(name, str):
            raise TypeError("embedding table name must be string.")
        regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
        if regex.search(name) is not None:
            raise ValueError("table name contains illegal character.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if name not in self._ps_table_name_list:
            raise ValueError("this table has not yet initialized.")
        table_id = self._table_name_to_id.get(name)
        with specified_ps_engine_scope():
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ops.convert_to_tensor(-1),
                                                       file_path=ops.convert_to_tensor(path),
                                                       table_id=ops.convert_to_tensor([table_id]),
                                                       embedding_dim=[self._table_to_embedding_dim.get(table_id)],
                                                       value_total_len=[self._table_to_embedding_dim.get(table_id)],
                                                       only_var_flag=True,
                                                       file_type="bin",
                                                       table_name=[name])
            return tf.group([embedding_table_import])

    def restore_incremental_embeddings(self, path: str):
        if path is None:
            raise ValueError("embedding table path can not be None.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        with specified_ps_engine_scope():
            table_id_list = []
            embedding_dim_list = []
            for table_id in self._ps_table_id_list:
                table_id_list.append(table_id)
                embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ops.convert_to_tensor(-1),
                                                       file_path=ops.convert_to_tensor(path),
                                                       table_id=ops.convert_to_tensor(table_id_list),
                                                       embedding_dim=embedding_dim_list,
                                                       value_total_len=embedding_dim_list,
                                                       only_var_flag=True,
                                                       file_type="bin",
                                                       table_name=self._ps_table_name_list)
            return tf.group([embedding_table_import])

    # old version
    def save_embedding_v1(self, path: str, table_id: int):
        """ Operator for save values in table_id embedding table. """
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if table_id not in self._ps_table_id_list:
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
                                                       file_type="bin",
                                                       table_name=[self._table_id_to_name.get(table_id)])
            return tf.group([embedding_table_export])

    def restore_embedding_v1(self, path: str, table_id: int):
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if table_id not in self._ps_table_id_list:
            raise ValueError("this table has not yet initialized.")
        with specified_ps_engine_scope():
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ops.convert_to_tensor(-1),
                                                       file_path=ops.convert_to_tensor(path),
                                                       table_id=ops.convert_to_tensor([table_id]),
                                                       embedding_dim=[self._table_to_embedding_dim.get(table_id)],
                                                       value_total_len=[self._table_to_embedding_dim.get(table_id)],
                                                       only_var_flag=True,
                                                       file_type="bin",
                                                       table_name=[self._table_id_to_name.get(table_id)])
            return tf.group([embedding_table_import])

    def save_checkpoint_v1(self, path: str, table_id: int):
        """ Operator for save values and optimizer params in table_id embedding table. """
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if table_id not in self._ps_table_id_list:
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
                                                                        (self._table_to_slot_var_num.get(table_id)
                                                                         + 1)],
                                                       export_mode="all",
                                                       only_var_flag=False,
                                                       file_type="bin",
                                                       table_name=[self._table_id_to_name.get(table_id)])
            with tf.control_dependencies([embedding_table_export]):
                embedding_compute_var_export = \
                    gen_npu_cpu_ops.embedding_compute_var_export(file_path=file_path_tensor,
                                                                 ps_id=ps_id_tensor,
                                                                 table_id=table_id_tensor,
                                                                 table_name=[self._table_id_to_name.get(table_id)])
                return tf.group([embedding_compute_var_export])

    def restore_checkpoint_v1(self, path: str, table_id: int):
        """ Operator for restore values and optimizer params in table_id embedding table. """
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if table_id not in self._ps_table_id_list:
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
                                                                        (self._table_to_slot_var_num.get(table_id)
                                                                         + 1)],
                                                       only_var_flag=False,
                                                       file_type="bin",
                                                       table_name=[self._table_id_to_name.get(table_id)])
            with tf.control_dependencies([embedding_table_import]):
                embedding_compute_var_import = \
                    gen_npu_cpu_ops.embedding_compute_var_import(file_path=file_path_tensor,
                                                                 ps_id=ps_id_tensor,
                                                                 table_id=table_id_tensor,
                                                                 table_name=[self._table_id_to_name.get(table_id)])
                return tf.group([embedding_compute_var_import])

    def save_incremental_embedding_v1(self, path: str, table_id: int):
        """ Operator for save incremental values in table_id embedding table. """
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if table_id not in self._ps_table_id_list:
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
                                                       file_type="bin",
                                                       table_name=[self._table_id_to_name.get(table_id)])
            return tf.group([embedding_table_export])

    def restore_incremental_embedding_v1(self, path: str, table_id: int):
        if path is None or table_id is None:
            raise ValueError("table_id, embedding table path can not be None.")
        if path[-1] == '/':
            raise ValueError("path format is wrong, please check.")
        if not self._init_table_flag:
            raise ValueError("Not any table has been initialized.")
        if table_id not in self._ps_table_id_list:
            raise ValueError("this table has not yet initialized.")
        with specified_ps_engine_scope():
            embedding_table_import = \
                gen_npu_cpu_ops.embedding_table_import(ps_id=ops.convert_to_tensor(-1),
                                                       file_path=ops.convert_to_tensor(path),
                                                       table_id=ops.convert_to_tensor([table_id]),
                                                       embedding_dim=[self._table_to_embedding_dim.get(table_id)],
                                                       value_total_len=[self._table_to_embedding_dim.get(table_id)],
                                                       only_var_flag=True,
                                                       file_type="bin",
                                                       table_name=[self._table_id_to_name.get(table_id)])
            return tf.group([embedding_table_import])

    def _init_hashmap_and_table_import(self, bucket_size, table_id, embedding_dim, ev_option):
        if (ev_option is not None) and (ev_option.filter_option is not None):
            filter_mode = "counter"
            self._table_to_counter_filter[table_id] = ev_option.filter_option
            self._table_use_counter_filter[table_id] = 1
        else:
            filter_mode = "no_filter"
            self._table_use_counter_filter[table_id] = 0
        if isinstance(self._table_to_optimizer.get(table_id), embedding_optimizer.AdagradOptimizer):
            self._ps_table_id_to_optimizer_mode[table_id] = "adagrad"
            self._ps_table_id_to_optimizer_params[table_id].append(
                self._table_to_optimizer.get(table_id)._initial_accumulator_value
            )
        if isinstance(self._table_to_optimizer.get(table_id), embedding_optimizer.AdamOptimizer):
            self._ps_table_id_to_optimizer_mode[table_id] = "adam"
            self._ps_table_id_to_optimizer_params[table_id].append(0)
        if isinstance(self._table_to_optimizer.get(table_id), embedding_optimizer.AdamWOptimizer):
            self._ps_table_id_to_optimizer_mode[table_id] = "adamw"
            self._ps_table_id_to_optimizer_params[table_id].append(0)

        with tf.control_dependencies([self._init_partition_maps.get(table_id)]):
            if self._train_mode:
                if self._train_level:
                    self._init_embedding_hash_maps[table_id] = \
                        gen_npu_cpu_ops.init_embedding_hashmap(table_id=ops.convert_to_tensor(table_id),
                                                               bucket_size=bucket_size,
                                                               value_total_len=embedding_dim *
                                                               (self._table_to_slot_var_num.get(table_id) + 1),
                                                               embedding_dim=embedding_dim,
                                                               initializer_mode=
                                                               self._table_id_to_initializer.get(table_id)
                                                               .initializer_mode,
                                                               constant_value=
                                                               self._table_id_to_initializer.get(table_id).
                                                               constant_value,
                                                               min=self._table_id_to_initializer.get(table_id).min,
                                                               max=self._table_id_to_initializer.get(table_id).max,
                                                               mu=self._table_id_to_initializer.get(table_id).mu,
                                                               sigma=self._table_id_to_initializer.get(table_id).sigma,
                                                               seed=self._table_id_to_initializer.get(table_id).seed,
                                                               seed2=self._table_id_to_initializer.get(table_id).seed,
                                                               filter_mode=filter_mode,
                                                               optimizer_mode=
                                                               self._ps_table_id_to_optimizer_mode.get(table_id),
                                                               optimizer_params=
                                                               self._ps_table_id_to_optimizer_params.get(table_id))
                else:
                    self._init_embedding_hash_maps[table_id] = \
                        gen_npu_cpu_ops.init_embedding_hashmap(table_id=ops.convert_to_tensor(table_id),
                                                               bucket_size=bucket_size,
                                                               value_total_len=embedding_dim *
                                                               (self._table_to_slot_var_num.get(table_id) + 1),
                                                               embedding_dim=embedding_dim,
                                                               initializer_mode=None, constant_value=None,
                                                               min=None, max=None, mu=None, sigma=None,
                                                               seed=None, seed2=None, filter_mode=filter_mode,
                                                               optimizer_mode=
                                                               self._ps_table_id_to_optimizer_mode.get(table_id),
                                                               optimizer_params=
                                                               self._ps_table_id_to_optimizer_params.get(table_id))
            else:
                self._init_embedding_hash_maps[table_id] = \
                    gen_npu_cpu_ops.init_embedding_hashmap(table_id=ops.convert_to_tensor(table_id),
                                                           bucket_size=bucket_size,
                                                           value_total_len=embedding_dim,
                                                           embedding_dim=embedding_dim,
                                                           initializer_mode=None, constant_value=None,
                                                           min=None, max=None, mu=None, sigma=None,
                                                           seed=None, seed2=None, filter_mode=filter_mode,
                                                           optimizer_mode=
                                                           self._ps_table_id_to_optimizer_mode.get(table_id),
                                                           optimizer_params=
                                                           self._ps_table_id_to_optimizer_params.get(table_id))
        self._init_table_flag = True
        self._init_table_flag = True
        if self._train_mode:
            return tf.group(
                [tf.initializers.variables(self._optimizer.variables()), self._init_embedding_hash_maps.get(table_id)])
        else:
            return tf.group([self._init_embedding_hash_maps.get(table_id)])
