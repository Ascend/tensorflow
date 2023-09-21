#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2021. Huawei Technologies Co., Ltd. All rights reserved.
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

"""Construct NPU configurations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from enum import Enum
import json
import os
from inspect import signature
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.distribute.experimental import ParameterServerStrategy
from tensorflow.contrib.distribute import DistributeConfig
from tensorflow.python.training import server_lib
from npu_bridge.estimator.npu import util


class NPURunConfig(run_config_lib.RunConfig):
    """RunConfig with NPU support."""

    def __init__(self,
                 iterations_per_loop=1,
                 profiling_config=None,
                 model_dir=None,
                 tf_random_seed=None,
                 save_summary_steps=0,
                 save_checkpoints_steps=None,
                 save_checkpoints_secs=None,
                 session_config=None,
                 keep_checkpoint_max=5,
                 keep_checkpoint_every_n_hours=10000,
                 log_step_count_steps=100,
                 distribute=None,
                 enable_data_pre_proc=True,
                 precision_mode=None,
                 enable_reduce_precision=False,
                 variable_format_optimize=None,
                 mix_compile_mode=False,
                 hcom_parallel=True,
                 graph_memory_max_size=None,
                 variable_memory_max_size=None,
                 dump_config=None,
                 stream_max_parallel_num=None,
                 is_tailing_optimization=False,
                 horovod_mode=False,
                 graph_run_mode=1,
                 op_debug_level=None,
                 enable_scope_fusion_passes=None,
                 enable_exception_dump=0,
                 op_select_implmode=None,
                 optypelist_for_implmode=None,
                 dynamic_input_config=None,
                 aoe_mode=None,
                 work_path=None,
                 buffer_optimize="l2_optimize",
                 enable_small_channel=0,
                 fusion_switch_file=None,
                 enable_compress_weight=False,
                 compress_weight_conf=None,
                 op_compiler_cache_mode=None,
                 op_compiler_cache_dir=None,
                 debug_dir=None,
                 hcom_multi_mode=False,
                 dynamic_input=None,
                 dynamic_graph_execute_mode="dynamic_execute",
                 dynamic_inputs_shape_range=None,
                 train_distribute=None,
                 eval_distribute=None,
                 local_rank_id=None,
                 local_device_list=None,
                 session_device_id=None,
                 distribute_config=None,
                 modify_mixlist=None,
                 op_precision_mode=None,
                 device_type="default_device_type",
                 soc_config=None,
                 hccl_timeout=None,
                 op_wait_timeout=None,
                 op_execute_timeout=None,
                 HCCL_algorithm=None,
                 customize_dtypes=None,
                 op_debug_config=None,
                 memory_config=None,
                 experimental_config=None,
                 topo_sorting_mode=None,
                 aoe_config_file=None,
                 insert_op_file=None,
                 stream_sync_timeout=-1,
                 event_sync_timeout=-1,
                 external_weight=False,
                 es_cluster_config=None,
                 deterministic=0,
                 frozen_variable=False,
                 variable_placement="Device",
                 jit_compile="auto",
                 precision_mode_v2=None
                 ):
        """
        Constructs a NPUConfig.

        Args:
        iterations_per_loop: This is the number of train steps running in NPU
            system before returning to CPU host for each `Session.run`. This means
            global step is increased `iterations_per_loop` times in one `Session.run`.
            It is recommended to be set as number of global steps for next checkpoint.
        profiling_config: The profiling configuration.
        model_dir: Reference tensorflow tf.estimator.RunConfig model_dir.
        tf_random_seed: Reference tensorflow tf.estimator.RunConfig tf_random_seed.
        save_summary_steps: Reference tensorflow tf.estimator.RunConfig save_summary_steps.
        save_checkpoints_steps: Reference tensorflow tf.estimator.RunConfig save_checkpoints_steps.
        save_checkpoints_secs: Reference tensorflow tf.estimator.RunConfig save_checkpoints_secs.
        session_config: Reference tensorflow tf.estimator.RunConfig session_config.
        keep_checkpoint_max: Reference tensorflow tf.estimator.RunConfig keep_checkpoint_max.
        keep_checkpoint_every_n_hours: Reference tensorflow tf.estimator.RunConfig keep_checkpoint_every_n_hours.
        log_step_count_steps: Reference tensorflow tf.estimator.RunConfig log_step_count_steps.
        enabel_data_pre_proc: This is the switch of data preprocess.
        precision_mode: default is: ''.
        variable_format_optimize: enable or disable variable format optimize while graph
            engineer optimize process.
        mix_compile_mode: This is the swith of mix_compile_mode. When the value is
            False, all graphs run on device. Otherwise, some graphs run on host.
        hcom_parallel: This is the switch of hcom parallel. When the value is True,
            hcom will execute with parallel mode. Otherwise, hcom will execute with
            serialize mode.
        graph_memory_max_size: The max size of ge graph memory size.
        variable_memory_max_size: The max size of ge variable memory size.
        dump_config: The dump configuration.
        stream_max_parallel_num: Specify the degree of parallelism of the AICPU / AICORE engine
                                 to achieve parallel execution between AICPU / AICORE operators.
        op_select_implmode: Selecting whether the operator is implemented with high_precision
                            or high_performance or high_precision_for_all or high_performance_for_all.
        optypelist_for_implmode: Operator list.
        dynamic_input_config:Dynamic dims configuration
        aoe_mode: Optimization Task Type."1": model tune; "2": optune;
                     "3": model tune & optune; "4": gradient split tune.
        work_path: Stores temporary files generated during optimization, default is current path.
        buffer_optimize: Whether to enable buffer optimization.
        enable_small_channel: Whether to enable small channel optimization.
        deterministic: Whether to enable deterministic calculation.
        fusion_switch_file: Fusion switch configuration file path.
        enable_compress_weight: Whether to enable global weight compression.
        compress_weight_conf:Path and file name of the node list configuration file to be compressed.
        dynamic_input:Whether Input is dynamic.
        dynamic_graph_execute_mode:Dynamic graph execute mode. lazy_recompile or dynamic_execute
        dynamic_inputs_shape_range:Inputs shape range.
        local_rank_id: Local sequence number of the device in a group.
        local_device_list: Available devices.
        distribute_config: Specify the NCA configuration file path
        modify_mixlist: Set the path of operator mixed precision configuration file.
        op_precision_mode: Set the path of operator precision mode configuration file (.ini)
        experimental_config: The experimental configuration.
        topo_sorting_mode: Provides an interface for users to customize topology sorting.
        external_weight: Whether convert const to fileconstant and save weight to file.
        es_cluster_config: esClusterConfig from user input in embedding service.
        frozen_variable: Whether folding constant variables
        variable_placement: Process variable on host or device
        jit_compile: Whether enable jit compile
        precision_mode_v2: default is: ''.
        """

        # Check iterations_per_loop.
        util.check_positive_integer(iterations_per_loop, "iterations_per_loop")
        if not isinstance(mix_compile_mode, bool):
            raise TypeError('"mix_compile_mode" type must be bool')
        if mix_compile_mode is True and iterations_per_loop != 1:
            raise ValueError(
                '"iterations_per_loop" must be 1 with "mix_compile_mode" is True')
        if not isinstance(jit_compile, str):
            raise TypeError('Expected str, got ' + type(jit_compile).__name__)
        tf_config = json.loads(os.environ.get(run_config_lib._TF_CONFIG_ENV, '{}'))
        tmp_cluster_spec = server_lib.ClusterSpec(tf_config.get(run_config_lib._CLUSTER_KEY, {}))
        if ((tmp_cluster_spec and not isinstance(distribute, ParameterServerStrategy)) or
                (not tmp_cluster_spec and isinstance(distribute, ParameterServerStrategy))):
            raise ValueError('"cluster" and "distribute" must all be set in ps mode')
        if tmp_cluster_spec and mix_compile_mode is False:
            raise ValueError(
                '"mix_compile_mode" can only be True with "cluster" is set')
        self.iterations_per_loop = iterations_per_loop
        self.mix_compile_mode = mix_compile_mode
        self.enable_data_pre_proc = enable_data_pre_proc
        self.is_tailing_optimization = is_tailing_optimization
        save_checkpoints_steps = self._get_save_checkpoints_steps(save_checkpoints_secs,
                                                                  save_checkpoints_steps)
        self._profiling_config = profiling_config

        # mix precision configuration
        self._precision_mode = precision_mode
        self._precision_mode_v2 = precision_mode_v2
        self._enable_reduce_precision = enable_reduce_precision
        self.variable_format_optimize = variable_format_optimize
        self._hcom_parallel = hcom_parallel
        self._graph_memory_max_size = graph_memory_max_size
        self._variable_memory_max_size = variable_memory_max_size

        self._dump_config = self._get_dump_config(dump_config)
        self._stream_max_parallel_num = stream_max_parallel_num

        self.horovod_mode = self._get_horovod_mode(horovod_mode)
        util.check_nonnegative_integer(graph_run_mode, "graph_run_mode")
        self.graph_run_mode = self._get_graph_run_mode(graph_run_mode)
        self.op_debug_level = op_debug_level
        self.enable_scope_fusion_passes = enable_scope_fusion_passes
        experimental_distribute = self._get_experimental_distribute(tmp_cluster_spec, distribute)
        util.check_nonnegative_integer(enable_exception_dump, "enable_exception_dump")
        self.enable_exception_dump = enable_exception_dump
        self._op_select_implmode = op_select_implmode
        self._optypelist_for_implmode = optypelist_for_implmode

        self._dynamic_input_config = self._get_dynamic_input_config(dynamic_input_config)
        self._aoe_mode = aoe_mode
        self._work_path = work_path
        self._buffer_optimize = buffer_optimize
        self._enable_small_channel = enable_small_channel
        self._deterministic = deterministic
        self._fusion_switch_file = fusion_switch_file
        self._enable_compress_weight = enable_compress_weight
        self._frozen_variable = frozen_variable
        self._variable_placement = variable_placement
        self._compress_weight_conf = compress_weight_conf
        self._op_compiler_cache_mode = op_compiler_cache_mode
        self._op_compiler_cache_dir = op_compiler_cache_dir
        self._debug_dir = debug_dir
        self._hcom_multi_mode = hcom_multi_mode
        self._dynamic_input = dynamic_input
        self._dynamic_graph_execute_mode = dynamic_graph_execute_mode
        self._dynamic_inputs_shape_range = dynamic_inputs_shape_range
        self._local_rank_id = local_rank_id
        self._local_device_list = local_device_list
        self._session_device_id = session_device_id
        self._distribute_config = distribute_config
        self._modify_mixlist = modify_mixlist
        self._op_precision_mode = op_precision_mode
        self._device_type = device_type
        self._soc_config = soc_config
        self._hccl_timeout = hccl_timeout
        self._op_wait_timeout = op_wait_timeout
        self._op_execute_timeout = op_execute_timeout
        self._HCCL_algorithm = HCCL_algorithm
        self._customize_dtypes = customize_dtypes
        self._op_debug_config = op_debug_config
        self._memory_config = memory_config
        self._experimental_config = self._get_experimental_config(experimental_config)
        self.topo_sorting_mode = topo_sorting_mode
        self.aoe_config_file = aoe_config_file
        self.insert_op_file = insert_op_file
        self.stream_sync_timeout = stream_sync_timeout
        self.event_sync_timeout = event_sync_timeout
        self._external_weight = external_weight
        self.es_cluster_config = es_cluster_config
        self._jit_compile = jit_compile

        super(NPURunConfig, self).__init__(
            model_dir=model_dir,
            tf_random_seed=tf_random_seed,
            save_summary_steps=save_summary_steps,
            save_checkpoints_steps=save_checkpoints_steps,
            save_checkpoints_secs=save_checkpoints_secs,
            session_config=session_config,
            keep_checkpoint_max=keep_checkpoint_max,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
            log_step_count_steps=log_step_count_steps,
            experimental_distribute=experimental_distribute,
            train_distribute=train_distribute,
            eval_distribute=eval_distribute)

    def _get_save_checkpoints_steps(self, save_checkpoints_secs, save_checkpoints_steps):
        if save_checkpoints_secs is None and save_checkpoints_steps is None:
            return 100
        return save_checkpoints_steps

    def _get_dump_config(self, dump_config):
        if dump_config is not None and not isinstance(dump_config, DumpConfig):
            raise ValueError(
                '`dump_config` must be provided with type `DumpConfig`')
        return dump_config

    def _get_experimental_config(self, experimental_config):
        if experimental_config is not None and not isinstance(experimental_config, ExperimentalConfig):
            raise ValueError(
                '`experimental_config` must be provided with type `ExperimentalConfig`')
        return experimental_config

    def _get_horovod_mode(self, horovod_mode):
        if not isinstance(horovod_mode, bool):
            raise ValueError('"horovod_mode" type must be bool')
        return horovod_mode

    def _get_graph_run_mode(self, graph_run_mode):
        if graph_run_mode > 1:
            raise ValueError('"graph_run_mode" value must be 0 or 1')
        return graph_run_mode

    def _get_experimental_distribute(self, tmp_cluster_spec, distribute):
        experimental_distribute = None
        if tmp_cluster_spec and isinstance(distribute, ParameterServerStrategy):
            experimental_distribute = DistributeConfig(distribute, distribute, None)
        return experimental_distribute

    def _get_dynamic_input_config(self, dynamic_input_config):
        if dynamic_input_config is not None and not isinstance(dynamic_input_config, DynamicInputConfig):
            raise ValueError('dynamic_input_config must be provided with type DynamicInputConfig')
        return dynamic_input_config


class ProfilingConfig():
    """Profiling config with NPU support."""

    def __init__(self,
                 enable_profiling=False,
                 profiling_options=None):
        """
        Constructs a ProfilingConfig.
        Args:
            enable_profiling: Enable profiling, default is False.
            profiling_options: Profiling options, a string include all profiling options.
        """

        self._enable_profiling = enable_profiling
        self._profiling_options = profiling_options


class MemoryConfig():
    """Memory config with NPU support."""

    def __init__(self,
                 atomic_clean_policy=0,
                 static_memory_policy=0,
                 memory_optimization_policy=None):
        """
        Constructs a MemoryConfig.
        Args:
            atomic_clean_policy: atomic_clean_policy, default is 0.
            static_memory_policy: static_memory_policy.
            memory_optimization_policy: memory_optimization_policy.
        """

        self._atomic_clean_policy = atomic_clean_policy
        self._static_memory_policy = static_memory_policy
        self._memory_optimization_policy = memory_optimization_policy



class DumpConfig():
    """Dump Config with NPU support."""

    def __init__(self,
                 enable_dump=False,
                 dump_path=None,
                 dump_step=None,
                 dump_mode="output",
                 enable_dump_debug=False,
                 dump_debug_mode="all",
                 dump_data="tensor",
                 dump_layer=None):
        """
        Constructs a DumpConfig.

        Args:
            enable_dump: Enable dump, default is False.
            dump_path: The dump path.
            dump_step: Specify step dump data. eg."0|5|10".
            dump_mode: Specify dump Op input or output or both.
            enable_dump_debug: Enable dump debug, default is False.
            dump_debug_mode: Debug dump mode, only support three kinds of mode(aicore_overflow, atomic_overflow or all).
        """
        self._enable_dump = enable_dump
        self._dump_path = dump_path
        self._dump_step = dump_step
        self._dump_mode = dump_mode
        self._enable_dump_debug = enable_dump_debug
        self._dump_debug_mode = dump_debug_mode
        self.dump_data = dump_data
        self.dump_layer = dump_layer


def _check_graph_mem_optimize_config(param, arg_name):
    if param is not None and param not in ["auto", "manual"]:
        raise ValueError(f"{arg_name} should be in ['auto', 'manual']")


class GraphMemoryOptimizeConfig():

    def __init__(self, recompute=None, graph_slice=None):
        """
        Args: recompute: "auto" or "manual"
        """
        _check_graph_mem_optimize_config(recompute, "recompute")
        _check_graph_mem_optimize_config(graph_slice, "graph_slice")
        self._recompute = recompute
        self._graph_slice = graph_slice


class ExperimentalConfig():
    """Experimental Config with NPU support."""

    def __init__(self,
                 logical_device_cluster_deploy_mode="LB",
                 logical_device_id=None,
                 model_deploy_mode=None,
                 model_deploy_devicelist=None,
                 graph_memory_optimize_config=None,
                 enable_graph_parallel=None,
                 graph_parallel_option_path=None,
                 graph_compiler_cache_dir=None,
                 resource_config_path=None,
                 accelerate_train_mode=None):
        """
        Constructs a ExperimentalConfig.

        Args:
            logical_device_cluster_deploy_mode: Specify the helper model deployment mode,
                the default is LB (load balance), or it can be specified as Single.
            logical_device_id: When logical_device_cluster_deploy_mode is specified as SINGLE mode,
                the specified model with logical_device_id is deployed on a device of 2PG.
        """
        self._logical_device_cluster_deploy_mode = logical_device_cluster_deploy_mode
        self._logical_device_id = logical_device_id
        self._model_deploy_mode = model_deploy_mode
        self._model_deploy_devicelist = model_deploy_devicelist
        if graph_memory_optimize_config is not None and not isinstance(graph_memory_optimize_config,
                                                                       GraphMemoryOptimizeConfig):
            raise ValueError("graph_memory_optimize_config type must be GraphMemoryOptimizeConfig")
        self._graph_memory_optimize_config = graph_memory_optimize_config
        self._enable_graph_parallel = enable_graph_parallel
        self._graph_parallel_option_path = graph_parallel_option_path
        self._graph_compiler_cache_dir = graph_compiler_cache_dir
        self._resource_config_path = resource_config_path
        self._accelerate_train_mode = accelerate_train_mode


class NpuExecutePlacement(Enum):
    """npu execute place option. """
    ALL = "all"
    CUBE = "cube"
    VECTOR = "vector"
    TAISHAN = "taishan"
    DVPP = "dvpp"
    HOST = "host"


class DynamicInputConfig():
    """dynamic dims and input shape config with npu support"""

    def __init__(self, input_shape, dynamic_dims, dynamic_node_type):
        """
        Constructs a DynamicInputConfig.

        Args:
            input_shape: the network's inputs shapes.
            dynamic_dims: This parameter corresponds to input_shape.
                          The dim value in dims corresponds to the parameter "-1" in input_shape.
            dynamic_node_type: Dataset or placeholder is dynamic input. type: 0 or 1.
        """
        self._input_shape = input_shape
        self._dynamic_dims = dynamic_dims
        self._dynamic_node_type = dynamic_node_type


def set_npu_default_config(func, args, kwargs):
    config_index = None
    for index, param in enumerate(signature(func).parameters):
        if param == "config":
            config_index = index - 1  # class func first arg is self
    tf_config = None
    if kwargs.get("config") is not None:
        tf_config = kwargs["config"]
    elif len(args) > config_index:
        tf_config = args[config_index]
    else:
        tf_config = config_pb2.ConfigProto()

    npu_optimizer = None
    for custom_optimizer in tf_config.graph_options.rewrite_options.custom_optimizers:
        if custom_optimizer.name == 'NpuOptimizer':
            npu_optimizer = custom_optimizer
            break
    if not npu_optimizer:
        npu_optimizer = tf_config.graph_options.rewrite_options.custom_optimizers.add()
        npu_optimizer.name = 'NpuOptimizer'
        tf_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    tf_config.allow_soft_placement = True
    tf_config.log_device_placement = False
    tf_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    tf_config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
    kwargs["config"] = tf_config
    if len(args) > config_index:
        args = args[:config_index] + args[config_index + 1:]
    return args, kwargs


def decorator_tf_session(func):
    def class_func_wrapper(self, *args, **kwargs):
        args, kwargs = set_npu_default_config(func, args, kwargs)
        return func(self, *args, **kwargs)

    return class_func_wrapper
