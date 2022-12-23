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

"""Functions for NPU estimator"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import collections
import json
import random
import string
import six
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from npu_bridge.estimator.npu import util as util_lib

from npu_bridge.estimator.npu import util
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_hook import NPUBroadcastGlobalVariablesHook
from npu_bridge.estimator.npu.npu_hook import NPUCheckpointSaverHook
from npu_bridge.estimator.npu.npu_hook import SetIterationsVarHook
from npu_bridge.estimator.npu.npu_hook import NPULogOutfeedSessionHook
from npu_bridge.estimator.npu.npu_hook import NPUInfeedOutfeedSessionHook
from npu_bridge.estimator.npu.npu_common import NPUBasics
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_saver import NPUSaver


def no_check_override():
    """Without checking override"""
    class _Manager:
        def __init__(self):
            self.__orign = None

        def __enter__(self):
            self.__orign = estimator_lib.Estimator._assert_members_are_not_overridden
            estimator_lib.Estimator._assert_members_are_not_overridden = lambda x: None

        def __exit__(self, exc_type, exc_val, exc_tb):
            estimator_lib.Estimator._assert_members_are_not_overridden = self.__orign

    return _Manager()


def _wrap_computation_in_while_loop(iterations_per_loop_var, op_fn):
    def computation(i):
        with ops.control_dependencies([op_fn]):
            return i + 1

    iterations = array_ops.identity(iterations_per_loop_var)
    return control_flow_ops.while_loop(
        lambda i: i < iterations,
        computation, [constant_op.constant(0)],
        parallel_iterations=1)


class _OutfeedHostCall:
    def __init__(self, channel_name):
        self._channel_name = str(channel_name)
        self._names = []
        # All of these are dictionaries of lists keyed on the name.
        self._host_fns = {}
        self._tensor_keys = collections.defaultdict(list)
        self._tensors = collections.defaultdict(list)
        self._tensor_dtypes = collections.defaultdict(list)
        self._tensor_shapes = collections.defaultdict(list)

    @staticmethod
    def validate(host_calls):
        """Validates the `eval_metrics` and `host_call` in `NPUEstimatorSpec`."""

        for name, host_call in host_calls.items():
            if not isinstance(host_call, (tuple, list)):
                raise ValueError('{} should be tuple or list'.format(name))
            if len(host_call) != 2:
                raise ValueError('{} should have two elements.'.format(name))
            if not callable(host_call[0]):
                raise TypeError('{}[0] should be callable.'.format(name))
            if not isinstance(host_call[1], (tuple, list, dict)):
                raise ValueError('{}[1] should be tuple or list, or dict.'.format(name))

            if isinstance(host_call[1], (tuple, list)):
                fullargspec = tf_inspect.getfullargspec(host_call[0])
                fn_args = function_utils.fn_args(host_call[0])
                # wrapped_hostcall_with_global_step uses varargs, so we allow that.
                if fullargspec.varargs is None and len(host_call[1]) != len(fn_args):
                    raise RuntimeError(
                        'In NPUEstimatorSpec.{}, length of tensors {} does not match '
                        'method args of the function, which takes {}.'.format(
                            name, len(host_call[1]), len(fn_args)))

    def create_npu_hostcall(self):
        """Sends the tensors through outfeed and runs the host_fn on CPU.

            The tensors are concatenated along dimension 0 to form a global tensor
            across all shards. The concatenated function is passed to the host_fn and
            executed on the first host.

            Returns:
              A dictionary mapping name to the return type of the host_call by that
              name.

            Raises:
              RuntimeError: If outfeed tensor is scalar.
        """
        if not self._names:
            return {}

        ret = {}
        # For each i, dequeue_ops[i] is a list containing the tensors from all
        # shards. This list is concatenated later.
        dequeue_ops = []
        tensor_dtypes = []
        tensor_shapes = []
        for name in self._names:
            for _ in self._tensors[name]:
                dequeue_ops.append([])
            for dtype in self._tensor_dtypes[name]:
                tensor_dtypes.append(dtype)
            for shape in self._tensor_shapes[name]:
                tensor_shapes.append(shape)

        outfeed_tensors = npu_ops.outfeed_dequeue_op(
            channel_name=self._channel_name,
            output_types=tensor_dtypes,
            output_shapes=tensor_shapes)

        # Deconstruct dequeue ops.
        outfeed_tensors_by_name = {}
        pos = 0
        for name in self._names:
            outfeed_tensors_by_name[name] = outfeed_tensors[pos:pos + len(self._tensors[name])]
            pos += len(self._tensors[name])

        for name in self._names:
            host_fn_tensors = outfeed_tensors_by_name[name]
            if self._tensor_keys[name] is not None:
                host_fn_tensors = dict(zip(self._tensor_keys[name], host_fn_tensors))
                try:
                    ret[name] = self._host_fns[name](**host_fn_tensors)
                except TypeError as e:
                    logging.warning(
                        'Exception while calling %s: %s. It is likely the tensors '
                        '(%s[1]) do not match the '
                        'function\'s arguments', name, e, name)
                    raise e
            else:
                ret[name] = self._host_fns[name](*host_fn_tensors)

        return ret

    def create_enqueue_op(self):
        """Create the op to enqueue the recorded host_calls.

        Returns:
          A list of enqueue ops, which is empty if there are no host calls.
        """
        if not self._names:
            return []

        tensors = []
        for name in self._names:
            tensors.extend(self._tensors[name])
        if len(tensors) == 0:
            return []
        return npu_ops.outfeed_enqueue_op(inputs=tensors, channel_name=self._channel_name)

    def record(self, host_calls):
        """Used to record host_calls"""
        for name, host_call in host_calls.items():
            host_fn, tensor_list_or_dict = host_call
            self._names.append(name)
            self._host_fns[name] = host_fn

            if isinstance(tensor_list_or_dict, dict):
                for (key, tensor) in six.iteritems(tensor_list_or_dict):
                    self._tensor_keys[name].append(key)
                    self._tensors[name].append(tensor)
                    self._tensor_dtypes[name].append(tensor.dtype)
                    self._tensor_shapes[name].append(tensor.shape)
            else:
                # List or tuple.
                self._tensor_keys[name] = None
                for tensor in tensor_list_or_dict:
                    self._tensors[name].append(tensor)
                    self._tensor_dtypes[name].append(tensor.dtype)
                    self._tensor_shapes[name].append(tensor.shape)


class NPUEstimatorSpec(model_fn_lib.EstimatorSpec):
    """Ops and objects returned from a `model_fn` and passed to an `NPUEstimator`.

    `NPUEstimatorSpec` fully defines the model to be run by an `Estimator`.
    """

    def __new__(cls,
                mode,
                predictions=None,
                loss=None,
                train_op=None,
                eval_metric_ops=None,
                export_outputs=None,
                training_chief_hooks=None,
                training_hooks=None,
                scaffold=None,
                evaluation_hooks=None,
                prediction_hooks=None,
                host_call=None):
        """
        Args:
            mode: Reference tensorflow tf.estimator.EstimatorSpec model_dir.
            predictions: Reference tensorflow tf.estimator.EstimatorSpec predictions.
            loss: Reference tensorflow tf.estimator.EstimatorSpec loss.
            train_op: Reference tensorflow tf.estimator.EstimatorSpec train_op.
            eval_metric_ops: Reference tensorflow tf.estimator.EstimatorSpec eval_metric_ops.
            export_outputs: Reference tensorflow tf.estimator.EstimatorSpec export_outputs.
            training_chief_hooks: Reference tensorflow tf.estimator.EstimatorSpec training_chief_hooks.
            training_hooks: Reference tensorflow tf.estimator.EstimatorSpec training_hooks.
            scaffold: Reference tensorflow tf.estimator.EstimatorSpec scaffold.
            evaluation_hooks: Reference tensorflow tf.estimator.EstimatorSpec evaluation_hooks.
            prediction_hooks: Reference tensorflow tf.estimator.EstimatorSpec prediction_hooks.
            host_call:  A tuple of `func`, or a list of `tensor` or `dict`.Get	255
                summary infomation, and send to host every step. Only used if mode	256
                is `ModeKeys.TRAIN` or  `ModeKeys.EVAL`.

        Returns:
            A validated `NPUEstimatorSpec` object.
        """
        host_calls = {}
        if host_call is not None:
            host_calls["host_call"] = host_call
        _OutfeedHostCall.validate(host_calls)
        spec = super(NPUEstimatorSpec, cls).__new__(
            cls,
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            export_outputs=export_outputs,
            training_chief_hooks=training_chief_hooks,
            training_hooks=training_hooks,
            scaffold=scaffold,
            evaluation_hooks=evaluation_hooks,
            prediction_hooks=prediction_hooks,
        )
        spec._host_call = host_call
        return spec


class NPUEstimator(estimator_lib.Estimator):
    """Estimator with NPU support.

    NPUEstimator handles many of the details of running on NPU devices, such as
    replicating inputs and models for each core, and returning to host
    periodically to run hooks.
    """

    def __init__(self,
                 model_fn=None,
                 model_dir=None,
                 config=None,
                 params=None,
                 job_start_file='',
                 warm_start_from=None
                 ):
        """Constructs an `NPUEstimator` instance.

        Args:
            model_fn: Model function as required by `Estimator` which returns
                EstimatorSpec. `training_hooks`, 'evaluation_hooks',
                and `prediction_hooks` must not capure any NPU Tensor inside the model_fn.
            config: An `NPURunConfig` configuration object. Cannot be `None`.
            params: An optional `dict` of hyper parameters that will be passed into
                `input_fn` and `model_fn`.  Keys are names of parameters, values are
                basic python types..
            job_start_file: The path of the job start file. Cannot be `None`.
            warm_start_from: Optional string filepath to a checkpoint or SavedModel to
               warm-start from, or a `tf.estimator.WarmStartSettings`
               object to fully configure warm-starting.  If the string
               filepath is provided instead of a`tf.estimator.WarmStartSettings`,
               then all variables are warm-started, and it is assumed that vocabularies
               and `tf.Tensor` names are unchanged.
         """
        logging.info("NPUEstimator init...")

        if config is None or not isinstance(config, NPURunConfig):
            raise ValueError(
                '`config` must be provided with type `NPUConfigs`')

        # Verifies the model_fn signature according to Estimator framework.
        estimator_lib._verify_model_fn_args(model_fn, params)  # pylint: disable=protected-access

        # Load the graph optimizers.
        config = self.__load_graph_optimizers(config)

        # Init npu system: get task and device info from configuration file.
        if not self.__load_job_info(job_start_file):
            raise ValueError('Load job info failed, '
                             'please check whether `JOB_ID` is set in environment variable')

        # Check modie dir in NPUEstimator and NPURunConfig
        model_dir = self.__check_model_dir(model_dir, config)

        # Wrap model_fn to adding npu sessionhooks.
        model_function = self.__augment_model_fn(model_fn, model_dir, config)

        # Get the checkpoint file.
        if not warm_start_from:
            restore_from = self.__job_info._local_checkpoint_dir
            # tf use restore_from variable, no need to check safety.
            if restore_from is None or restore_from == "":
                restore_from = os.getenv('RESTORE_FROM')
        else:
            restore_from = warm_start_from

        # Passing non-None params as wrapped model_fn use it.
        params = params or {}
        with no_check_override():
            super(NPUEstimator, self).__init__(
                model_fn=model_function,
                model_dir=model_dir,
                config=config,
                params=params,
                warm_start_from=restore_from)

    def __augment_model_fn(self, model_fn, model_dir, config):
        """Returns a new model_fn, which wraps the NPU support."""

        def _model_fn(features, labels, mode, params):
            """A Estimator `model_fn` for NPUEstimator."""
            model_fn_args = function_utils.fn_args(model_fn)
            kwargs = {}
            if 'labels' in model_fn_args:
                kwargs['labels'] = labels
            if 'mode' in model_fn_args:
                kwargs['mode'] = mode
            if 'params' in model_fn_args:
                kwargs['params'] = params
            if 'config' in model_fn_args:
                kwargs['config'] = config
            estimator_spec = model_fn(features=features, **kwargs)

            """
            add hooks:
                NPUInitHook: for all mode, NPUInitHook should be the first session hook
                NPUShutDownHook: for all mode, NPUShutDownHook should be the first session hook
                NPUBroadcastGlobalVariablesHook: train
                NPUCheckpointSaverHook:train
            """
            npu_hooks = []

            if mode == model_fn_lib.ModeKeys.TRAIN:
                if not isinstance(estimator_spec, NPUEstimatorSpec) and not isinstance(estimator_spec,
                                                                                       model_fn_lib.EstimatorSpec):
                    raise RuntimeError('estimator_spec used by NPU train must have type '
                                       '`NPUEstimatorSpec` or `EstimatorSpec`. Got {}'.format(type(estimator_spec)))
                # 1. NPUBroadcastGlobalVariablesHook
                rank_size = util_lib.get_ranksize()
                if rank_size is not None and rank_size.isdigit() and int(rank_size) > 1 and not config.horovod_mode:
                    npu_hooks.append(
                        NPUBroadcastGlobalVariablesHook(self.__device_info._root_rank, self.__device_info._index))

                # 2. NPUCheckpointSaverHook
                if config.save_checkpoints_steps or config.save_checkpoints_secs:
                    npu_hooks.append(NPUCheckpointSaverHook(
                        checkpoint_dir=model_dir,
                        save_secs=config.save_checkpoints_secs,
                        save_steps=config.save_checkpoints_steps,
                        saver=NPUSaver()))

                if isinstance(estimator_spec, NPUEstimatorSpec):
                    if estimator_spec._host_call is not None:
                        host_call = _OutfeedHostCall(mode)
                        host_call.record({"host_call": estimator_spec._host_call})
                        # add outfeed enqueue op
                        loss, train_op = estimator_spec.loss, estimator_spec.train_op
                        with ops.control_dependencies([train_op]):
                            host_call_outfeed_op = host_call.create_enqueue_op()
                            with ops.control_dependencies([host_call_outfeed_op]):
                                loss = array_ops.identity(loss)
                                estimator_spec = estimator_spec._replace(loss=loss)
                        # add outfeed dnqueue op
                        host_call_ops = host_call.create_npu_hostcall()
                        npu_hooks.append(NPUInfeedOutfeedSessionHook(host_call_ops, mode))
                    npu_hooks.append(NPULogOutfeedSessionHook(sys.stderr))

                # 3. set iterations per loop hook
                if config.iterations_per_loop > 1:
                    npu_hooks.append(SetIterationsVarHook(config.iterations_per_loop))
                    train_op = tf.group(estimator_spec.train_op, name="IterationOp")
                    estimator_spec = estimator_spec._replace(train_op=train_op)

                train_hooks = estimator_spec.training_hooks
                train_hooks = list(train_hooks or [])
                new_train_hooks = npu_hooks + train_hooks

                estimator_spec = estimator_spec._replace(training_hooks=tuple(new_train_hooks))

            elif mode == model_fn_lib.ModeKeys.EVAL:
                if not isinstance(estimator_spec, NPUEstimatorSpec) and not isinstance(estimator_spec,
                                                                                       model_fn_lib.EstimatorSpec):
                    raise RuntimeError('estimator_spec used by NPU evaluate must have type '
                                       '`NPUEstimatorSpec` or `EstimatorSpec`. Got {}'.format(type(estimator_spec)))
                if isinstance(estimator_spec, NPUEstimatorSpec):
                    if estimator_spec._host_call is not None:
                        host_call = _OutfeedHostCall(mode)
                        host_call.record({"host_call": estimator_spec._host_call})
                        # add outfeed enqueue op
                        loss, train_op = estimator_spec.loss, estimator_spec.train_op
                        with ops.control_dependencies([loss]):
                            host_call_outfeed_op = host_call.create_enqueue_op()
                            with ops.control_dependencies([host_call_outfeed_op]):
                                loss = array_ops.identity(loss)
                                estimator_spec = estimator_spec._replace(loss=loss)
                        # add outfeed dnqueue op
                        host_call_ops = host_call.create_npu_hostcall()
                        npu_hooks.append(NPUInfeedOutfeedSessionHook(host_call_ops, mode))
                    npu_hooks.append(NPULogOutfeedSessionHook(sys.stderr))
                if len(npu_hooks) > 0:
                    evaluation_hooks = estimator_spec.evaluation_hooks
                    evaluation_hooks = list(evaluation_hooks or [])
                    new_evaluation_hooks = npu_hooks + evaluation_hooks
                    estimator_spec = estimator_spec._replace(evaluation_hooks=tuple(new_evaluation_hooks))

            elif mode == model_fn_lib.ModeKeys.PREDICT:
                if len(npu_hooks) > 0:
                    prediction_hooks = estimator_spec.prediction_hooks
                    prediction_hooks = list(prediction_hooks or [])
                    new_prediction_hooks = npu_hooks + prediction_hooks

                    estimator_spec = estimator_spec._replace(prediction_hooks=tuple(new_prediction_hooks))
            return estimator_spec

        return _model_fn

    def __check_profiling_options(self, profiling_options=()):
        """Check profiling options .
        Args:
            profiling_options: Profiling options.
        Return:
            Valid options
        Raise:
            If profiling_options is null or option is not `training_trace` or `task_trace`, `op_trace`'.
        """

        error_mag = 'profiling options must be in `training_trace`, `task_trace` or `op_trace`'

        if len(profiling_options) == 0:
            raise ValueError(error_mag)

        profiling_types = ["training_trace", "task_trace", "op_trace"]
        for option in profiling_options:
            if option not in profiling_types:
                raise ValueError(error_mag)

        result = ":".join(profiling_options)
        return result

    def __load_session_device_id(self, config, custom_op):
        if config._session_device_id is not None:
            custom_op.parameter_map["session_device_id"].i = config._session_device_id

    def __load_modify_mixlist(self, config, custom_op):
        if config._modify_mixlist is not None:
            custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes(config._modify_mixlist)

    def __load_op_precision_mode(self, config, custom_op):
        if config._op_precision_mode is not None:
            custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes(config._op_precision_mode)

    def __load_profiling_options(self, config, custom_op):
        """Load profiling config ,and add to custom_optimizers
        Args:
            config: NPURunConfig.
            custom_op: Custom optimizers.
        """
        if config._profiling_config is not None:
            custom_op.parameter_map["profiling_mode"].b = config._profiling_config._enable_profiling
            if config._profiling_config._enable_profiling:
                if config._profiling_config._profiling_options is None:
                    config._profiling_config._profiling_options = ""
                custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
                    config._profiling_config._profiling_options)

    def __load_memory_config(self, config, custom_op):
        """Load memory config ,and add to custom_optimizers
        Args:
            config: NPURunConfig.
            custom_op: Custom optimizers.
        """
        if config._memory_config is not None:
            custom_op.parameter_map["atomic_clean_policy"].i = config._memory_config._atomic_clean_policy
            if config._memory_config._static_memory_policy is not None:
                custom_op.parameter_map["static_memory_policy"].i = config._memory_config._static_memory_policy
        

    def __load_mix_precision(self, config, custom_op):
        """Load mix precision config ,and add to custom_optimizers
        Args:
            config: NPURunConfig.
            custom_op: Custom optimizers.
        """
        if config._precision_mode is not None:
            custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(config._precision_mode)
        else:
            if config.graph_run_mode:
                custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_fp32_to_fp16")
            else:
                custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

        custom_op.parameter_map["enable_reduce_precision"].b = config._enable_reduce_precision

    def __load__variable_format_optimize(self, config, custom_op):
        """Load variable acceleration config ,and add to custom_optimizers
        Args:
            config: NPURunConfig.
            custom_op: Customer optimizers.
        """

        custom_op.parameter_map["variable_format_optimize"].b = config._variable_format_optimize

    def __load_dump_config(self, config, custom_op):
        """Load dump config ,and add to custom_optimizers
        Args:
            config: NPURunConfig.
            custom_op: Customer optimizers.
        """
        if config._dump_config is None:
            """
            there is no dump config in user's python script,
            then use the default dump configuration
            """
            custom_op.parameter_map["enable_dump"].b = False
            custom_op.parameter_map["enable_dump_debug"].b = False

        else:
            custom_op.parameter_map["enable_dump"].b = config._dump_config._enable_dump
            custom_op.parameter_map["enable_dump_debug"].b = config._dump_config._enable_dump_debug
            if config._dump_config._dump_path is not None:
                custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(config._dump_config._dump_path)
            if config._dump_config._dump_step is not None:
                custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(config._dump_config._dump_step)
            if config._dump_config._dump_mode is not None:
                custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes(config._dump_config._dump_mode)
            if config._dump_config._dump_mode is not None:
                custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes(config._dump_config._dump_debug_mode)
            if config._dump_config.dump_data is not None:
                custom_op.parameter_map["dump_data"].s = tf.compat.as_bytes(config._dump_config.dump_data)
            if config._dump_config.dump_layer is not None:
                custom_op.parameter_map["dump_layer"].s = tf.compat.as_bytes(config._dump_config.dump_layer)

    def __load_experimental_config(self, config, custom_op):
        """Load experimental config ,and add to custom_optimizers
        Args:
            config: NPURunConfig.
            custom_op: Customer optimizers.
        """
        if config._experimental_config is None:
            """
            there is no experimental config in user's python script,
            then use the default experimental configuration
            """
            custom_op.parameter_map["experimental_logical_device_cluster_deploy_mode"].s = tf.compat.as_bytes("LB")

        else:
            if config._experimental_config._logical_device_cluster_deploy_mode is not None:
                custom_op.parameter_map["experimental_logical_device_cluster_deploy_mode"].s = tf.compat.as_bytes(
                    config._experimental_config._logical_device_cluster_deploy_mode)
            if config._experimental_config._logical_device_id is not None:
                custom_op.parameter_map["experimental_logical_device_id"].s = tf.compat.as_bytes(
                    config._experimental_config._logical_device_id)
            if config._experimental_config._resource_config_path is not None:
                custom_op.parameter_map["resource_config_path"].s = tf.compat.as_bytes(
                    config._experimental_config._resource_config_path)

    def __load_stream_max_config(self, config, custom_op):
        """Load stream_max_parallel_num config ,and add to custom_optimizers
        Args:
            config: NPURunConfig.
            custom_op: Customer optimizers.
        """
        if config._stream_max_parallel_num is not None:
            custom_op.parameter_map["stream_max_parallel_num"].s = tf.compat.as_bytes(config._stream_max_parallel_num)

    def __load_ps_mode_config(self, custom_op):
        """Load stream_max_parallel_num config ,and add to custom_optimizers
        Args:
            custom_op: Customer optimizers.
        """
        config_info = json.loads(os.environ.get('TF_CONFIG') or '{}')

        # Set task_type and task_id if the TF_CONFIG environment variable is
        # present.  Otherwise, use the respective default (None / 0).
        task_env = config_info.get('task', {})
        task_type = task_env.get('type', None)
        task_index = task_env.get('index', 0)
        if task_type:
            custom_op.parameter_map["job"].s = tf.compat.as_bytes(task_type)
            custom_op.parameter_map["task_index"].i = int(task_index)
        else:
            custom_op.parameter_map["job"].s = tf.compat.as_bytes('localhost')
            custom_op.parameter_map["task_index"].i = 0

    def _load_op_performance_config(self, config, custom_op):
        """Load _load_op_performance_config ,and add to custom_optimizers
        Args:
            config: NPURunConfig.
            custom_op: Customer optimizers.
        """
        if config._op_select_implmode is not None:
            custom_op.parameter_map["op_select_implmode"].s = tf.compat.as_bytes(config._op_select_implmode)
        if config._optypelist_for_implmode is not None:
            custom_op.parameter_map["optypelist_for_implmode"].s = tf.compat.as_bytes(config._optypelist_for_implmode)

    def __load_dynamic_input_config(self, config, custom_op):
        """Load dynamic input config,and add to custom_optimizers
        Args:
            config: NPURunConfig.
            custom_op: Customer optimizers.
        """

        if (config._dynamic_input_config is not None and
                config._dynamic_input_config._input_shape is not None and
                config._dynamic_input_config._dynamic_dims is not None and
                config._dynamic_input_config._dynamic_node_type is not None):
            custom_op.parameter_map["input_shape"].s = tf.compat.as_bytes(config._dynamic_input_config._input_shape)
            custom_op.parameter_map["dynamic_dims"].s = tf.compat.as_bytes(config._dynamic_input_config._dynamic_dims)
            custom_op.parameter_map["dynamic_node_type"].i = config._dynamic_input_config._dynamic_node_type

    def __load_mstune_config(self, config, custom_op):
        """Load mstune config ,and add to custom_optimizers
        Args:
            config: NPURunConfig.
            custom_op: Customer optimizers.
        """
        if config._aoe_mode is not None:
            custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes(config._aoe_mode)
            if config._work_path is not None:
                custom_op.parameter_map["work_path"].s = tf.compat.as_bytes(config._work_path)
            else:
                custom_op.parameter_map["work_path"].s = tf.compat.as_bytes("./")
            if config._distribute_config is not None:
                custom_op.parameter_map["distribute_config"].s = tf.compat.as_bytes(config._distribute_config)
        if config.aoe_config_file is not None:
            custom_op.parameter_map["aoe_config_file"].s = tf.compat.as_bytes(config.aoe_config_file)

    def __load_graph_optimizers(self, config):
        """
        Change the session config and load the graph optimizers:
        GradFusionOptimizer and OMPartitionSubgraphsPass.
        """

        if config.session_config is None:
            config = config.replace(session_config=tf.ConfigProto())

        config.session_config.graph_options.rewrite_options.optimizers.extend(["pruning",
                                                                               "function",
                                                                               "constfold",
                                                                               "shape",
                                                                               "arithmetic",
                                                                               "loop",
                                                                               "dependency",
                                                                               "layout",
                                                                               "GradFusionOptimizer"])
        # config set
        custom_op = config.session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_data_pre_proc"].b = config.enable_data_pre_proc
        custom_op.parameter_map["mix_compile_mode"].b = config.mix_compile_mode
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["iterations_per_loop"].i = config.iterations_per_loop
        custom_op.parameter_map["is_tailing_optimization"].b = config.is_tailing_optimization
        custom_op.parameter_map["min_group_size"].b = 1
        custom_op.parameter_map["hcom_parallel"].b = config._hcom_parallel
        if config._graph_memory_max_size is not None:
            custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(config._graph_memory_max_size))
        if config._variable_memory_max_size is not None:
            custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(
                str(config._variable_memory_max_size))
        custom_op.parameter_map["graph_run_mode"].i = config.graph_run_mode
        custom_op.parameter_map["op_debug_level"].i = config.op_debug_level
        if config.enable_scope_fusion_passes is not None:
            custom_op.parameter_map["enable_scope_fusion_passes"].s = tf.compat.as_bytes(
                config.enable_scope_fusion_passes)
        custom_op.parameter_map["enable_exception_dump"].i = config.enable_exception_dump
        if config._buffer_optimize is not None:
            custom_op.parameter_map["buffer_optimize"].s = tf.compat.as_bytes(config._buffer_optimize)
        custom_op.parameter_map["enable_small_channel"].i = config._enable_small_channel
        if config._fusion_switch_file is not None:
            custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes(config._fusion_switch_file)
        custom_op.parameter_map["enable_compress_weight"].b = config._enable_compress_weight
        if config._compress_weight_conf is not None:
            custom_op.parameter_map["compress_weight_conf"].s = tf.compat.as_bytes(config._compress_weight_conf)
        if config._op_compiler_cache_mode is not None:
            custom_op.parameter_map["op_compiler_cache_mode"].s = tf.compat.as_bytes(config._op_compiler_cache_mode)
        if config._op_compiler_cache_dir is not None:
            custom_op.parameter_map["op_compiler_cache_dir"].s = tf.compat.as_bytes(config._op_compiler_cache_dir)
        if config._debug_dir is not None:
            custom_op.parameter_map["debug_dir"].s = tf.compat.as_bytes(config._debug_dir)
        custom_op.parameter_map["hcom_multi_mode"].b = config._hcom_multi_mode
        custom_op.parameter_map["dynamic_input"].b = config._dynamic_input
        custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes(config._dynamic_graph_execute_mode)
        if config._dynamic_inputs_shape_range is not None:
            custom_op.parameter_map["dynamic_inputs_shape_range"].s = tf.compat.as_bytes(
                config._dynamic_inputs_shape_range)
        if config._local_rank_id is not None:
            custom_op.parameter_map["local_rank_id"].i = config._local_rank_id
        if config._local_device_list is not None:
            custom_op.parameter_map["local_device_list"].s = tf.compat.as_bytes(config._local_device_list)
        custom_op.parameter_map["device_type"].s = tf.compat.as_bytes(config._device_type)
        if config._soc_config is not None:
            custom_op.parameter_map["soc_config"].s = tf.compat.as_bytes(config._soc_config)
        if config._hccl_timeout is not None:
            custom_op.parameter_map["hccl_timeout"].i = config._hccl_timeout
        if config._op_wait_timeout is not None:
            custom_op.parameter_map["op_wait_timeout"].i = config._op_wait_timeout
        if config._op_execute_timeout is not None:
            custom_op.parameter_map["op_execute_timeout"].i = config._op_execute_timeout
        if config._HCCL_algorithm is not None:
            custom_op.parameter_map["HCCL_algorithm"].s = tf.compat.as_bytes(config._HCCL_algorithm)
        if config._customize_dtypes is not None:
            custom_op.parameter_map["customize_dtypes"].s = tf.compat.as_bytes(config._customize_dtypes)
        if config._op_debug_config is not None:
            custom_op.parameter_map["op_debug_config"].s = tf.compat.as_bytes(config._op_debug_config)
        if config.topo_sorting_mode is not None:
            custom_op.parameter_map["topo_sorting_mode"].i = config.topo_sorting_mode
        if config.insert_op_file is not None:
            custom_op.parameter_map["insert_op_file"].s = config.insert_op_file
        custom_op.parameter_map["jit_compile"].b = config._jit_compile
        custom_op.parameter_map["external_weight"].b = config._external_weight

        self.__load_session_device_id(config, custom_op)
        self.__load_modify_mixlist(config, custom_op)
        self.__load_op_precision_mode(config, custom_op)

        # add profiling options to custom_op
        self.__load_profiling_options(config, custom_op)
        self.__load_memory_config(config, custom_op)

        # add mix precision to custom_op
        self.__load_mix_precision(config, custom_op)

        # add variable acceleration to custom_op
        self.__load__variable_format_optimize(config, custom_op)

        # add dump config to custom_op
        self.__load_dump_config(config, custom_op)

        # add stream_max_parallel to custom_op
        self.__load_stream_max_config(config, custom_op)

        self.__load_ps_mode_config(custom_op)

        self._load_op_performance_config(config, custom_op)

        # add dynamic_input_config to custom_op
        self.__load_dynamic_input_config(config, custom_op)

        self.__load_mstune_config(config, custom_op)

        # add experimental config to custom_op
        self.__load_experimental_config(config, custom_op)

        return config

    def __load_job_info(self, job_start_file):
        """Parse the file from the CSA."""
        # Read the job config file.
        basic = NPUBasics(job_start_file)
        if basic.jobinfo is None:
            return False

        # Get Device info from config file.
        self.__job_info = basic.jobinfo
        self.__device_info = basic.jobinfo._device_info
        return True

    def __check_model_dir(self, model_dir, config):
        """Check model dir. If model dir is None, create temp dir.

        Returns:
        Model dir.

        Raises:
        ValueError: If model_dir of NPUEstimator is different with model_dir of NPURunConfig.
        """
        if (model_dir is not None) and (config.model_dir is not None):
            if model_dir != config.model_dir:
                raise ValueError(
                    'model_dir are set both in NPUEstimator and NPURunConfig, but with '
                    "different values. In constructor: '{}', in NPURunConfig: "
                    "'{}' ".format(model_dir, config.model_dir))

        model_dir = model_dir or config.model_dir
        if model_dir is None:
            while True:
                model_dir = "".join(["model_dir_"] + random.sample(string.ascii_letters + string.digits, 10))
                cwd = os.getcwd()
                model_dir = os.path.join(cwd, model_dir)
                if not tf.io.gfile.exists(model_dir):
                    break
            logging.warning('Using temporary folder as model directory: %s', model_dir)
            tf.io.gfile.mkdir(model_dir)
        return model_dir
