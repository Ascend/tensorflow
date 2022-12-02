#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

"""Functions used for NPU device"""

import os
import atexit
import threading
import absl.logging as logging

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops import script_ops
from tensorflow.python.util import tf_contextlib

from npu_device.configs.npu_config import NpuConfig
from npu_device.configs.npu_run_context_option import NpuRunContextOptions

NPU = "/job:localhost/replica:0/task:0/device:NPU"

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _npu_device_backends
else:
    import _npu_device_backends


@tf_contextlib.contextmanager
def watch_op_register():
    try:
        _npu_device_backends.WatchOpRegister()
        yield
    finally:
        _npu_device_backends.StopWatchOpRegister()


with watch_op_register():
    gen_npu_ops = tf.load_op_library(os.path.join(os.path.dirname(__file__), "_npu_ops.so"))


def stupid_repeat(word, times):
    """Simple repeated"""
    return _npu_device_backends.StupidRepeat(word, times)


def set_npu_loop_size(loop_size):
    """Set loop size for NPU"""
    _npu_device_backends.SetNpuLoopSize(loop_size)


def set_device_sat_mode(mode):
    _npu_device_backends.SetDeviceSatMode(mode)


def get_device_sat_mode():
    return _npu_device_backends.GetDeviceSatMode()


_global_options = None
_global_options_lock = threading.Lock()


def global_options():
    """Set global options"""
    global _global_options
    if _global_options is None:
        with _global_options_lock:
            if _global_options is None:
                _global_options = NpuConfig()
    return _global_options


class _ContextWithDefaultDevice(context.Context):
    def __init__(self, device=''):
        self.__default_device = device
        self.__default_device_spec = pydev.DeviceSpec.from_string(device)  # Must set before super init
        super().__init__()

    @property
    def default_device(self):
        """Return default device"""
        return self.__default_device

    @default_device.setter
    def default_device(self, value):
        self.__default_device = value
        self.__default_device_spec = pydev.DeviceSpec.from_string(value)

    @property
    def _thread_local_data(self):
        if not self.__thread_local_data.device_name:
            self.__thread_local_data.device_name = self.__default_device
            self.__thread_local_data.device_spec = self.__default_device_spec
        return self.__thread_local_data

    @_thread_local_data.setter
    def _thread_local_data(self, value):
        self.__thread_local_data = value


@tf.function
def _graph_engine_warmup():
    return tf.constant(0)


_npu_ctx_lock = threading.Lock()
_npu_device_instances = dict()


def enable_v1():
    if len(_npu_device_instances) > 0:
        os.environ['ASCEND_DEVICE_ID'] = str(list(_npu_device_instances.keys())[0])

    tf.compat.v1.disable_v2_behavior()
    tf.load_op_library(os.path.join(os.path.dirname(__file__), "compat", "v1", "_tf_adapter.so"))


def open(device_id=None):
    """Initiate and return a NPU device handle"""
    if device_id is None:
        device_id = int(os.getenv("ASCEND_DEVICE_ID", '0'))

    with _npu_ctx_lock:
        if not isinstance(context.context(), _ContextWithDefaultDevice):
            ctx = _ContextWithDefaultDevice()
            ctx.ensure_initialized()
            context._set_context(ctx)
            _npu_device_instances.clear()  # Global context has changed since last init npu

        if device_id in _npu_device_instances.keys():
            logging.info('Npu instance on device %s already created', str(device_id))
            return _npu_device_instances.get(device_id)

        if len(_npu_device_instances) > 0:
            raise RuntimeError('Failed create npu instance on device {} as existed instance on {}'
                               ''.format(device_id, list(_npu_device_instances.keys())))

        global_kw_options = global_options().as_dict()
        workers_num = int(os.getenv('RANK_SIZE', '1'))
        if workers_num > 1:
            env_rank_table = os.getenv("RANK_TABLE_FILE")
            env_worker_id = os.getenv('RANK_ID')
            if not env_rank_table:
                raise RuntimeError('You must specify a rank table file by set env RANK_TABLE_FILE in distribution mode')

            if not env_worker_id:
                raise RuntimeError('You must specify rank id by set env RANK_ID in distribution mode')

            global_kw_options['_distribute.rank_table'] = env_rank_table
            global_kw_options['_distribute.rank_id'] = env_worker_id

        device_options = {}
        error_message = _npu_device_backends.Open(context.context()._handle, NPU, device_id, global_kw_options,
                                                  device_options)
        if error_message:
            raise RuntimeError("Failed open npu device %s : %s" % (str(device_id), error_message))

        if workers_num > 1:
            from hccl.manage.api import get_rank_id
            worker_id = get_rank_id()
        else:
            worker_id = 0

        _npu_device_instances[device_id] = NpuDeviceHandle(context.context(), device_id, device_options, workers_num,
                                                           worker_id)
        return _npu_device_instances[device_id]


def close():
    """Close NPU device"""
    _npu_device_backends.Close()


atexit.register(close)

_global_npu_ctx = None


def global_npu_ctx():
    """Get global NPU context"""
    global _global_npu_ctx
    return _global_npu_ctx


_hacked_tensorflow_function = def_function.function
_hacked_def_function_function_call = def_function.Function.__call__
_thread_local = threading.local()


def _never_nested_function_call(self, *func_args, **func_kwargs):
    if not hasattr(_thread_local, "entrance_function"):
        _thread_local.entrance_function = None
    if _thread_local.entrance_function is not None:
        logging.info("Inlining nested tf function %s under %s on npu", self._python_function.__name__,
                     _thread_local.entrance_function)
        try:
            return self._python_function(*func_args, **func_kwargs)
        except:
            logging.info("Bypass inlining nested tf function %s under %s on npu", self._python_function.__name__,
                         _thread_local.entrance_function)
            return _hacked_def_function_function_call(self, *func_args, **func_kwargs)
    _thread_local.entrance_function = self._python_function.__name__
    try:
        return _hacked_def_function_function_call(self, *func_args, **func_kwargs)
    finally:
        _thread_local.entrance_function = None


def npu_compat_function(func=None, *args, **kwargs):
    """NPU compatible function"""

    def never_nested_decorator(f):
        if kwargs.get('experimental_compile'):
            logging.info("Skip xla compile tf function %s on npu", f.__name__)
            kwargs['experimental_compile'] = False
        if kwargs.get('jit_compile'):
            logging.info("Skip xla compile tf function %s on npu", f.__name__)
            kwargs['jit_compile'] = False

        return _hacked_tensorflow_function(*args, **kwargs)(f)

    if func is not None:
        return never_nested_decorator(func)
    return never_nested_decorator


class NpuCompatEagerFunc(script_ops.EagerFunc):
    def __init__(self, *args, **kwargs):
        if hasattr(_thread_local, 'npu_specific_device'):
            self._npu_specific_device = _thread_local.npu_specific_device
        else:
            self._npu_specific_device = None
        super(NpuCompatEagerFunc, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self._npu_specific_device:
            with context.device(self._npu_specific_device):
                return super(NpuCompatEagerFunc, self).__call__(*args, **kwargs)
        else:
            return super(NpuCompatEagerFunc, self).__call__(*args, **kwargs)


def wrap_cpu_only_api(func):
    def wrapper(*args, **kwargs):
        _thread_local.npu_specific_device = '/job:localhost/replica:0/task:0/device:CPU:0'
        try:
            return func(*args, **kwargs)
        finally:
            _thread_local.npu_specific_device = None

    return wrapper


class NpuDeviceHandle:
    """Class for creating handle of NPU device"""

    def __init__(self, ctx, device_id, device_options, workers_num, worker_id):
        self._ctx = ctx
        self._device_id = device_id
        self._device_name = NPU + ":" + str(device_id)
        self._device_options = device_options
        self.workers_num = workers_num
        self.worker_id = worker_id

    def name(self):
        """Return device name"""
        return self._device_name

    def scope(self):
        """Return NPU scope"""

        @tf_contextlib.contextmanager
        def _scope():
            with self._ctx.device(self._device_name):
                yield

        return _scope()

    def is_cluster_worker(self):
        """Whether NPU device is in cluster worker"""
        return self.workers_num > 1 and self.workers_num > self.worker_id >= 0

    def as_default(self):
        """Set device as default one"""

        @tf_contextlib.contextmanager
        def _consistent_with_context_ctx():
            try:
                with context.device(self._ctx.device_name):
                    yield
            except ImportError:  # ImportError: sys.meta_path is None, Python is likely shutting down
                yield

        def _device_consistent_with_context(*args, **kwargs):
            return _consistent_with_context_ctx()

        def_function.Function.__call__ = _never_nested_function_call
        def_function.function = npu_compat_function
        tf.function = npu_compat_function

        ops.device = _device_consistent_with_context
        script_ops.EagerFunc = NpuCompatEagerFunc

        tf.py_function = wrap_cpu_only_api(tf.py_function)

        self._ctx.default_device = self._device_name

        global _global_npu_ctx
        _global_npu_ctx = self

        if os.getenv('GE_USE_STATIC_MEMORY') == '1':  # Warmup graph engine for malloc npu memory in static memory mode
            logging.info("Warmup graph engine in static memory mode")
            _graph_engine_warmup()

        from npu_device.train import npu_convert
        npu_convert.npu_convert_api()

        return self


@tf_contextlib.contextmanager
def npu_run_context(options=None):
    if options is not None and not isinstance(options, NpuRunContextOptions):
        raise ValueError("options type must be NpuRunContextOptions")
    if options is None:
        options = NpuRunContextOptions()
    _thread_local.npu_run_options = options
    try:
        if _thread_local.npu_run_options.experimental.graph_memory_optimize_config.recompute.value is not None:
            _npu_device_backends.RunContextOptionsSetMemoryOptimizeOptions(
                _thread_local.npu_run_options.experimental.graph_memory_optimize_config.recompute.value)
        yield
    finally:
        _npu_device_backends.CleanRunContextOptions()
        _thread_local.npu_run_context = None
