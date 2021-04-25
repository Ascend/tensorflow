# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# Description: Common depends and micro defines for and only for data preprocess module

import os
import atexit
import threading
import absl.logging as logging

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.util import tf_contextlib

from npu_device.configs.npu_config import NpuConfig

gen_npu_ops = tf.load_op_library(os.path.dirname(__file__) + "/_npu_ops.so")
NPU = "/job:localhost/replica:0/task:0/device:NPU"

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _npu_device_backends
else:
    import _npu_device_backends


def stupid_repeat(word, times):
    return _npu_device_backends.StupidRepeat(word, times)


_global_options = None
_global_options_lock = threading.Lock()


def global_options():
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
    def _thread_local_data(self):
        if not self.__thread_local_data.device_name:
            self.__thread_local_data.device_name = self.__default_device
            self.__thread_local_data.device_spec = self.__default_device_spec
        return self.__thread_local_data

    @_thread_local_data.setter
    def _thread_local_data(self, value):
        self.__thread_local_data = value

    @property
    def default_device(self):
        return self.__default_device

    @default_device.setter
    def default_device(self, value):
        self.__default_device = value
        self.__default_device_spec = pydev.DeviceSpec.from_string(value)


def open(device_id=None):
    global_kw_options = global_options().as_dict()

    ctx = _ContextWithDefaultDevice()
    ctx.ensure_initialized()

    if device_id is None:
        device_id = int(os.getenv("ASCEND_DEVICE_ID", 0))

    workers_num = 1
    worker_id = 0
    rank_table = os.getenv("RANK_TABLE_FILE")
    if rank_table is not None and len(rank_table.strip()) > 0:
        try:
            workers_num = int(os.getenv('RANK_SIZE'))
            worker_id = int(os.getenv('RANK_ID'))
        except:
            raise RuntimeError('RANK_TABLE_FILE is set, but RANK_SIZE and RANK_ID are not set correctly')
        if not (workers_num > worker_id >= 0):
            raise RuntimeError('RANK_TABLE_FILE is set, but RANK_SIZE and RANK_ID are not set correctly')
        if workers_num > 1:
            global_kw_options['_distribute.rank_id'] = str(worker_id)
            global_kw_options['_distribute.rank_table'] = str(rank_table)

    device_options = {}
    error_message = _npu_device_backends.Open(ctx._handle, NPU, device_id, global_kw_options, device_options)
    if len(error_message):
        raise RuntimeError("Failed open npu device " + str(device_id) + ":" + error_message)

    context._set_context(ctx)
    return NpuDeviceHandle(ctx, device_id, device_options, workers_num, worker_id)


def close():
    _npu_device_backends.Close()


atexit.register(close)

_global_npu_ctx = None


def global_npu_ctx():
    global _global_npu_ctx
    return _global_npu_ctx


_hacked_tensorflow_function = tf.function
_thread_local = threading.local()


def never_nested_function(func=None, *args, **kwargs):
    def never_nested_decorator(f):
        if kwargs.get('experimental_compile'):
            logging.info("Skip xla compile tf function %s on npu", f.__name__)
        kwargs['experimental_compile'] = False
        tf_decorated_func = _hacked_tensorflow_function(*args, **kwargs)(f)

        def wrapper(*func_args, **func_kwargs):
            if not hasattr(_thread_local, "entrance_function"):
                _thread_local.entrance_function = None
            if _thread_local.entrance_function is not None:
                logging.info("Inlining nested tf function %s under %s on npu", f.__name__,
                             _thread_local.entrance_function)
                return f(*func_args, **func_kwargs)
            _thread_local.entrance_function = f.__name__
            result = tf_decorated_func(*func_args, **func_kwargs)
            _thread_local.entrance_function = None
            return result

        return wrapper

    if func is not None:
        return never_nested_decorator(func)
    else:
        return never_nested_decorator


class NpuDeviceHandle(object):
    def __init__(self, ctx, device_id, device_options, workers_num, worker_id):
        self._ctx = ctx
        self._device_id = device_id
        self._device_name = NPU + ":" + str(device_id)
        self._device_options = device_options
        self.workers_num = workers_num
        self.worker_id = worker_id

    def name(self):
        return self._device_name

    def scope(self):
        @tf_contextlib.contextmanager
        def _scope():
            with self._ctx.device(self._device_name):
                yield

        return _scope()

    def is_cluster_worker(self):
        return self.workers_num > 1 and self.workers_num > self.worker_id >= 0

    def as_default(self):
        @tf_contextlib.contextmanager
        def combined():
            try:
                with context.device(self._device_name):
                    yield
            except ImportError:  # ImportError: sys.meta_path is None, Python is likely shutting down
                yield

        def _f(*args, **kwargs):
            return combined()

        ops.device = _f
        tf.function = never_nested_function

        self._ctx.default_device = self._device_name

        global _global_npu_ctx
        _global_npu_ctx = self

        return self
