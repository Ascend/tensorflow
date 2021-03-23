# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# Description: Common depends and micro defines for and only for data preprocess module

import os
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.distribute import distribute_lib
import threading

NPU = "/job:localhost/replica:0/task:0/device:NPU"

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _npu_device_backends
else:
    import _npu_device_backends


def stupid_repeat(word, times):
    return _npu_device_backends.StupidRepeat(word, times)


def open(ctx=None, device_index=None, global_options=None, session_options=None):
    if global_options is None:
        global_options = {}
    if session_options is None:
        session_options = {}
    if ctx is None:
        ctx = context.context()
    ctx.ensure_initialized()

    if device_index is None:
        device_index = int(os.getenv("ASCEND_DEVICE_ID", 0))

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
            global_options['ge.exec.rankTableFile'] = str(rank_table)
            global_options['ge.exec.deployMode'] = "0"
            global_options['ge.exec.isUseHcom'] = "1"
            global_options['ge.exec.hcclFlag'] = "1"
            global_options['ge.exec.rankId'] = str(worker_id)

    error_message = _npu_device_backends.Open(ctx._handle, NPU, device_index, global_options, session_options)
    if len(error_message):
        raise RuntimeError("Failed open npu device " + str(device_index) + ":" + error_message)
    return NpuDeviceHandle(ctx, device_index, workers_num, worker_id)


def close():
    _npu_device_backends.Close()


import atexit

atexit.register(close)
from tensorflow.python.util import tf_contextlib

_global_npu_ctx = None


def global_npu_ctx():
    global _global_npu_ctx
    return _global_npu_ctx


class NpuDeviceHandle(object):
    def __init__(self, ctx, device_index, workers_num, worker_id):
        self._ctx = ctx
        self._device_name = NPU + ":" + str(device_index)
        self.workers_num = workers_num
        self.worker_id = worker_id
        self._hacked_tensorflow_function = tf.function
        self._thread_local = threading.local()
        self._thread_local._entrance_function = None

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
        self._ctx._set_device(self._device_name, pydev.DeviceSpec.from_string(self._device_name))

        def never_nested_function(func=None, *args, **kwargs):
            if not hasattr(self._thread_local, "_entrance_function"):
                self._thread_local._entrance_function = None

            def never_nested_decorator(func):
                tf_decorated_func = self._hacked_tensorflow_function(*args, **kwargs)(func)

                def wrapper(*func_args, **func_kwargs):
                    if not hasattr(self._thread_local, "_entrance_function"):
                        self._thread_local._entrance_function = None
                    if self._thread_local._entrance_function is not None:
                        return func(*func_args, **func_kwargs)
                    self._thread_local._entrance_function = func.__name__
                    result = tf_decorated_func(*func_args, **func_kwargs)
                    self._thread_local._entrance_function = None
                    return result

                return wrapper

            if func is not None:
                return never_nested_decorator(func)
            else:
                return never_nested_decorator

        tf.function = never_nested_function

        global _global_npu_ctx
        _global_npu_ctx = self

        return self
