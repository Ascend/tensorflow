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
            global_options['ge.exec.hccl_tailing_optimize'] = '1'
            global_options['ge.exec.hcomParallel'] = "1"
            global_options['ge.exec.rankId'] = str(worker_id)

    error_message = _npu_device_backends.Open(ctx._handle, NPU, device_index, global_options, session_options)
    if len(error_message):
        raise RuntimeError("Failed open npu device " + str(device_index) + ":" + error_message)
    return NpuDeviceHandle(ctx, device_index, workers_num, worker_id)


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
        tf_decorated_func = _hacked_tensorflow_function(*args, **kwargs)(f)

        def wrapper(*func_args, **func_kwargs):
            if not hasattr(_thread_local, "entrance_function"):
                _thread_local.entrance_function = None
            if _thread_local.entrance_function is not None:
                logging.info("Flat nested tf function %s", f.__name__)
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
    def __init__(self, ctx, device_index, workers_num, worker_id):
        self._ctx = ctx
        self._device_name = NPU + ":" + str(device_index)
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
        self._ctx._set_device(self._device_name, pydev.DeviceSpec.from_string(self._device_name))

        tf.function = never_nested_function

        global _global_npu_ctx
        _global_npu_ctx = self

        return self
