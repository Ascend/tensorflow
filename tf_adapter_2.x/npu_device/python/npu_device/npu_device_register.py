# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# Description: Common depends and micro defines for and only for data preprocess module

import tensorflow as tf
from tensorflow.python.eager import context
from sys import version_info as _swig_python_version_info

if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

NPU = "/job:localhost/replica:0/task:0/device:NPU"

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _npu_device_backends
else:
    import _npu_device_backends


def stupid_repeat(word, times):
    return _npu_device_backends.StupidRepeat(word, times)


def open(ctx=None, device_index=0, global_options={}, session_options={}):
    if ctx is None:
        ctx = context.context()
    ctx.ensure_initialized()
    error_message = _npu_device_backends.Open(ctx._handle, NPU, device_index, global_options, session_options)
    if len(error_message):
        raise RuntimeError("Failed open npu device " + str(device_index) + ":" + error_message)
    return NpuDeviceHandle(ctx, device_index)


def close():
    _npu_device_backends.Close()


import atexit

atexit.register(close)
from tensorflow.python.util import tf_contextlib


class NpuDeviceHandle(object):
    def __init__(self, ctx, device_index):
        self._ctx = ctx
        self._device_name = NPU + ":" + str(device_index)

    def name(self):
        return self._device_name

    def scope(self):
        @tf_contextlib.contextmanager
        def _scope():
            with self._ctx.device(self._device_name):
                yield

        return _scope()

    def as_default(self):
        from tensorflow.python.framework import device as pydev
        from tensorflow.python.framework import ops

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
        return self
