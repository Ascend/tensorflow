import tensorflow as tf
import contextlib
import functools

from tensorflow.python.ops import math_ops
from npu_device.npu_device import global_npu_ctx
from npu_device import gen_npu_ops
from npu_device.npu_device import never_nested_function
from npu_device.distribute.hccl import all_reduce


@never_nested_function
def _is_npu_float_overflow(*args, **kwargs):
    float_status = gen_npu_ops.npu_get_float_status(gen_npu_ops.npu_alloc_float_status())
    no_overflow_status = gen_npu_ops.npu_clear_float_status(float_status)
    worker_status = tf.reduce_all(tf.equal(float_status, no_overflow_status))
    if global_npu_ctx() is None or global_npu_ctx().workers_num <= 1:
        return worker_status
    else:
        return tf.greater(all_reduce(tf.cast(worker_status, dtype=tf.float32), 'sum'), 0.5)


def _is_finite_npu(float_overflow, *args, **kwargs):
    return float_overflow


@contextlib.contextmanager
def _is_finite_npu_hacker():
    _tf_is_finite = math_ops.is_finite
    npu_float_status = _is_npu_float_overflow()
    math_ops.is_finite = functools.partial(_is_finite_npu, npu_float_status)
    yield
    math_ops.is_finite = _tf_is_finite


class NpuLossScaleOptimizer(tf.keras.mixed_precision.LossScaleOptimizer):
    def __init__(self, inner_optimizer, dynamic=True, initial_scale=None, dynamic_growth_steps=None):
        super().__init__(inner_optimizer, dynamic, initial_scale, dynamic_growth_steps)

    @never_nested_function
    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        with tf.control_dependencies([grad for grad, _ in grads_and_vars]):
            with _is_finite_npu_hacker():
                super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)
