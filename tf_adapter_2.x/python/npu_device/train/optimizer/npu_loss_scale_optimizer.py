import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables

from npu_device.npu_device import global_npu_ctx
from npu_device import gen_npu_ops
from npu_device.npu_device import never_nested_function
from npu_device.distribute.hccl import all_reduce


@never_nested_function
def _npu_finite_status_after_executed(executed_ops):
    with tf.control_dependencies([executed_ops]):
        current_status = gen_npu_ops.npu_alloc_float_status()
        assign_float_status = gen_npu_ops.npu_get_float_status(current_status)
        finite_status = gen_npu_ops.npu_clear_float_status(assign_float_status)
        if global_npu_ctx() and global_npu_ctx().workers_num > 1:
            with tf.control_dependencies([assign_float_status]):
                reduced_status = all_reduce(current_status, 'sum', fusion=0)
            return tf.reduce_all(tf.equal(reduced_status, finite_status))
        else:
            return tf.reduce_all(tf.equal(current_status, finite_status))


def _npu_compat_loss_scale_update(m, grads):
    def update_if_finite_grads():
        def incr_loss_scale():
            incr_result_finite = tf.less(m.current_loss_scale, 4e+38 / m.multiplier)
            update_if_finite_fn = tf.cond(incr_result_finite,
                                          m.current_loss_scale.assign(m.current_loss_scale * m.multiplier),
                                          tf.no_op())
            return tf.group(update_if_finite_fn, m.counter.assign(0))

        return tf.cond(m.counter + 1 >= m.growth_steps,
                       incr_loss_scale,
                       lambda: m.counter.assign_add(1))

    def update_if_not_finite_grads():
        new_loss_scale = tf.maximum(m.current_loss_scale / m.multiplier, 1)
        return tf.group(m.counter.assign(0), m.current_loss_scale.assign(new_loss_scale))

    is_finite = _npu_finite_status_after_executed(grads)
    update_op = tf.cond(is_finite,
                        update_if_finite_grads,
                        update_if_not_finite_grads)
    should_apply_gradients = is_finite
    return update_op, should_apply_gradients


class NpuLossScaleOptimizer(tf.keras.mixed_precision.LossScaleOptimizer):
    def __init__(self, inner_optimizer, dynamic=True, initial_scale=None, dynamic_growth_steps=None):
        super().__init__(inner_optimizer, dynamic, initial_scale, dynamic_growth_steps)
        self._last_step_infinite = variable_scope.variable(
            initial_value=False,
            name="npu_last_step_infinite",
            dtype=tf.bool,
            trainable=False,
            use_resource=True,
            synchronization=variables.VariableSynchronization.AUTO,
            aggregation=variables.VariableAggregation.NONE
        )

    @property
    def last_step_infinite(self):
        return self._last_step_infinite

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        if global_npu_ctx() is None:
            super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

        def apply_fn():
            self._apply_gradients(grads, vars, name, experimental_aggregate_gradients)

        def do_not_apply_fn():
            self._optimizer.iterations.assign_add(1, read_value=False)

        grads, vars = list(zip(*grads_and_vars))
        if self.dynamic:
            loss_scale_update_op, should_apply_grads = _npu_compat_loss_scale_update(self._loss_scale, grads)
        else:
            loss_scale_update_op = tf.no_op()
            should_apply_grads = _npu_finite_status_after_executed(grads)

        self._last_step_infinite.assign(should_apply_grads)
        maybe_apply_op = tf.cond(should_apply_grads, apply_fn, do_not_apply_fn)
        return tf.group(maybe_apply_op, loss_scale_update_op)
