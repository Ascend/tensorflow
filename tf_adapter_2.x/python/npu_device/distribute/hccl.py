from npu_device.distribute import hccl_ops
from npu_device.npu_device import global_npu_ctx
from npu_device.npu_device import never_nested_function

import tensorflow as tf
from absl import logging


def shard_and_rebatch_dataset(dataset, global_bs):
    if global_npu_ctx() is None or global_npu_ctx().workers_num <= 1:
        return dataset, global_bs
    if global_bs % global_npu_ctx().workers_num != 0:
        raise ValueError('Batch size must be divisible by num npus: {}'.format(global_npu_ctx().workers_num))

    batch_size = int(global_bs) / global_npu_ctx().workers_num
    dataset = dataset.shard(global_npu_ctx().workers_num, global_npu_ctx().worker_id)

    return dataset, int(batch_size)


@never_nested_function
def _all_reduce(values, reduction, fusion, fusion_id, group):
    workers_num = global_npu_ctx().workers_num

    mean_reduce = False
    if reduction == 'mean':
        mean_reduce = True
        reduction = 'sum'

    reduced_values = []
    for value in values:
        reduced_value = hccl_ops.allreduce(value, reduction, fusion, fusion_id, group)
        is_float = reduced_value.dtype in (tf.float16, tf.float32, tf.float64)
        if is_float:
            typed_workers_num = tf.cast(1.0 / float(workers_num), reduced_value.dtype)
        else:
            typed_workers_num = tf.cast(workers_num, reduced_value.dtype)
        with tf.control_dependencies([tf.group(*values)]):
            if mean_reduce:
                if is_float:
                    reduced_values.append(tf.multiply(reduced_value, typed_workers_num))
                else:
                    reduced_values.append(tf.divide(reduced_value, typed_workers_num))
            else:
                reduced_values.append(tf.multiply(reduced_value, tf.cast(1, reduced_value.dtype)))
    return reduced_values


def all_reduce(values, reduction, fusion=1, fusion_id=-1, group="hccl_world_group"):
    if global_npu_ctx() is None or not global_npu_ctx().is_cluster_worker():
        logging.info("Skip all reduce as current process is not npu cluster worker")
        return values

    if isinstance(values, (list, tuple,)):
        return _all_reduce(values, reduction, fusion, fusion_id, group)
    else:
        return _all_reduce([values], reduction, fusion, fusion_id, group)[0]


@never_nested_function
def _broadcast(values, root_rank, fusion, fusion_id, group):
    for value in values:
        value.assign(hccl_ops.broadcast([value], root_rank, fusion, fusion_id, group)[0])


def broadcast(values, root_rank, fusion=2, fusion_id=0, group="hccl_world_group"):
    if global_npu_ctx() is None or not global_npu_ctx().is_cluster_worker():
        logging.info("Skip broadcast as current process is not npu cluster worker")
        return
    if isinstance(values, (list, tuple,)):
        _broadcast(values, root_rank, fusion, fusion_id, group)
    else:
        _broadcast([values], root_rank, fusion, fusion_id, group)
