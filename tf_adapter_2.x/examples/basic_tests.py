#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import os
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_resource_variable_ops

import npu_device

npu = npu_device.open().as_default()


def tensor_equal(t1, t2):
  return (t1.numpy() == t2.numpy()).all()


@tf.function
def foo_add(v1, v2):
  return v1 + v2


@tf.function
def foo_add_(v):
  return v.assign_add(1)


@tf.function
def foo_cpu_add_(v):
  with context.device("/job:localhost/replica:0/task:0/device:CPU:0"):
    return v.assign_add(1)


class RaiseTest(unittest.TestCase):
  def test_raise1(self):
    with context.device("/job:localhost/replica:0/task:0/device:CPU:0"):
      x = tf.Variable(1)
    y = tf.Variable(1)
    self.assertRaises(tf.errors.InvalidArgumentError, foo_add, x, y)

  def test_basic1(self):
    self.assertTrue(tensor_equal(foo_add(1, 2), tf.constant(3)))

  def test_basic2(self):
    self.assertTrue(tensor_equal(tf.add(1, 2), tf.constant(3)))

  def test_basic3(self):
    x = tf.Variable(1)
    self.assertTrue(tensor_equal(foo_add_(x), tf.constant(2)))

  def test_basic4(self):
    with context.device("/job:localhost/replica:0/task:0/device:CPU:0"):
      x = tf.Variable(1)
    self.assertTrue(tensor_equal(foo_add_(x), tf.constant(2)))

  def test_basic5(self):
    with context.device("/job:localhost/replica:0/task:0/device:CPU:0"):
      x = tf.Variable(1)
    self.assertTrue(tensor_equal(foo_cpu_add_(x), tf.constant(2)))

  def test_basic6(self):  # Force run on npu by tensorflow
    x = tf.Variable(1)
    self.assertTrue(tensor_equal(foo_cpu_add_(x), tf.constant(2)))

  def test_basic7(self):  # Force run on npu by tensorflow
    x = tf.Variable(1)
    self.assertTrue(x.device == npu.name())
    self.assertTrue(foo_cpu_add_(x).device == "/job:localhost/replica:0/task:0/device:CPU:0")
    with context.device("/job:localhost/replica:0/task:0/device:CPU:0"):
      x = tf.Variable(1)
    self.assertTrue(foo_add_(x).device == "/job:localhost/replica:0/task:0/device:CPU:0")

  def test_shared_variable(self):
    x = gen_resource_variable_ops.var_handle_op(dtype=tf.float32, shape=(1, 2), shared_name="variable_1")
    gen_resource_variable_ops.assign_variable_op(x, tf.constant([[1.0, 2.0]]))
    y = gen_resource_variable_ops.var_handle_op(dtype=tf.float32, shape=(1, 2), shared_name="variable_1")
    gen_resource_variable_ops.assign_variable_op(y, tf.constant([[2.0, 3.0]]))
    read_x = gen_resource_variable_ops.read_variable_op(x, dtype=tf.float32)
    read_y = gen_resource_variable_ops.read_variable_op(y, dtype=tf.float32)
    self.assertTrue(tensor_equal(read_x, read_y))

    x = gen_resource_variable_ops.var_handle_op(dtype=tf.float32, shape=(1, 2), shared_name=context.shared_name())
    gen_resource_variable_ops.assign_variable_op(x, tf.constant([[1.0, 2.0]]))
    y = gen_resource_variable_ops.var_handle_op(dtype=tf.float32, shape=(1, 2), shared_name=context.shared_name())
    gen_resource_variable_ops.assign_variable_op(y, tf.constant([[2.0, 3.0]]))
    read_x = gen_resource_variable_ops.read_variable_op(x, dtype=tf.float32)
    read_y = gen_resource_variable_ops.read_variable_op(y, dtype=tf.float32)
    self.assertFalse(tensor_equal(read_x, read_y))

  def test_anonymous_variable(self):
    x = tf.Variable([[1.0, 2.0]], dtype=tf.float32, name="x")
    y = tf.Variable([[1.0, 2.0]], dtype=tf.float32, name="x")
    x.assign_add([[1.0, 1.0]])
    self.assertFalse(tensor_equal(x, y))

  def test_matmul(self):
    input = tf.constant([[1.0], [2.0]])
    weight = tf.Variable([[2.0, 1.0]], dtype=tf.float32)
    logit = tf.matmul(input, weight)
    self.assertTrue(tensor_equal(logit, tf.constant([[2., 1.], [4., 2.]])))

  def test_unique(self):
    x = tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8])
    y, idx = tf.unique(x)
    self.assertTrue(tensor_equal(y, tf.constant([1, 2, 4, 7, 8])))

  def test_dataset(self):
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant([2]))
    iterator = iter(dataset)
    self.assertTrue(tensor_equal(next(iterator), tf.constant(2)))
    try:
      next(iterator)
    except Exception as e:
      self.assertTrue(isinstance(e, StopIteration))

  def test_dataset_function(self):
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant([2]))
    iterator = iter(dataset)

    @tf.function
    def f(iterator):
      return next(iterator)

    self.assertTrue(tensor_equal(f(iterator), tf.constant(2)))
    self.assertRaises(tf.errors.OutOfRangeError, f, iterator)

  def test_checkpoint(self):
    step = tf.Variable(0, name="step")  # 0
    checkpoint = tf.train.Checkpoint(step=step)
    checkpoint.write("./ckpt")
    step.assign_add(1)  # 1
    checkpoint.read("./ckpt")
    self.assertTrue(tensor_equal(step, tf.constant(0)))

  def test_same_python_name_function(self):
    def f1():
      @tf.function
      def f(x):
        return x + 1

      return f(tf.constant(1))

    def f2():
      @tf.function
      def f(x):
        return x + 2

      return f(tf.constant(1))

    self.assertTrue(tensor_equal(f1(), tf.constant(2)))
    self.assertTrue(tensor_equal(f2(), tf.constant(3)))

  def test_cond1(self):
    cond = tf.Variable(1.0)
    x = tf.Variable(1.0)
    y = tf.Variable(2.0)

    @tf.function
    def f():
      tf.cond(cond < tf.constant(2.0), lambda: x.assign_add(y), lambda: y.assign_add(x))
      return x, y

    v1, v2 = f()
    self.assertTrue(tensor_equal(v1, tf.constant(3.0)))
    self.assertTrue(tensor_equal(v2, tf.constant(2.0)))

  def test_cond2(self):
    cond = tf.Variable(1.0)
    x = tf.Variable(0.0)
    y = tf.Variable(0.0)

    @tf.function
    def f():
      tf.cond(cond < tf.constant(2.0), lambda: x.assign_add(1.0), lambda: y.assign_add(1.0))
      return x, y

    v1, v2 = f()
    self.assertTrue(tensor_equal(v1, tf.constant(1.0)))
    self.assertTrue(tensor_equal(v2, tf.constant(0.0)))

  def test_cond3(self):
    v = tf.Variable(1.0)
    x = tf.Variable(0.0)
    y = tf.Variable(0.0)

    def x_add():
      return x.assign_add(1.0)

    def y_add():
      return y.assign_add(1.0)

    @tf.function
    def f():
      tf.cond(v < tf.constant(2.0), x_add, y_add)
      return x, y

    v1, v2 = f()
    self.assertTrue(tensor_equal(v1, tf.constant(1.0)))
    self.assertTrue(tensor_equal(v2, tf.constant(0.0)))

  def test_cond4(self):
    v = tf.Variable(1.0)
    x = tf.Variable(0.0)
    y = tf.Variable(0.0)

    @tf.function
    def x_add():
      return x.assign_add(1.0)

    @tf.function
    def y_add():
      return y.assign_add(1.0)

    @tf.function
    def f():
      tf.cond(v < tf.constant(2.0), x_add, y_add)
      return x, y

    v1, v2 = f()
    self.assertTrue(tensor_equal(v1, tf.constant(1.0)))
    self.assertTrue(tensor_equal(v2, tf.constant(0.0)))

  def test_cond5(self):
    v = tf.Variable(1.0)
    x = tf.Variable(0.0)
    y = tf.Variable(0.0)

    c = tf.constant(1.0)

    @tf.function
    def x_add():
      return x.assign_add(c)

    @tf.function
    def y_add():
      return y.assign_add(c)

    @tf.function
    def f():
      tf.cond(v < tf.constant(2.0), x_add, y_add)
      return x, y

    v1, v2 = f()
    self.assertTrue(tensor_equal(v1, tf.constant(1.0)))
    self.assertTrue(tensor_equal(v2, tf.constant(0.0)))

  def test_cond6(self):
    cond = tf.Variable(1.0)
    x = tf.Variable(1.0)
    y = tf.Variable(2.0)

    @tf.function
    def f():
      return tf.cond(cond < tf.constant(2.0), lambda: x.assign_add(y), lambda: y.assign_add(x))

    self.assertTrue(tensor_equal(f(), tf.constant(3.0)))

  def test_while(self):
    v = tf.Variable(1.0)

    @tf.function
    def f():
      for i in tf.range(10):
        v.assign_add(1.0)
      return v

    self.assertTrue(tensor_equal(f(), tf.constant(11.0)))

  def test_variable_need_different_format_in_subgraph_with_control(self):
    x = tf.Variable(tf.constant([[[[0.0]]]]), dtype=tf.float32, shape=(1, 1, 1, 1))

    @tf.function
    def f():
      xv = tf.cond(x < tf.constant([[[[2.0]]]]), lambda: x.assign(tf.constant([[[[10.0]]]])),
              lambda: x.assign(tf.constant([[[[20.0]]]])))
      return tf.nn.conv2d(xv, tf.constant([[[[3.0]]], ], dtype=tf.float32), strides=[1, 1, 1, 1], padding='VALID')
    self.assertTrue(tensor_equal(f(), tf.constant([[[[30.0]]], ], dtype=tf.float32)))
    self.assertTrue(tensor_equal(x, tf.constant([[[[10.0]]], ], dtype=tf.float32)))

  def test_variable_need_different_format_in_subgraph(self):
    x = tf.Variable(tf.constant([[[[0.0]]]]), dtype=tf.float32, shape=(1, 1, 1, 1))

    @tf.function
    def f():
      tf.cond(x < tf.constant([[[[2.0]]]]), lambda: x.assign(tf.constant([[[[10.0]]]])),
                   lambda: x.assign(tf.constant([[[[20.0]]]])))
      return tf.nn.conv2d(x, tf.constant([[[[3.0]]], ], dtype=tf.float32), strides=[1, 1, 1, 1], padding='VALID')
    self.assertTrue(tensor_equal(f(), tf.constant([[[[30.0]]], ], dtype=tf.float32)))
    self.assertTrue(tensor_equal(x, tf.constant([[[[10.0]]], ], dtype=tf.float32)))

  def test_variable_need_different_format_in_subgraph_cross(self):
    x = tf.Variable(tf.constant([[[[10.0]]]]), dtype=tf.float32, shape=(1, 1, 1, 1))

    @tf.function
    def f():
      c1 = tf.nn.conv2d(x, tf.constant([[[[3.0]]], ], dtype=tf.float32), strides=[1, 1, 1, 1], padding='VALID')
      tf.cond(x < tf.constant([[[[2.0]]]]), lambda: x.assign(tf.constant([[[[10.0]]]])),
              lambda: x.assign(tf.constant([[[[20.0]]]])))
      return c1, tf.nn.conv2d(x, tf.constant([[[[3.0]]], ], dtype=tf.float32), strides=[1, 1, 1, 1], padding='VALID')
    c1, c2 = f()
    self.assertTrue(tensor_equal(c1, tf.constant([[[[30.0]]], ], dtype=tf.float32)))
    self.assertTrue(tensor_equal(c2, tf.constant([[[[60.0]]], ], dtype=tf.float32)))
    self.assertTrue(tensor_equal(x, tf.constant([[[[20.0]]], ], dtype=tf.float32)))

  def test_variable_need_different_format_in_subgraph_trans_merge(self):
    x = tf.Variable(tf.constant([[[[10.0]]]]), dtype=tf.float32, shape=(1, 1, 1, 1))

    @tf.function
    def f():
      c1 = tf.nn.conv2d(x, tf.constant([[[[3.0]]], ], dtype=tf.float32), strides=[1, 1, 1, 1], padding='VALID')
      c2 = tf.nn.conv2d(x, tf.constant([[[[3.0]]], ], dtype=tf.float32), strides=[1, 1, 1, 1], padding='VALID')
      tf.cond(x < tf.constant([[[[2.0]]]]), lambda: x.assign(tf.constant([[[[10.0]]]])),
              lambda: x.assign(tf.constant([[[[20.0]]]])))
      return c1, c2
    c1, c2 = f()
    self.assertTrue(tensor_equal(c1, tf.constant([[[[30.0]]], ], dtype=tf.float32)))
    self.assertTrue(tensor_equal(c2, tf.constant([[[[30.0]]], ], dtype=tf.float32)))
    self.assertTrue(tensor_equal(x, tf.constant([[[[20.0]]], ], dtype=tf.float32)))

  def test_bert_dp_under_one_device_distribute_strategy(self):
    def decode_record(record, name_to_features):
      """Decodes a record to a TensorFlow example."""
      example = tf.io.parse_single_example(record, name_to_features)

      # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
      # So cast all int64 to int32.
      for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
          t = tf.cast(t, tf.int32)
        example[name] = t

      return example

    def dataset_fn(ctx=None):
      """Creates input dataset from (tf)records files for pretraining."""
      input_patterns = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_examples.tfrecord")]
      seq_length = 128
      max_predictions_per_seq = 20
      batch_size = 32
      is_training = True
      input_pipeline_context = None
      use_next_sentence_label = True
      use_position_id = False
      output_fake_labels = True

      name_to_features = {
        'input_ids':
          tf.io.FixedLenFeature([seq_length], tf.int64),
        'input_mask':
          tf.io.FixedLenFeature([seq_length], tf.int64),
        'segment_ids':
          tf.io.FixedLenFeature([seq_length], tf.int64),
        'masked_lm_positions':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        'masked_lm_ids':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        'masked_lm_weights':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
      }
      if use_next_sentence_label:
        name_to_features['next_sentence_labels'] = tf.io.FixedLenFeature([1],
                                                                         tf.int64)
      if use_position_id:
        name_to_features['position_ids'] = tf.io.FixedLenFeature([seq_length],
                                                                 tf.int64)
      for input_pattern in input_patterns:
        if not tf.io.gfile.glob(input_pattern):
          raise ValueError('%s does not match any files.' % input_pattern)

      dataset = tf.data.Dataset.list_files(input_patterns, shuffle=is_training)

      if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                                input_pipeline_context.input_pipeline_id)
      if is_training:
        dataset = dataset.repeat()

        # We set shuffle buffer to exactly match total number of
        # training files to ensure that training data is well shuffled.
        input_files = []
        for input_pattern in input_patterns:
          input_files.extend(tf.io.gfile.glob(input_pattern))
        dataset = dataset.shuffle(len(input_files))

      # # In parallel, create tf record dataset for each train files.
      # # cycle_length = 8 means that up to 8 files will be read and deserialized in
      # # parallel. You may want to increase this number if you have a large number of
      # # CPU cores.
      dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=8,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

      if is_training:
        dataset = dataset.shuffle(100)

      decode_fn = lambda record: decode_record(record, name_to_features)
      dataset = dataset.map(
        decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

      def _select_data_from_record(record):
        """Filter out features to use for pretraining."""
        x = {
          'input_word_ids': record['input_ids'],
          'input_mask': record['input_mask'],
          'input_type_ids': record['segment_ids'],
          'masked_lm_positions': record['masked_lm_positions'],
          'masked_lm_ids': record['masked_lm_ids'],
          'masked_lm_weights': record['masked_lm_weights'],
        }
        if use_next_sentence_label:
          x['next_sentence_labels'] = record['next_sentence_labels']
        if use_position_id:
          x['position_ids'] = record['position_ids']

        # TODO(hongkuny): Remove the fake labels after migrating bert pretraining.
        if output_fake_labels:
          return (x, record['masked_lm_weights'])
        else:
          return x

      dataset = dataset.map(
        _select_data_from_record,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      dataset = dataset.batch(batch_size, drop_remainder=is_training)
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      return dataset

    strategy = tf.distribute.OneDeviceStrategy("device:CPU:0")
    dataset = strategy.experimental_distribute_datasets_from_function(dataset_fn)
    iterator = iter(dataset)

    @tf.function
    def bert_step(iterator):
      return next(iterator)

    bert_step(iterator)


if __name__ == '__main__':
  unittest.main()
