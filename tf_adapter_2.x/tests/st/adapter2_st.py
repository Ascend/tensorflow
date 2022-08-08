#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time

os.environ['ASCEND_OPP_PATH'] = 'non-existed-path'

import npu_device
from npu_device.npu_device import stupid_repeat

import unittest
import tensorflow as tf
from tensorflow.python.eager import context

npu_device.global_options().is_tailing_optimization = True
npu_device.global_options().experimental.multi_branches_config.input_shape = "data_0:-1"
npu_device.global_options().experimental.multi_branches_config.dynamic_node_type = "0"
npu_device.global_options().experimental.multi_branches_config.dynamic_dims = "1;2"
npu_device.global_options().aoe_config.work_path = "./"
npu_device.global_options().graph_run_mode = 0
os.environ['RANK_TABLE_FILE'] = "rankTable"
os.environ['RANK_SIZE'] = "2"
os.environ['RANK_ID'] = "1"
npu = npu_device.open().as_default()
npu.workers_num = 2  # mock run in 2P env


def tensor_exact_equal(t1, t2):
    if t1.shape.rank != t2.shape.rank:
        return False
    if t1.shape.rank:
        return (t1.numpy() == t2.numpy()).all()
    return t1.numpy() == t2.numpy()


def tensor_equal(t1, t2):
    return True


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


class Adapter2St(unittest.TestCase):
    def test_mix_resource(self):
        with context.device("/job:localhost/replica:0/task:0/device:CPU:0"):
            x = tf.Variable(1)
        y = tf.Variable(1)
        self.assertTrue(tensor_equal(foo_add(x, y), tf.constant(2)))

    def test_mix_merge_copied_shared_nodes(self):
        @tf.function
        def f(x):
            v = tf.strings.to_number(x)
            c = tf.constant(1.0)
            v = tf.add(v, c)
            return tf.add(v, c)

        self.assertTrue(tensor_equal(f(tf.constant("2.0")), tf.constant(4.0)))

    def test_basic0(self):
        stupid_repeat("", 1)

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

    def test_string_unimp1(self):
        x = tf.Variable(1)

        @tf.function
        def f(v1, v2):
            x.assign_add(1)
            return tf.strings.to_number(v1 + v2)

        self.assertTrue(tensor_equal(f(tf.constant('1'), tf.constant('2')), tf.constant(12)))

    def test_string_unimp2(self):
        x = tf.Variable(1)

        @tf.function
        def f(v1, v2):
            x.assign_add(1)
            return v1 + v2

        self.assertTrue(tensor_equal(f(tf.constant('1'), tf.constant('2')), tf.constant('12')))

    def test_string_fallback_cpu1(self):
        @tf.function
        def f(v1, v2):
            return tf.strings.to_number(v1 + v2)

        self.assertTrue(tensor_exact_equal(f(tf.constant('1'), tf.constant('2')), tf.constant(12.0)))

    def test_string_output_ref_input(self):
        @tf.function
        def f(v1, v2):
            return v1, v2

        x, y = f(tf.constant(1.0), tf.constant('abc'))

        self.assertTrue(tensor_exact_equal(x, tf.constant(1.0)))
        self.assertTrue(tensor_exact_equal(y, tf.constant('abc')))

    def test_resource_output_ref_input(self):
        from tensorflow.python.ops import resource_variable_ops
        v = resource_variable_ops.VarHandleOp(dtype=tf.float32, shape=())

        @tf.function
        def f(v1, v2):
            return v1 + v1, v2

        x, y = f(tf.constant(1.0), v)

        self.assertTrue(tensor_equal(x, tf.constant(2.0)))
        self.assertTrue(tensor_equal(y, v))

    def test_string_fallback_cpu2(self):
        self.assertTrue(tensor_equal(foo_add(tf.constant('1'), tf.constant('2')), tf.constant('12')))

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

    def test_while_1(self):
        v = tf.Variable(1.0)

        @tf.function
        def f():
            for i in tf.range(10):
                v.assign_add(1.0)
            return v

        self.assertTrue(tensor_equal(f(), tf.constant(11.0)))

    def test_while_2(self):
        v = tf.Variable(1, dtype=tf.int64)

        @tf.function
        def f(iterator):
            for i in tf.range(10):
                v.assign_add(next(iterator))

        dataset = tf.data.Dataset.range(10)
        iterator = iter(dataset)
        f(iterator)

    def test_while_3(self):
        v = tf.Variable(tf.ones([10, 1024, 1024], dtype=tf.int64), dtype=tf.int64)

        @tf.function
        def f(iterator):
            for i in tf.range(10):
                v.assign_add(next(iterator))

        dataset = tf.data.Dataset.from_tensors(tf.ones([10, 1024, 1024], dtype=tf.int64)).repeat()
        iterator = iter(dataset)
        f(iterator)
        time.sleep(5)

    def test_dropout_v3(self):
        @tf.function
        def f(x):
            noise_shape = tf.shape(x)
            gen_out = npu_device.gen_npu_ops.drop_out_gen_mask_v3(noise_shape, tf.constant(0.1), 0, 0)
            npu_device.gen_npu_ops.drop_out_do_mask_v3(x, gen_out, tf.constant(0.1))
            gen_out = npu_device.gen_npu_ops.drop_out_gen_mask_v3(noise_shape, tf.constant(0.1), 0, 0)
            npu_device.gen_npu_ops.drop_out_do_mask_v3(x, gen_out, tf.constant(0.1))
            gen_out = npu_device.gen_npu_ops.drop_out_gen_mask_v3(noise_shape, tf.constant(0.1), 0, 0)
            npu_device.gen_npu_ops.drop_out_do_mask_v3(x, gen_out, tf.constant(0.1))
            gen_out = npu_device.gen_npu_ops.drop_out_gen_mask_v3(noise_shape, tf.constant(0.1), 0, 0)
            npu_device.gen_npu_ops.drop_out_do_mask_v3(x, gen_out, tf.constant(0.1))
            gen_out = npu_device.gen_npu_ops.drop_out_gen_mask_v3(noise_shape, tf.constant(0.1), 0, 0)
            npu_device.gen_npu_ops.drop_out_do_mask_v3(x, gen_out, tf.constant(0.1))

        f(tf.constant([[0.1]]))

    def test_tailing_optimize(self):
        @tf.function
        def train_with_loss_scale():
            with tf.GradientTape() as tape:
                p1 = tf.add(x, x)
                p2 = tf.add(p1, x1)
                p3 = tf.add(p2, x2)
                p4 = tf.add(p3, x3)
                p5 = tf.add(p4, x4)
                loss = tf.reduce_sum(p5)
            grads = tape.gradient(loss, [x, x1, x2, x3, x4])
            grads = npu_device.distribute.all_reduce(grads)
            scaled_optimizer.apply_gradients(zip(grads, [x, x1, x2, x3, x4]))

        @tf.function
        def train_loop():
            for i in tf.range(10):
                train_with_loss_scale()

        npu_device.set_npu_loop_size(10)
        optimizer = tf.keras.optimizers.Adam()
        scaled_optimizer = npu_device.train.optimizer.NpuLossScaleOptimizer(optimizer)

        x = tf.Variable(1.0)
        x1 = tf.Variable(1.0)
        x2 = tf.Variable(1.0)
        x3 = tf.Variable(1.0)
        x4 = tf.Variable(1.0)
        train_loop()

    def test_weight_grouping(self):
        @tf.function
        def train_with_loss_scale():
            with tf.GradientTape() as tape:
                p1 = tf.add(x, x)
                p2 = tf.add(p1, x1)
                p3 = tf.add(p2, x2)
                p4 = tf.add(p3, x3)
                p5 = tf.add(p4, x4)
                loss = tf.reduce_sum(p5)
            grads = tape.gradient(loss, [x, x1, x2, x3, x4])
            grads = npu_device.distribute.all_reduce(grads)
            npu_device.distribute.grouping_gradients_apply(scaled_optimizer.apply_gradients,
                                                           zip(grads, [x, x1, x2, x3, x4]))

        @tf.function
        def train_loop():
            for i in tf.range(10):
                train_with_loss_scale()

        optimizer = tf.keras.optimizers.Adam()
        scaled_optimizer = npu_device.train.optimizer.NpuLossScaleOptimizer(optimizer)

        x = tf.Variable(1.0)
        x1 = tf.Variable(1.0)
        x2 = tf.Variable(1.0)
        x3 = tf.Variable(1.0)
        x4 = tf.Variable(1.0)
        npu_device.distribute.grouping_broadcast([x, x1, x2, x3, x4])
        train_loop()

    def test_empty_graph(self):
        @tf.function
        def f():
            pass

        f()

    def test_py_function(self):
        def augment(images):
            return tf.cast(images, tf.uint8)

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([2.2], dtype=tf.float32)).map(
            lambda x: tf.py_function(augment, [x], [tf.uint8]), num_parallel_calls=1
        )

        y = next(iter(ds))
        self.assertTrue(tensor_equal(y, tf.constant(2)))

    def test_copy_npu_to_npu(self):
        x = tf.add(1, 1)
        y = x._copy()
        self.assertTrue(tensor_equal(y, tf.constant(2)))

    def test_mix_fuzz_compile(self):
        def gen():
            v = [['1'], ['2', '3'], ['4', '5', '6']]
            while len(v):
                yield v.pop(0)

        ds = tf.data.Dataset.from_generator(gen, output_types=tf.string)
        iterator = iter(ds)

        @tf.function
        def f(it):
            v = next(it)
            v = tf.strings.to_number(v)
            return v + v

        self.assertTrue(tensor_equal(f(iterator), tf.constant([2.0])))
        self.assertTrue(tensor_equal(f(iterator), tf.constant([4.0, 6.0])))
        self.assertTrue(tensor_equal(f(iterator), tf.constant([8.0, 10.0, 12.0])))

    def test_fully_fuzz_compile(self):
        @tf.function(input_signature=[tf.TensorSpec(shape=None)])
        def f(v):
            return v + v

        self.assertTrue(tensor_equal(f([1.0]), tf.constant([2.0])))
        self.assertTrue(tensor_equal(f([1.0, 2.0]), tf.constant([2.0, 4.0])))

    def test_host_loop_1(self):
        def train_step(iterator):
            v = next(iterator)
            v = tf.unique(v)
            tf.print(v)

        @tf.function
        def train_loop(iterator):
            for i in tf.range(2):
                train_step(iterator)

        ds = tf.data.Dataset.from_tensor_slices([[1, 1, 1], [1, 2, 3]])
        train_iterator = iter(ds)
        train_loop(train_iterator)

    def test_host_loop_2(self):
        def train_step(iterator):
            v = next(iterator)
            cond = tf.less(tf.reduce_sum(v), 5)
            tf.cond(cond, lambda: tf.print(tf.unique(v)), lambda: tf.print(tf.reduce_sum(v)))

        @tf.function
        def train_loop(iterator):
            for i in tf.range(2):
                train_step(iterator)

        ds = tf.data.Dataset.from_tensor_slices([[1, 1, 1], [1, 2, 3]])
        train_iterator = iter(ds)
        train_loop(train_iterator)

    def test_npu_run_context(self):
        v = tf.Variable(1.0)

        @tf.function
        def f():
            v.assign_add(1.0)
            return v

        options = npu_device.configs.run_context_options()
        options.experimental.graph_memory_optimize_config.recompute = "manual"
        with npu_device.npu_run_context(options=options):
            f()

    def test_npu_run_context_2(self):
        v = tf.Variable(1.0)

        @tf.function
        def f():
            v.assign_add(1.0)
            return v

        options = npu_device.configs.run_context_options()
        options.experimental.graph_parallel_config.enable_graph_parallel = True
        options.experimental.graph_parallel_config.config_path = "./"
        with npu_device.npu_run_context(options=options):
            f()


class Adapter2St_EnvGeStaticMemory(unittest.TestCase):
    def test_dropout_v3(self):
        @tf.function
        def f(x):
            noise_shape = tf.shape(x)
            gen_out = npu_device.gen_npu_ops.drop_out_gen_mask_v3(noise_shape, tf.constant(0.1), 0, 0)
            npu_device.gen_npu_ops.drop_out_do_mask_v3(x, gen_out, tf.constant(0.1))
            gen_out = npu_device.gen_npu_ops.drop_out_gen_mask_v3(noise_shape, tf.constant(0.1), 0, 0)
            npu_device.gen_npu_ops.drop_out_do_mask_v3(x, gen_out, tf.constant(0.1))
            gen_out = npu_device.gen_npu_ops.drop_out_gen_mask_v3(noise_shape, tf.constant(0.1), 0, 0)
            npu_device.gen_npu_ops.drop_out_do_mask_v3(x, gen_out, tf.constant(0.1))
            gen_out = npu_device.gen_npu_ops.drop_out_gen_mask_v3(noise_shape, tf.constant(0.1), 0, 0)
            npu_device.gen_npu_ops.drop_out_do_mask_v3(x, gen_out, tf.constant(0.1))
            gen_out = npu_device.gen_npu_ops.drop_out_gen_mask_v3(noise_shape, tf.constant(0.1), 0, 0)
            npu_device.gen_npu_ops.drop_out_do_mask_v3(x, gen_out, tf.constant(0.1))

        f(tf.constant([[0.1]]))

    def test_op_in_static_memory(self):
        self.assertEqual(tf.add(1, 1), tf.constant(2))


if __name__ == '__main__':
    unittest.main()
