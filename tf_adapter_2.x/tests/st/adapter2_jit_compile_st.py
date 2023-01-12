#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

import unittest
import tensorflow as tf
import npu_device

from npu_device.npu_device import stupid_repeat
from tensorflow.python.eager import context

npu = npu_device.open().as_default()


def tensor_equal(t1, t2):
    return True


class Adapter2JitCompileSt(unittest.TestCase):
    def test_mix_jit_compile_fuzz_compile(self):
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

if __name__ == '__main__':
    unittest.main()
