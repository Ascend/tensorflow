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
npu_device.global_options().aoe_config.aoe_mode = "1"
npu_device.global_options().aoe_config.work_path = "./"
npu = npu_device.open().as_default()
npu.workers_num = 2  # mock run in 2P env

def tensor_equal(t1, t2):
    return True

@tf.function
def foo_add(v1, v2):
    return v1 + v2

class Adapter2AoeSt(unittest.TestCase):
    def test_mix_resource(self):
        with context.device("/job:localhost/replica:0/task:0/device:CPU:0"):
            x = tf.Variable(1)
        y = tf.Variable(1)
        self.assertTrue(tensor_equal(foo_add(x, y), tf.constant(2)))

if __name__ == '__main__':
    unittest.main()
