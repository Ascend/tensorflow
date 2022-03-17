#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorflow.python.framework import ops
from npu_bridge.helper import helper

npu_aicore_ops = helper.get_gen_ops()


def getshape(x, name=None):
    """
    Args:
        x: A tensor with type is float.
    Returns:
        A tensor.
    """
    x = ops.convert_to_tensor(x, name="x")
    result = npu_aicore_ops.get_shape(x)
    return result