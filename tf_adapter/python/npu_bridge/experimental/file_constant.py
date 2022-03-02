#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
from tensorflow.python.framework import dtypes
from npu_bridge.helper import helper

gen_npu_cpu_ops = helper.get_gen_ops();

## 提供FileConstant功能
#  @param file_id string 类型
#  @param shape list(int) 类型
#  @param dtype float, float16, int8, int16, uint16,
#               uint8, int32, int64, uint32, uint64, bool, double 类型
#  @return y float, float16, int8, int16, uint16,
#            uint8, int32, int64, uint32, uint64, bool, double 类型
def file_constant(file_id, shape, dtype, name=None):
    """ file constant. """
    result = gen_npu_cpu_ops.file_constant(
        file_id=file_id,
        shape=shape,
        dtype=dtype,
        name=name)
    return result