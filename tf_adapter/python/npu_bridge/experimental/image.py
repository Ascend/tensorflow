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

def decode_image(contents, channels=0, dtype=dtypes.uint8, expand_animations=True):
    """
    Decode image.

    :param contents: string 类型.
    :param channels int 类型.
    :param expand_animations bool 类型.
    :return image
    """
    return gen_npu_cpu_ops.decode_image_v3(
        contents=contents,
        channels=channels,
        dtype=dtype,
        expand_animations=expand_animations)