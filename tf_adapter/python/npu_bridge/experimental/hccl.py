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

"""NPU hccl functions"""

import os
import ctypes
from npu_bridge.estimator.npu import util as util_lib


hccl_graph_adp_ctypes = ctypes.CDLL('libhcom_graph_adaptor.so')


def c_str(string):
    return ctypes.c_char_p(string.encode('utf-8'))


def get_actual_rank_size(group="hccl_world_group"):
    c_group = c_str(group)
    c_rank_size = ctypes.c_uint()
    ret = hccl_graph_adp_ctypes.HcomGetActualRankSize(c_group, ctypes.byref(c_rank_size))
    if ret != 0:
        raise ValueError('get actual rank size error.')
    return c_rank_size.value


def get_user_rank_size():
    rank_size = int(util_lib.get_ranksize())
    return rank_size


def get_user_rank_id():
    rank_id = int(os.getenv('RANK_ID'))
    return rank_id