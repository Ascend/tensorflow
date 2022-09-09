#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
"""Python wrappers for Datasets."""

from npu_device.compat.v1.experimental.dataset_ops import map as tf1_map
from npu_device.compat.v1.experimental.dataset_ops import map_and_batch as tf1_map_and_batch
import npu_device as npu
npu.open().as_default()


def map(map_func, num_parallel_calls=None, num_parallel_npu=28,
    deterministic=None, output_device="cpu"):
    """Map function for create a MapDatasetOp."""

    return tf1_map(map_func=map_func,
                   num_parallel_calls=num_parallel_calls,
                   num_parallel_npu=num_parallel_npu,
                   deterministic=deterministic,
                   output_device=output_device)


def map_and_batch(map_func,
                  batch_size,
                  num_parallel_batches=None,
                  drop_remainder=False,
                  num_parallel_calls=None,
                  num_parallel_npu=28,
                  output_device="cpu"):
    """Fused implementation of `map` and `batch`."""

    return tf1_map_and_batch(map_func=map_func,
                             batch_size=batch_size,
                             num_parallel_batches=num_parallel_batches,
                             drop_remainder=drop_remainder,
                             num_parallel_calls=num_parallel_calls,
                             num_parallel_npu=num_parallel_npu,
                             output_device=output_device)
