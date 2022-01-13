#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""import files when initiating"""

from npu_device.distribute.hccl import all_reduce
from npu_device.distribute.hccl import broadcast
from npu_device.distribute.hccl import npu_distributed_keras_optimizer_wrapper
from npu_device.distribute.hccl import shard_and_rebatch_dataset
from npu_device.distribute.weight_update_grouping import grouping_gradients_apply
from npu_device.distribute.weight_update_grouping import grouping_broadcast
