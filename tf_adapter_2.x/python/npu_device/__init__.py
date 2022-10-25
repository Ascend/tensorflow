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


from npu_device.npu_device import open
from npu_device.npu_device import npu_compat_function
from npu_device.npu_device import gen_npu_ops
from npu_device.npu_device import global_options
from npu_device.npu_device import set_npu_loop_size
from npu_device.npu_device import npu_run_context
from npu_device.npu_device import set_device_sat_mode

from npu_device.utils.scope import keep_dtype_scope
from npu_device.utils.scope import npu_recompute_scope

from npu_device._api import distribute
from npu_device._api import train
from npu_device._api import ops
from npu_device._api import compat
from npu_device._api import configs
