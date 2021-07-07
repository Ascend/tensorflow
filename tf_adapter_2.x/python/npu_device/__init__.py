#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from npu_device.npu_device import open
from npu_device.npu_device import never_nested_function
from npu_device.npu_device import gen_npu_ops
from npu_device.npu_device import global_options

from npu_device.utils.scope import keep_dtype_scope

from npu_device._api import distribute
from npu_device._api import train
from npu_device._api import ops
