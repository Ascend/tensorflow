#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless REQUIRED by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""functions for global varabiles manegement"""

import json
import time
import os
import collections

ParamConfig = collections.namedtuple('ParamConfig', \
    ['short_opts', 'long_opts', 'opt_err_prompt', 'opt_help', 'support_list_filename', 'main_arg_not_set_promt'])


def init():
    global _global_dict
    _global_dict = {}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mappings', 'ast.json')) as f:
        load_dict = json.load(f)
        items = load_dict.items()
        for key, value in items:
            set_value(key, value)
    value = "_npu_" + time.strftime('%Y%m%d%H%M%S')
    set_value('timestap', value)


def set_value(key, value):
    """Set value for global dictionary"""
    _global_dict[key] = value


def get_value(key, def_value=None):
    """Get value by key from global dictionary"""
    try:
        return _global_dict[key]
    except KeyError:
        return def_value
