#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# ==============================================================================

"""Make adapter 1.x compat for tf2"""

import os
import sys
import shutil
import re

REPLACE_RULES = dict()
REPLACE_RULES['from npu_bridge'] = 'from npu_device.compat.v1'
REPLACE_RULES['import tensorflow'] = 'import tensorflow.compat.v1'
REPLACE_RULES["@ops.RegisterGradient('HcomAllReduce')"] = ''
REPLACE_RULES[
    'from tensorflow.distribute.experimental import ParameterServerStrategy'] = 'from tensorflow.python.distribute.parameter_server_strategy import ParameterServerStrategyV1 as ParameterServerStrategy'
REPLACE_RULES[
    'from tensorflow.contrib.distribute import DistributeConfig'] = 'from tensorflow.python.distribute.distribute_config import DistributeConfig'
REPLACE_RULES["from npu_device.compat.v1.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer"] = ''
REPLACE_RULES["from npu_device.compat.v1.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager"] = ''
REPLACE_RULES["from npu_device.compat.v1.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager"] = ''

REGEXP_RULES = dict()
REGEXP_RULES['import npu_bridge$'] = 'from npu_device.compat import v1 as npu_bridge'

FILE_REPLACED = (
    'helper/helper.py',
)

FILE_REMOVED = (
    'estimator/npu/npu_loss_scale_manager.py',
    'estimator/npu/npu_loss_scale_optimizer.py'
)

REPLACE_BASE = os.path.join(os.path.dirname(__file__), 'replacement')


def make_compat(root, absf):
    fn = os.path.relpath(absf, root)
    if fn in FILE_REMOVED:
        print('>>> File removed', flush=True)
        os.remove(absf)
    elif fn in FILE_REPLACED:
        print('>>> File replaced with', os.path.join('replacement', fn), flush=True)
        shutil.copyfile(os.path.join(REPLACE_BASE, fn), absf)
    else:
        with open(absf, 'r') as f:
            lines = f.readlines()
        with open(absf, 'w+') as f:
            for line in lines:
                origin_line = line
                for k, v in REPLACE_RULES.items():
                    line = line.replace(k, v, 1)
                for k, v in REGEXP_RULES.items():
                    line = re.sub(k, v, line)
                f.writelines(line)
                if origin_line != line:
                    print('>>> Replace', origin_line, "with", line, flush=True)


def main():
    tree = sys.argv[1]
    for path, _, files in os.walk(tree):
        for fn in files:
            if fn.endswith('.py'):
                print("--- Processing", os.path.join(path, fn), '---', flush=True)
                make_compat(tree, os.path.join(path, fn))


if __name__ == '__main__':
    main()
