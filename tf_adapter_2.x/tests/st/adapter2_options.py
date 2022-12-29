#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
#
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
#

import io
import sys
from contextlib import contextmanager
import unittest
from npu_device.configs.npu_config import NpuConfig


@contextmanager
def stub_print():
    origin_cout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield sys.stdout
    finally:
        sys.stdout = origin_cout


class Adapter2Options(unittest.TestCase):
    # check default value of options
    def test_0_check_deprecated_option_default_none(self):
        with stub_print() as result:
            config = NpuConfig()
            init_options = config.as_dict()
            # 不设置弃用选项，没有弃用参数打印
            self.assertEqual(result.getvalue().strip(), '')
            # 不设置弃用选项，参数不传递
            self.assertTrue('variable_format_optimize' not in init_options, True)
            self.assertTrue('op_select_implmode' not in init_options, True)
            self.assertTrue('optypelist_for_implmode' not in init_options, True)
            self.assertTrue('op_debug_level' not in init_options, True)
    
    def test_1_set_deprecated_option_error(self):
        config = NpuConfig()
        try:
            config.variable_format_optimize = 88
        except ValueError as e:
            err = "'88' not in optional list [True, False]"
            self.assertEqual(err, str(e))
        try:
            config.op_select_implmode = 88
        except ValueError as e:
            err = "'88' not in optional list ['high_performance', 'high_precision']"
            self.assertEqual(err, str(e))

    def test_2_logging_when_config_with_deprecated(self):
        with stub_print() as result:
            config = NpuConfig()
            config.variable_format_optimize = False
            config.as_dict()
            expect = f"Option 'variable_format_optimize' is deprecated and will be removed in future version. "\
                     f"Please use 'None' instead."
            self.assertEqual(expect, result.getvalue().strip())

    def test_3_logging_when_config_with_deprecated(self):
        with stub_print() as result:
            config = NpuConfig()
            config.op_select_implmode = 'high_precision'
            config.as_dict()
            expect = f"Option 'op_select_implmode' is deprecated and will be removed in future version. "\
                     f"Please use 'op_precision_mode' instead."
            self.assertEqual(expect, result.getvalue().strip())

    def test_4_logging_when_config_with_deprecated(self):
        with stub_print() as result:
            config = NpuConfig()
            config.optypelist_for_implmode = 'cf.ini'
            config.as_dict()
            expect = f"Option 'optypelist_for_implmode' is deprecated and will be removed in future version. "\
                     f"Please use 'op_precision_mode' instead."
            self.assertEqual(expect, result.getvalue().strip())

    def test_5_logging_when_config_with_deprecated(self):
        with stub_print() as result:
            config = NpuConfig()
            config.op_debug_level = 2
            config.as_dict()
            expect = f"Option 'op_debug_level' is deprecated and will be removed in future version. "\
                     f"Please use 'op_debug_config' instead."
            self.assertEqual(expect, result.getvalue().strip())


if __name__ == '__main__':
    unittest.main()