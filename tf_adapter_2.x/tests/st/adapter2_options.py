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
            expect = f"[warning][tf_adapter] Option 'variable_format_optimize' is deprecated "\
                     f"and will be removed in future version. Please do not configure this option in the future."
            self.assertEqual(expect, result.getvalue().strip())

    def test_3_logging_when_config_with_deprecated(self):
        with stub_print() as result:
            config = NpuConfig()
            config.op_select_implmode = 'high_precision'
            config.as_dict()
            expect = f"[warning][tf_adapter] Option 'op_select_implmode' is deprecated "\
                     f"and will be removed in future version. Please use 'op_precision_mode' instead."
            self.assertEqual(expect, result.getvalue().strip())

    def test_4_logging_when_config_with_deprecated(self):
        with stub_print() as result:
            config = NpuConfig()
            config.optypelist_for_implmode = 'cf.ini'
            config.as_dict()
            expect = f"[warning][tf_adapter] Option 'optypelist_for_implmode' is deprecated "\
                     f"and will be removed in future version. Please use 'op_precision_mode' instead."
            self.assertEqual(expect, result.getvalue().strip())

    def test_5_logging_when_config_with_deprecated(self):
        with stub_print() as result:
            config = NpuConfig()
            config.op_debug_level = 2
            config.as_dict()
            expect = f"[warning][tf_adapter] Option 'op_debug_level' is deprecated "\
                     f"and will be removed in future version. Please use 'op_debug_config' instead."
            self.assertEqual(expect, result.getvalue().strip())

    def test_6_set_option_deterministic(self):
        config = NpuConfig()
        config.deterministic = 1
        options = config.as_dict()
        self.assertTrue(options['deterministic'], True)
        try:
            config.deterministic = 88
        except ValueError as e:
            err = "'88' not in optional list [0, 1]"
            self.assertEqual(err, str(e))

    def test_7_set_option_op_precision_mode(self):
        config = NpuConfig()
        config.op_precision_mode = "op_precision.ini"
        options = config.as_dict()
        print(f"op_precision_mode: '{options['op_precision_mode']}'")
        self.assertTrue(options['op_precision_mode'] == "op_precision.ini", True)

    def test_8_set_option_hcom_parallel(self):
        config = NpuConfig()
        options = config.as_dict()
        self.assertTrue(options['hcom_parallel'], True)
        config.hcom_parallel = False
        options = config.as_dict()
        self.assertTrue(options['hcom_parallel'], False)

    def test_9_set_option_graph_compiler_cache_dir(self):
        config = NpuConfig()
        options = config.as_dict()
        self.assertTrue('graph_compiler_cache_dir' not in options, True)
        config.graph_compiler_cache_dir = "./st_graph_cache_dir"
        options = config.as_dict()
        self.assertEqual(options['graph_compiler_cache_dir'], "./st_graph_cache_dir")

    def test_10_set_option_graph_slice(self):
        config = NpuConfig()
        config.experimental.graph_memory_optimize_config.graph_slice = "auto"
        options = config.as_dict()
        self.assertEqual(options['graph_slice'], "auto")

    def test_10_set_jit_compile_option_error(self):
        config = NpuConfig()
        try:
            config.jit_compile = "true"
        except ValueError as e:
            err = "'true' not in optional list ['True', 'False', 'Auto']"
            self.assertEqual(err, str(e))
        try:
            config.jit_compile = "false"
        except ValueError as e:
            err = "'false' not in optional list ['True', 'False', 'Auto']"
            self.assertEqual(err, str(e))
        try:
            config.jit_compile = "auto"
        except ValueError as e:
            err = "'auto' not in optional list ['True', 'False', 'Auto']"
            self.assertEqual(err, str(e))

if __name__ == '__main__':
    unittest.main()