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

"""Basic class to transform ast node"""

import os
import sys
import ast
import pasta
import util_global
from file_op import scan_file
from ast_impl import ConverByAst
from ast_impl import conver
import visit_by_ast


def clean_up_global():
    util_global.set_value('need_conver', False)
    util_global.set_value('is_keras_net', False)
    util_global.set_value('has_hvd_api', False)
    util_global.set_value('is_main_file', False)
    util_global.set_value('has_main_func', False)
    util_global.set_value('need_import_npu_callbacks_func', False)
    util_global.set_value('need_import_npu_broadcast_func', False)
    util_global.set_value('need_import_experimental_loss_scale_optimizer', False)


def conver_ast(path, out_path_dst, file_name):
    """Convert script by python ast"""
    clean_up_global()
    if os.path.join(path, file_name) == util_global.get_value('main', ""):
        util_global.set_value('is_main_file', True)
    with open(os.path.join(path, file_name), "r", encoding='utf-8') as file:
        source = file.read() + "\n"
    try:
        r_node = pasta.parse(source)
    except Exception as e:
        print(repr(e))
        content = ("There is a format problem in the script, please check the python code "
                   "specification or whether it is converted to a linux file through 'dos2unix'")
        os.system("cd .")
        print("".join(["\033[1;31mERROR\033[0m:", content]))
        return

    sys.setrecursionlimit(10000)
    visitor = ConverByAst()
    visitor.visit(r_node)
    ast.fix_missing_locations(r_node)

    (api, lineno) = visit_by_ast.get_tf_api(os.path.join(path, file_name))
    if len(api) == 0:
        print("No Tensorflow module is imported in script {}.".format(file_name))
    scan_file(path, file_name, api, lineno)

    if util_global.get_value('need_conver', False):
        conver(r_node, out_path_dst, file_name)

