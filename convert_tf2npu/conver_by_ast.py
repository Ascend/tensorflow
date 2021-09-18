#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
import os
import sys
import ast
import pasta
import util_global
import subprocess
from file_op import write_output_after_conver
from file_op import write_report_after_conver
from file_op import scan_file
from util import *
from ast_impl import *
from visit_by_ast import get_tf_api

class ConverByAst(ast.NodeTransformer):
    def generic_visit(self, node):
        ast.NodeTransformer.generic_visit(self, node)
        return node
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "keras":
            util_global.set_value('is_keras_net', True)
        if node.attr in util_global.get_value('hvd'):
            distributed_mode = util_global.get_value("distributed_mode", "")
            if isinstance(node.value, ast.Name) and 'hvd' in str(node.value.id):
                if distributed_mode == "tf_strategy" or distributed_mode == "":
                    log_strategy_distributed_mode_error(node)
                    return node
                return attribute(node)
        return node

    def visit_FunctionDef(self, node):
        if node.name == 'gelu':
            return ast_function_def(node)
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        node = ast_call(node)
        return node

    def visit_ImportFrom(self, node):
        self.generic_visit(node)
        node = import_from(node)
        return node

    def visit_Import(self, node):
        self.generic_visit(node)
        node = ast_import(node)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        ast_if(node)
        return node

def conver(r_node, out_path_dst, file_name):
    if file_name != "__init__.py":
        insert_npu_import(r_node)
    if util_global.get_value('use_keras_dropout', False):
        insert_keras_dropout_import(r_node)
    distributed_mode = util_global.get_value('distributed_mode', "")
    if not util_global.get_value('has_main_func', False) and (util_global.get_value('has_hvd_api', False)
        or util_global.get_value('is_keras_net', False)) and  not util_global.get_value('main', ""):
        log_warning_main_arg_not_set()
    if distributed_mode == "horovod" and util_global.get_value('is_main_file', False):
        insert_npu_resource_init(r_node)
        insert_npu_resource_shutdown(r_node)
    if util_global.get_value('is_main_file', False) and util_global.get_value('is_keras_net', False):
        insert_keras_sess_npu_config(r_node)
        insert_keras_sess_close(r_node)
    dst_content = pasta.dump(r_node)
    write_output_after_conver(os.path.join(util_global.get_value('output'), out_path_dst, file_name), dst_content)

def conver_ast(path, out_path_dst, file_name):
    util_global.set_value('need_conver', False)
    util_global.set_value('is_keras_net', False)
    util_global.set_value('has_hvd_api', False)
    util_global.set_value('is_main_file', False)
    util_global.set_value('has_main_func', False)
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
        subprocess.run(["cd", "."], shell=True)
        print("".join(["\033[1;31mERROR\033[0m:", content]))
        return

    sys.setrecursionlimit(10000)
    visitor = ConverByAst()
    visitor.visit(r_node)
    ast.fix_missing_locations(r_node)

    (api, lineno) = get_tf_api(os.path.join(path, file_name))
    if len(api) == 0:
        print("No Tensorflow module is imported in script {}.".format(file_name))
    scan_file(path, file_name, api, lineno)

    if util_global.get_value('need_conver', False):
        conver(r_node, out_path_dst, file_name)

    if file_name.endswith("a.py"):
        write_report_after_conver("only_for_test", file_name, node_tree(ast.dump(r_node)))