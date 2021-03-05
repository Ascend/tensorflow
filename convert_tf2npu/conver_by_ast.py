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
import astunparse
import util_global
from file_op import write_output_after_conver
from file_op import write_report_after_conver
from file_op import scan_file
from util import log_success_report
from util import log_migration_report
from ast_impl import attribute
from ast_impl import node_tree
from ast_impl import insert_config_pb2_import
from ast_impl import insert_npu_init_func
from ast_impl import insert_NPUBroadcastGlobalVariablesHook_import
from ast_impl import insert_npu_hooks_append_func
from ast_impl import insert_npu_run_config_func
from ast_impl import insert_npu_session_config_func
from ast_impl import insert_RewriterConfig_import
from ast_impl import insert_npu_import
from ast_impl import insert_npu_tf_opt_func
from ast_impl import insert_npu_keras_opt_func
from ast_impl import insert_empty_hook
from ast_impl import insert_os_import
from ast_impl import import_from
from ast_impl import ast_import
from ast_impl import ast_function_def
from ast_impl import ast_call
from ast_impl import ast_assign
from visit_by_ast import get_tf_api

class ConverByAst(ast.NodeTransformer):
    def generic_visit(self, node):
        ast.NodeTransformer.generic_visit(self, node)
        return node
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr in util_global.get_value('hvd'):
            if isinstance(node.value, ast.Name):
                if 'hvd' in str(node.value.id):
                    return attribute(node)
            if isinstance(node.value, ast.Attribute):
                if 'hvd' in str(node.value.attr):
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
        import_from(node)
        self.generic_visit(node)
        return node

    def visit_Import(self, node):
        ast_import(node)
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        for target in node.targets:
            if (isinstance(target, ast.Name) and target.id == 'global_jit_level') or (isinstance(target, ast.Attribute) and target.attr == 'global_jit_level'):
                return ast_assign(node)

        ast_assign(node)
        self.generic_visit(node)
        return node

def conver_ast(path, out_path_dst, file_name):
    util_global.set_value('need_conver', False)
    util_global.set_value('import_config_pb2', False)
    util_global.set_value('insert_npu_init_func', False)
    util_global.set_value('insert_estimator_add_hook_func', False)
    util_global.set_value('insert_npu_hooks_append', False)
    util_global.set_value('import_NPUBroadcastGlobalVariablesHook', False)
    util_global.set_value('insert_npu_run_config_func', False)
    util_global.set_value('insert_npu_session_config_func', False)
    util_global.set_value('import_RewriterConfig', False)
    util_global.set_value('insert_npu_tf_opt_func', False)
    util_global.set_value('insert_npu_keras_opt_func', False)
    util_global.set_value('insert_empty_hook', False)
    util_global.set_value('import_os', False)
    with open(os.path.join(path, file_name), "r", encoding='utf-8') as file:
        source = file.read()
    try:
        r_node = ast.parse(source)
    except Exception as e:
        print(repr(e))
        return

    sys.setrecursionlimit(10000)
    visitor = ConverByAst()
    visitor.visit(r_node)
    ast.fix_missing_locations(r_node)

    (api, lineno) = get_tf_api(os.path.join(path, file_name))
    if len(api) == 0:
        print("No Tensorflow module is imported in script {}.".format(file_name))
    scan_file(file_name, api, lineno)

    if util_global.get_value('need_conver', False):
        insert_npu_import(r_node)
        if util_global.get_value('insert_npu_hooks_append', False):
            insert_npu_hooks_append_func(r_node)
        if util_global.get_value('import_NPUBroadcastGlobalVariablesHook', False):
            insert_NPUBroadcastGlobalVariablesHook_import(r_node)
        if util_global.get_value('insert_npu_run_config_func', False):
            insert_npu_run_config_func(r_node)
        if util_global.get_value('insert_npu_session_config_func', False):
            insert_npu_session_config_func(r_node)
        if util_global.get_value('import_config_pb2', False):
            insert_config_pb2_import(r_node)
        if util_global.get_value('insert_npu_init_func', False):
            insert_npu_init_func(r_node)
        if util_global.get_value('import_RewriterConfig', False):
            insert_RewriterConfig_import(r_node)
        if util_global.get_value('insert_npu_tf_opt_func', False):
            insert_npu_tf_opt_func(r_node)
        if util_global.get_value('insert_npu_keras_opt_func', False):
            insert_npu_keras_opt_func(r_node)
        if util_global.get_value('insert_empty_hook', False):
            insert_empty_hook(r_node)
        if util_global.get_value('import_os', False):
            insert_os_import(r_node)
        dst_content = astunparse.unparse(r_node)
        write_output_after_conver(os.path.join(util_global.get_value('output'), out_path_dst, file_name), dst_content)

    if file_name.endswith("a.py"):
        write_report_after_conver("only_for_test", file_name, node_tree(ast.dump(r_node)))