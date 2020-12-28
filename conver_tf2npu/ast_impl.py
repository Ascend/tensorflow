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
import ast
import util_global
from util import log_success_report
from util import log_migration_report

def attribute(node):
    log_success_report(getattr(node, "lineno", "None"), node.attr)
    node = ast.Name(id=util_global.get_value(node.attr)[0], ctx=ast.Load())
    util_global.set_value('need_conver', True)
    return node

def import_from(node):
    if node.module != None:
        values = node.module.split(".")
        if "keras" in values:
            log_migration_report(getattr(node, "lineno", "None"), "keras")
            util_global.set_value('need_conver', True)

def ast_import(node):
    for value in node.names:
        if isinstance(value, ast.alias):
            values = value.name.split(".")
            if "keras" in values:
                log_migration_report(getattr(node, "lineno", "None"), "keras")
                util_global.set_value('need_conver', True)

def ast_function_def(node):
    log_success_report(getattr(node, "lineno", "None"), node.name)
    node.body = [ast.Return(value=ast.Call(
                                            func=ast.Attribute(value=ast.Name(id=util_global.get_value(node.name)[0],
                                                               ctx=ast.Load()), attr='gelu',
                                                               ctx=ast.Load()),
                                            args=[ast.Name(id='x', ctx=ast.Load())],
                                            keywords=[]))]

    util_global.set_value('need_conver', True)
    return node

def ast_call(node):
    if isinstance(node.func, ast.Attribute):
        if len(node.args) > 0:
            if isinstance(node.args[0], ast.Call):
                if isinstance(node.args[0].func, ast.Attribute):
                    if node.args[0].func.attr == 'BroadcastGlobalVariablesHook':
                        log_success_report(getattr(node, "lineno", "None"), 'BroadcastGlobalVariablesHook')
                        node.func = ast.Name(id=util_global.get_value('BroadcastGlobalVariablesHook')[0], ctx=ast.Load())
                        node.args = []
                        util_global.set_value('need_conver', True)
    if isinstance(node.func, ast.Attribute) and node.func.attr == 'shard':
        log_success_report(getattr(node, "lineno", "None"), 'shard')
        node.args = [ast.Call(func=ast.Name(id='get_rank_size', ctx=ast.Load()), args=[], keywords=[]),
                     ast.Call(func=ast.Name(id='get_rank_id', ctx=ast.Load()), args=[], keywords=[])]
        util_global.set_value('need_conver', True)
    if isinstance(node.func, ast.Attribute) and node.func.attr == 'dropout':
        if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'nn':
            log_success_report(getattr(node, "lineno", "None"), 'dropout')
            node.func=ast.Attribute(value=ast.Name(id='npu_ops', ctx=ast.Load()), attr='dropout', ctx=ast.Load())
            keywords_new = []
            for keyword in node.keywords:
                if keyword.arg != 'rate':
                    keywords_new.append(keyword)
            node.keywords = keywords_new
            util_global.set_value('need_conver', True)
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'batch' or node.func.attr == 'map_and_batch'):
        exist = False
        for keyword in node.keywords:
            if keyword.arg == 'drop_remainder':
                exist = True
                if ((isinstance(keyword.value, ast.NameConstant) and keyword.value.value != True) or
                   (not isinstance(keyword.value, ast.NameConstant))):
                    log_success_report(getattr(node, "lineno", "None"), node.func.attr)
                    keyword.value = ast.NameConstant(value=True)
                    util_global.set_value('need_conver', True)
        if not exist:
            log_success_report(getattr(node, "lineno", "None"), node.func.attr)
            keyword = ast.keyword(arg='drop_remainder', value=ast.NameConstant(value=True))
            node.keywords.insert(0, keyword)
            util_global.set_value('need_conver', True)
    if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
        node.func.value.id == 'tf' and node.func.attr == 'device'):
        log_success_report(getattr(node, "lineno", "None"), node.func.attr)
        node.args = [ast.Str(s='/cpu:0')]
        util_global.set_value('need_conver', True)
    return node

def insert_npu_import(r_node):
    log_success_report('0', 'import')
    npu_alias = ast.alias(name='*', asname=None)
    npu_import = ast.ImportFrom(module='npu_bridge.npu_init', names=[npu_alias], level=0)
    r_node.body.insert(0, npu_import)

def ast_assign(node):
    if isinstance(node.value, ast.Call):
        if isinstance(node.value.func, ast.Attribute):
            if isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == 'max_pooling2d':
                    log_success_report(getattr(node, "lineno", "None"), node.value.func.attr)
                    util_global.set_value('need_conver', True)
                    elts_new = []
                    for target in node.targets:
                        elts_new.append(target)
                    elts_new.append(ast.Name(id='argmax', ctx=ast.Load()))
                    node.targets=[ast.Tuple(elts=elts_new)]

                    keywords_new = []
                    for keyword in node.value.keywords:
                        if keyword.arg == 'inputs':
                            keyword_new = ast.keyword(arg='input', value=keyword.value)
                            keywords_new.append(keyword_new)
                        if keyword.arg == 'pool_size':
                            elts_new = [ast.Num(n=1), keyword.value, keyword.value, ast.Num(n=1)]
                            keyword_new = ast.keyword(arg='ksize', value=ast.Tuple(elts=elts_new))
                            keywords_new.append(keyword_new)
                        if keyword.arg == 'strides':
                            elts_new = [ast.Num(n=1), keyword.value, keyword.value, ast.Num(n=1)]
                            keyword_new = ast.keyword(arg='strides', value=ast.Tuple(elts=elts_new))
                            keywords_new.append(keyword_new)
                        if keyword.arg == 'padding' or keyword.arg == 'data_format':
                            keywords_new.append(keyword)
                    func_new = ast.Attribute(value=ast.Attribute(value=ast.Attribute(value=ast.Attribute(value=ast.Name(id='tf', ctx=ast.Load()), attr='compat', ctx=ast.Load()), attr='v1', ctx=ast.Load()), attr='nn', ctx=ast.Load()), attr='max_pool_with_argmax', ctx=ast.Load())
                    node.value = ast.Call(func=func_new,
                                          args=[],
                                          keywords=keywords_new)
    return node

# Format printing for locate
def node_tree(node:str):
    str2list = list(node.replace(' ', ''))
    count = 0
    for i, e in enumerate(str2list):
        if e == '(':
            count += 1
            str2list[i] = '(\n{}'.format('|   ' * count)
        elif e == ')':
            count -= 1
            str2list[i] = '\n{})'.format('|   ' * count)
        elif e == ',':
            str2list[i] = ',\n{}'.format('|   ' * count)
        elif e == '[':
            count += 1
            str2list[i] = '[\n{}'.format('|   ' * count)
        elif e == ']':
            count -= 1
            str2list[i] = '\n{}]'.format('|   ' * count)
    return ''.join(str2list)