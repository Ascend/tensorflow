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
    if isinstance(node.func, ast.Attribute) and (node.func.attr == "get_distribution_strategy" or
        node.func.attr == "MirroredStrategy" or node.func.attr == "MultiWorkerMirroredStrategy"):
        log_success_report(getattr(node, "lineno", "None"), node.func.attr)
        node.func = ast.Attribute(value=ast.Name(id="npu_strategy", ctx=ast.Load()),
                                  attr="NPUStrategy", ctx=ast.Load())
        node.keywords = []
        node.args = []
        util_global.set_value('need_conver', True)
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'AdviceProto'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.AdviceProto')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'Checker'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.AdviceProto.Checker')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'CheckersEntry'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.AdviceProto.CheckersEntry')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'GraphNodeProto'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.GraphNodeProto')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'InputShapesEntry'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.GraphNodeProto.InputShapesEntry')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'MultiGraphNodeProto'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.MultiGraphNodeProto')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'OpLogProto'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.OpLogProto')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'IdToStringEntry'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.OpLogProto.IdToStringEntry')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'ProfileOptionBuilder'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.ProfileOptionBuilder')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'advise'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.advise')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'profile'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.profile')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'write_op_log'):
        log_success_report(getattr(node, 'lineno', 'None'), 'tf.profiler.write_op_log')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'TPUEstimator' and (node.func.value.attr == 'tpu')):
        for keyword in node.keywords:
            if (keyword.arg == 'eval_on_tpu') or (keyword.arg == 'use_tpu') or (keyword.arg == 'export_to_tpu'):
                if (not isinstance(keyword.value, ast.NameConstant)) or (isinstance(keyword.value, ast.NameConstant) and (keyword.value.value != False)):
                    log_success_report(getattr(node, 'lineno', 'None'), 'TPUEstimator(' + keyword.arg + '=*)')
                    keyword.value = ast.NameConstant(value=False)
                    util_global.set_value('need_conver', True)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'RunConfig'):
        for keyword in node.keywords:
            if keyword.arg == 'session_config':
                log_success_report(getattr(node, 'lineno', 'None'), 'add_npu_config')
                keyword.value = ast.Call(func=ast.Name(id='npu_init', ctx=ast.Load()), args=[keyword.value], keywords=[])
                util_global.set_value('insert_npu_init_func', True)
                return node
    return node

def insert_npu_import(r_node):
    npu_alias = ast.alias(name='*', asname=None)
    npu_import = ast.ImportFrom(module='npu_bridge.npu_init', names=[npu_alias], level=0)
    for i in range(0, 5):
        if isinstance(r_node.body[i], ast.Import):
            r_node.body.insert(i, npu_import)
            log_success_report(i, "import")
            break
        elif isinstance(r_node.body[i], ast.ImportFrom):
            if r_node.body[i].module != "__future__":
                r_node.body.insert(i, npu_import)
                log_success_report(i, "import")
                break

def insert_npu_init_func(r_node):
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and (isinstance(r_node.body[n], ast.ImportFrom) or isinstance(r_node.body[n], ast.Import)):
        n += 1

    if n < lenline:
        custom_op_assign_node = ast.Assign(
            targets=[
                ast.Name(id='custom_op', ctx=ast.Store())
            ],
            value=ast.Call(args=[], keywords=[],
                func=ast.Attribute(attr='add', ctx=ast.Load(),
                    value=ast.Attribute(attr='custom_optimizers', ctx=ast.Load(),
                        value=ast.Attribute(attr='rewrite_options', ctx=ast.Load(),
                            value=ast.Attribute(attr='graph_options', ctx=ast.Load(),
                                value=ast.Name(id='config_proto', ctx=ast.Load())))))))

        custom_op_name_assign_node = ast.Assign(
            targets=[
                ast.Attribute(value=ast.Name(id='custom_op', ctx=ast.Load()), attr='name', ctx=ast.Store())
            ],
            value=ast.Str(s='NpuOptimizer'))

        custom_op_pm_enable_data_pre_proc_assign_node = ast.Assign(
            targets=[
                ast.Attribute(attr='b', ctx=ast.Store(),
                    value=ast.Subscript(slice=ast.Index(value=ast.Str(s='enable_data_pre_proc')), ctx=ast.Load(),
                        value=ast.Attribute(attr='parameter_map', ctx=ast.Load(),
                            value=ast.Name(id='custom_op', ctx=ast.Load()))))
            ],
            value=ast.NameConstant(value=True))

        custom_op_pm_use_off_line_assign_node = ast.Assign(
            targets=[
                ast.Attribute(attr='b', ctx=ast.Store(),
                    value=ast.Subscript(slice=ast.Index(value=ast.Str(s='use_off_line')), ctx=ast.Load(),
                        value=ast.Attribute(attr='parameter_map', ctx=ast.Load(),
                            value=ast.Name(id='custom_op', ctx=ast.Load()))))
            ],
            value=ast.NameConstant(value=True))

        remapping_assign_node = ast.Assign(
            targets=[
                ast.Attribute(attr='remapping', ctx=ast.Store(),
                    value=ast.Attribute(attr='rewrite_options', ctx=ast.Load(),
                        value=ast.Attribute(attr='graph_options', ctx=ast.Load(),
                            value=ast.Name(id='config_proto', ctx=ast.Load()))))
            ],
            value=ast.Attribute(attr='OFF', ctx=ast.Load(),
                value=ast.Attribute(attr='rewrite_config_pb2', ctx=ast.Load(),
                    value=ast.Attribute(attr='protobuf', ctx=ast.Load(),
                        value=ast.Attribute(attr='core', ctx=ast.Load(),
                            value=ast.Name(id='tf', ctx=ast.Load()))))))

        return_node = ast.Return(value=ast.Name(id='config_proto', ctx=ast.Load()))

        r_node.body.insert(n, ast.FunctionDef(
            name='npu_init',
            args=ast.arguments(
                args=[
                    ast.arg(arg='config_proto', annotation=None)
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]),
            body=[
                custom_op_assign_node,
                custom_op_name_assign_node,
                custom_op_pm_enable_data_pre_proc_assign_node,
                custom_op_pm_use_off_line_assign_node,
                remapping_assign_node,
                return_node
            ],
            decorator_list=[],
            returns=None))

def insert_config_pb2_import(r_node):
    n = 0
    lenline = len(r_node.body)
    while n < lenline:
        if isinstance(r_node.body[n], ast.ImportFrom) or isinstance(r_node.body[n], ast.Import):
            break
        n += 1

    while n < lenline:
        if isinstance(r_node.body[n], ast.ImportFrom) and (r_node.body[n].module == '__future__'):
            n += 1
            continue
        elif isinstance(r_node.body[n], ast.ImportFrom) and (r_node.body[n].module == 'npu_bridge.npu_init'):
            n += 1
            continue
        else:
            break

    if n < lenline:
        log_success_report(n, 'import config_pb2')
        config_pb2_alias = ast.alias(name='tensorflow.core.protobuf.config_pb2', asname=None)
        r_node.body.insert(n, ast.Import(names=[config_pb2_alias]))

def ast_assign(node):
    for target in node.targets:
        if (isinstance(target, ast.Name) and target.id == 'global_jit_level') or (isinstance(target, ast.Attribute) and target.attr == 'global_jit_level'):
            log_success_report(getattr(node, 'lineno', 'None'), '*.global_jit_level')
            util_global.set_value('import_config_pb2', True)
            global_jit_level_assign_node = ast.Assign(
                targets=node.targets,
                ctx=ast.Load(),
                value=ast.Attribute(attr='OFF', ctx=ast.Load(),
                    value=ast.Attribute(attr='OptimizerOptions', ctx=ast.Load(),
                        value=ast.Name(id='config_pb2', ctx=ast.Load()))))
            node = ast.If(test=ast.NameConstant(value=True), body=[global_jit_level_assign_node], orelse=[])
            return node

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
                elif node.value.func.attr == 'OptimizerOptions':
                    log_success_report(getattr(node, 'lineno', 'None'), 'OptimizerOptions.global_jit_level')
                    util_global.set_value('import_config_pb2', True)
                    target_list = []
                    for target in node.targets:
                        target.ctx = ast.Load()
                        target_list.append(target)
                    global_jit_level_assign_node_targets = []
                    for target in target_list:
                        global_jit_level_assign_node_targets.append(ast.Attribute(value=target, attr='global_jit_level', ctx=ast.Store()))
                    global_jit_level_assign_node = ast.Assign(
                        targets=global_jit_level_assign_node_targets,
                        value=ast.Attribute(attr='OFF', ctx=ast.Load(),
                            value=ast.Attribute(attr='OptimizerOptions', ctx=ast.Load(),
                                value=ast.Name(id='config_pb2', ctx=ast.Load()))))
                    node = ast.If(test=ast.NameConstant(value=True), body=[node, global_jit_level_assign_node], orelse=[])
                elif node.value.func.attr == 'GraphOptions':
                    log_success_report(getattr(node, 'lineno', 'None'), 'GraphOptions.global_jit_level')
                    util_global.set_value('import_config_pb2', True)
                    target_list = []
                    for target in node.targets:
                        target.ctx = ast.Load()
                        target_list.append(target)
                    global_jit_level_assign_node_targets = []

                    for target in target_list:
                        global_jit_level_assign_node_targets.append(ast.Attribute(attr='global_git_level', ctx=ast.Store(),
                            value=ast.Attribute(attr='optimizer_options', ctx=ast.Load(),
                                value=target)))
                    global_jit_level_assign_node = ast.Assign(
                        targets=global_jit_level_assign_node_targets,
                        value=ast.Attribute(attr='OFF', ctx=ast.Load(),
                            value=ast.Attribute(attr='OptimizerOptions', ctx=ast.Load(),
                                value=ast.Name(id='config_pb2', ctx=ast.Load()))))
                    node = ast.If(test=ast.NameConstant(value=True), body=[node, global_jit_level_assign_node], orelse=[])
                elif node.value.func.attr == 'ConfigProto':
                    log_success_report(getattr(node, 'lineno', 'None'), 'ConfigProto.global_jit_level')
                    util_global.set_value('import_config_pb2', True)
                    target_list = []
                    for target in node.targets:
                        target.ctx = ast.Load()
                        target_list.append(target)
                    global_jit_level_assign_node_targets = []

                    for target in target_list:
                        global_jit_level_assign_node_targets.append(ast.Attribute(attr='global_git_level', ctx=ast.Store(),
                            value=ast.Attribute(attr='optimizer_options', ctx=ast.Load(),
                                value=ast.Attribute(attr='graph_options', ctx=ast.Load(),
                                    value=target))))
                    global_jit_level_assign_node = ast.Assign(
                        targets=global_jit_level_assign_node_targets,
                        value=ast.Attribute(attr='OFF', ctx=ast.Load(),
                            value=ast.Attribute(attr='OptimizerOptions', ctx=ast.Load(),
                                value=ast.Name(id='config_pb2', ctx=ast.Load()))))
                    node = ast.If(test=ast.NameConstant(value=True), body=[node, global_jit_level_assign_node], orelse=[])

                elif node.value.func.attr.find("Optimizer") != -1:
                    log_success_report(getattr(node, "lineno", "None"), "NPUDistributedOptimizer")
                    npu_func = ast.Name(id="NPUDistributedOptimizer", ctx=ast.Load())
                    npu_opt = ast.Assign(targets=node.targets, value=ast.Call(func=npu_func, args=node.targets, keywords=[]))
                    node = ast.If(test=ast.NameConstant(value=True), body=[node, npu_opt], orelse=[])
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