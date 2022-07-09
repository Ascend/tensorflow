#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
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

"""NPU implemented abstract syntax tree"""

import ast
import os
import pasta
import util_global
from util import log_msg
from util import log_success_report
from util import log_warning_main_arg_not_set
from file_op import write_output_after_conver
from tf_func_def_v2 import Model
from ast_util import convert_origin_func_to_npu
from ast_impl_v1 import ast_import_from_helper as ast_import_from_v1
from ast_impl_v1 import ast_import_helper as ast_import_v1
from ast_impl_v1 import ast_function_def as ast_function_def_v1
from ast_impl_v1 import ast_attribute as ast_attribute
from ast_impl_v1 import ast_if as ast_if_v1
from ast_impl_v1 import insert_npu_resource_init
from ast_impl_v1 import insert_npu_resource_shutdown
from ast_impl_v1 import insert_keras_sess_npu_config
from ast_impl_v1 import insert_keras_sess_close
from ast_impl_v1 import ast_call as ast_call_v1


tf_v2_func_map = {
    "tf.keras.Model.compile": Model.compile,
    "tf.keras.Model.fit": Model.fit,
}


def _npu_distribute_node_helper(attr_name):
    return ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id='npu', ctx=ast.Load()),
                attr='distribute', ctx=ast.Load()),
            attr=attr_name, ctx=ast.Load())


def _npu_train_optimizer_node_helper(attr_name):
    return ast.Attribute(
            value=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id='npu', ctx=ast.Load()),
                    attr='train', ctx=ast.Load()),
                attr='optimizer', ctx=ast.Load()),
            attr=attr_name, ctx=ast.Load())


def get_npu_func_node(npu_func_name):
    """get npu func name node"""
    npu_func_map = {
        "npu.distribute.npu_distributed_keras_optimizer_wrapper": 
            _npu_distribute_node_helper('npu_distributed_keras_optimizer_wrapper'),
        "npu.distribute.all_reduce": _npu_distribute_node_helper('all_reduce'),
        "npu.train.optimizer.NpuLossScaleOptimizer": _npu_train_optimizer_node_helper('NpuLossScaleOptimizer'),
    }

    if npu_func_name in npu_func_map:
        return npu_func_map.get(npu_func_name)
    else:
        return ast.Name(id=npu_func_name, ctx=ast.Load())


def ast_import(node):
    """Modify import module"""
    if util_global.get_value('is_compat_v1', False):
        node = ast_import_v1(node)
    return node


def ast_import_from(node):
    """Modify node based on import module"""
    if util_global.get_value('is_compat_v1', False):
        node = ast_import_from_v1(node)
    return node


def find_import_insert_pos(r_node, max_insert_pos):
    """find insert position of import statement"""
    num = max_insert_pos if len(r_node.body) >= max_insert_pos else len(r_node.body)
    import_index = 0
    for i in range(0, num):
        if isinstance(r_node.body[i], ast.Import):
            return i
        if isinstance(r_node.body[i], ast.ImportFrom):
            if r_node.body[i].module != "__future__":
                return i
            import_index = i + 1
    return import_index


def _match_attribute(node, attribute):
    for field, _ in ast.iter_fields(node):
        if field == attribute:
            return True
    return False


def _match_attr(node, attr):
    if isinstance(node, ast.Attribute):
        for field, value in ast.iter_fields(node):
            if field == "attr" and value == attr:
                return True
    elif isinstance(node, ast.Name):
        for field, value in ast.iter_fields(node):
            if field == "id" and value == attr:
                return True
    return False


def api_name_match(node, api, module='tf'):
    """judge if `node` match pattern of `api`"""
    api_name = api.split('.')
    api_name.insert(0, module)
    node_name = []

    if isinstance(node.func, ast.Attribute) is False:
        return False
    sub_node = node.func
    while _match_attribute(sub_node, "value"):
        for field, value in ast.iter_fields(sub_node):
            if isinstance(value, str):
                node_name.append(value)
        sub_node = sub_node.value
    node_name.append(module)
    node_name.reverse()
    return node_name == api_name


def pattern_match(node, name, part, func):
    """judge if `node` match pattern of `name.part.func`"""
    if name == "":
        name = None
    if func == "":
        func = None
    if part is None:
        part = ""
    names = part.split('.')
    if names[0] == "":
        names = []
    match_list = [0 for _ in range(len(names)+2)]
    if isinstance(node.func, ast.Attribute):
        sub_node = node.func
        index = len(names)-1
        if func:
            if _match_attr(sub_node, func):
                match_list[0] = 1
        else:
            match_list[0] = 1
        while _match_attribute(sub_node, "value"):
            if index >= 0 and _match_attr(sub_node, names[index]):
                match_list[index+1] = 1
                index -= 1
            sub_node = sub_node.value
            if isinstance(sub_node, ast.Name):
                if name:
                    if _match_attr(sub_node, name):
                        match_list[-1] = 1
                else:
                    match_list[-1] = 1
    if sum(match_list) == len(names)+2:
        return True
    return False


def insert_npu_exprimental_loss_scale_optimizer_import(r_node):
    """Add NPU import module"""
    npu_alias = ast.alias(name='NpuExperimentalLossScaleOptimizer', asname=None)
    npu_import = ast.ImportFrom(module='npu_device.train.optimizer.npu_loss_scale_optimizer', 
        names=[npu_alias], level=0)

    max_import_npu_pos = 5
    insert_pos = find_import_insert_pos(r_node, max_import_npu_pos)
    r_node.body.insert(insert_pos, npu_import)
    log_msg(insert_pos, "from npu_device.train.optimizer.npu_loss_scale_optimizer \
        import NpuExperimentalLossScaleOptimizer")


def insert_npu_callbacks_func_import(r_node):
    """Add NPU import module"""
    npu_alias = ast.alias(name='npu_callbacks_append', asname=None)
    npu_import = ast.ImportFrom(module='npu_device.distribute.hccl', names=[npu_alias], level=0)

    max_import_npu_pos = 5
    insert_pos = find_import_insert_pos(r_node, max_import_npu_pos)
    r_node.body.insert(insert_pos, npu_import)
    log_msg(insert_pos, "from npu_device.distribute.hccl import npu_callbacks_append")


def insert_npu_broadcast_func_import(r_node):
    """Add NPU import module"""
    npu_alias = ast.alias(name='broadcast_keras_model', asname=None)
    npu_import = ast.ImportFrom(module='npu_device.distribute.npu_callbacks', names=[npu_alias], level=0)

    max_import_npu_pos = 5
    insert_pos = find_import_insert_pos(r_node, max_import_npu_pos)
    r_node.body.insert(insert_pos, npu_import)
    log_msg(insert_pos, "from npu_device.distribute.npu_callbacks import broadcast_keras_model")


def insert_npu_import(r_node):
    """Add NPU import module"""
    npu_alias = ast.alias(name='npu_device', asname='npu')
    npu_import = ast.Import(names=[npu_alias], level=0)

    max_import_npu_pos = 5
    insert_pos = find_import_insert_pos(r_node, max_import_npu_pos)
    r_node.body.insert(insert_pos, npu_import)
    log_msg(insert_pos, "import npu_device as npu")


def insert_compat_init_import(r_node):
    """Add Compat v1 npu_init import module"""
    npu_compatv1_alias = ast.alias(name="*", asname=None)
    npu_compatv1_import = ast.ImportFrom(module="npu_device.compat.v1.npu_init", names=[npu_compatv1_alias], level=0)

    max_import_npu_pos = 5
    num = max_import_npu_pos if len(r_node.body) >= max_import_npu_pos else len(r_node.body)
    for i in range(0, num):
        if isinstance(r_node.body[i], ast.Import) and r_node.body[i].names[0].name == 'npu_device':
            r_node.body.insert(i+1, npu_compatv1_import)
            log_msg(i+1, "from npu_device.compat.v1.npu_init import *")
            break


def insert_npu_device_init(r_node):
    """Add NPU device initiate"""
    npu_open = ast.Call(func=ast.Attribute(value=ast.Name(id='npu', atx=ast.Load()), attr='open', ctx=ast.Load()),
                        args=[],
                        keywords=[])
    npu_default_device = ast.Expr(value=ast.Call(func=ast.Attribute(value=npu_open, attr='as_default', ctx=ast.Load()),
                                                 args=[],
                                                 keywords=[]))

    max_import_npu_pos = 5
    num = max_import_npu_pos if len(r_node.body) >= max_import_npu_pos else len(r_node.body)
    for i in range(0, num):
        if isinstance(r_node.body[i], ast.Import) and r_node.body[i].names[0].name == 'npu_device':
            r_node.body.insert(i + 1, npu_default_device)
            log_msg(i + 1, "npu.open().as_default()")
            break


def is_keras_optimizer_name(func_name):
    """Judge if keras optimizers name"""
    keras_optimizer_names = {'SGD', 'RMSprop', 'Adam', 'Ftrl', \
        'Adagrad', 'Adadelta', 'Adamax', 'Nadam'}
    if func_name in keras_optimizer_names:
        return True
    return False


def is_keras_get_optimizer_param_name(param):
    """Judge if call tf.keras.optimizers.get(param)"""
    keras_get_optimizer_param_names = {'adadelta', 'adagrad', 'adam', 'adamax', 'nadam', \
        'rmsprop', 'sgd', 'ftrl'}
    if param.tolower() in keras_get_optimizer_param_names:
        return True
    return False


def _decorate_distribute_optimizer_wrapper_at_call(node):
    """decorate npu distribute optimizer wrapper at ast call node"""
    log_msg(getattr(node, "lineno", "None"), "add npu distribute optimizer to tensorflow optimizer")
    new_node = ast.Call(func=get_npu_func_node("npu.distribute.npu_distributed_keras_optimizer_wrapper"), args=[node],
                        keywords=[])
    ast.copy_location(new_node, node)
    util_global.set_value('need_conver', True)
    return new_node


def convert_tf_distribute_apis(node):
    """Convert distributed strategy API"""
    if isinstance(node.func, ast.Attribute) and is_keras_optimizer_name(node.func.attr):
        return _decorate_distribute_optimizer_wrapper_at_call(node)
    if isinstance(node.func, ast.Name) and is_keras_optimizer_name(node.func.id):
        return _decorate_distribute_optimizer_wrapper_at_call(node)
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'get'):
        if len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            if is_keras_get_optimizer_param_name(node.args[0].value):
                return _decorate_distribute_optimizer_wrapper_at_call(node)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
        node = convert_origin_func_to_npu(node, tf_v2_func_map.get("tf.keras.Model.fit"), 
            "Model.fit", ["callbacks"], True)
        util_global.set_value('need_import_npu_callbacks_func', True)
        return node
    return node


def insert_enable_v1(r_node):
    """Add NPU device compat enable v1"""
    npu_device_compat = ast.Attribute(value=ast.Name(id='npu', ctx=ast.Load()),
                              attr='compat', ctx=ast.Load())
    enable_v1 = ast.Expr(value=ast.Call(func=ast.Attribute(value=npu_device_compat, attr='enable_v1', ctx=ast.Load()),
                         args=[],
                         keywords=[]))

    max_import_npu_pos = 5
    num = max_import_npu_pos if len(r_node.body) >= max_import_npu_pos else len(r_node.body)
    for i in range(0, num):
        if isinstance(r_node.body[i], ast.ImportFrom) and r_node.body[i].module == "npu_device.compat.v1.npu_init":
            r_node.body.insert(i+1, enable_v1)
            log_msg(i+1, "npu.compat.v1.enable_v1()")
            break


def is_custom_keras_model(func_name):
    custom_keras_models = util_global.get_value('custom_keras_models', [])
    return func_name in custom_keras_models


def ast_call(node):
    """Visit and transform ast call node"""
    distributed_mode = util_global.get_value("distributed_mode", "")

    if isinstance(node.func, ast.Attribute) and \
        ((node.func.attr == "Model") or is_custom_keras_model(node.func.attr)):
        log_msg(getattr(node, "lineno", "None"), "add npu broadcast_keras_model to tensorflow keras Model")
        new_node = ast.Call(func=get_npu_func_node("broadcast_keras_model"), args=[node],
                            keywords=[])
        ast.copy_location(new_node, node)
        util_global.set_value('need_conver', True)
        util_global.set_value('need_import_npu_broadcast_func', True)
        return new_node

    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'set_memory_growth'):
        log_success_report(getattr(node, 'lineno', 'None'), 'set_memory_growth')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node

    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'LossScaleOptimizer'):
        if api_name_match(node, "keras.mixed_precision.LossScaleOptimizer"):
            log_msg(getattr(node, "lineno", "None"), "add npu.train.optimizer.NpuLossScaleOptimizer")
            new_func = get_npu_func_node("npu.train.optimizer.NpuLossScaleOptimizer")
        elif api_name_match(node, "keras.mixed_precision.experimental.LossScaleOptimizer"):
            log_msg(getattr(node, "lineno", "None"), "add npu.train.optimizer.NpuExperimentalLossScaleOptimizer")
            new_func = get_npu_func_node("NpuExperimentalLossScaleOptimizer")
            util_global.set_value('need_import_experimental_loss_scale_optimizer', True)
        else:
            return node
        ast.copy_location(new_func, node.func)
        node.func = new_func
        util_global.set_value('need_conver', True)
        return node


    if distributed_mode == "tf_strategy":  # this cond should be placed at the end of the Call function.
        return convert_tf_distribute_apis(node)

    return node


def ast_function_def(node):
    """Modify node based on function_def"""
    if util_global.get_value('is_compat_v1', False):
            if node.name == 'gelu':
                return ast_function_def_v1(node)
    return node


def ast_if(node):
    """Modify node based on if statement"""
    if isinstance(node.test, ast.Compare):
        if len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Str):
            if node.test.comparators[0].s == "__main__":
                util_global.set_value("has_main_func", True)
                num_main_found = util_global.get_value("num_main_found", 0)
                util_global.set_value("num_main_found", num_main_found + 1)

                if util_global.get_value('is_compat_v1', False):
                    node = ast_if_v1(node)
    return node


class ConverByAst(ast.NodeTransformer):
    """Class for transforming python ast node"""

    def visit_Import(self, node):
        """Visit and transform import node"""
        self.generic_visit(node)
        node = ast_import(node)
        return node

    def visit_ImportFrom(self, node):
        """Visit and transform importfrom node"""
        self.generic_visit(node)
        node = ast_import_from(node)
        return node

    def visit_FunctionDef(self, node):
        """Visit and transform function def node"""
        node = ast_function_def(node)
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        """Visit and transform attr node"""
        self.generic_visit(node)
        return ast_attribute(node)

    def visit_Call(self, node):
        """Visit and transform call node"""
        self.generic_visit(node)
        node = ast_call(node)
        if isinstance(node, ast.Call) and util_global.get_value('is_compat_v1', False):
            if (isinstance(node.func, ast.Attribute) and (pattern_match(node, "tf", "", "device") or \
                pattern_match(node, "tf", None, "device"))):
                log_success_report(getattr(node, "lineno", "None"), node.func.attr)
                node.args = [ast.Str(s='/cpu:0')]
                util_global.set_value('need_conver', True)

            node = ast_call_v1(node)
        return node

    def visit_If(self, node):
        """Visit and transform if node"""
        self.generic_visit(node)
        node = ast_if(node)
        return node

    def visit_Module(self, node):
        """Visit and transform Module node"""
        self.generic_visit(node)
        if util_global.get_value('is_main_file', False) or util_global.get_value('has_main_func', False):
            util_global.set_value('need_conver', True)
        return node


def conver(r_node, out_path_dst, file_name):
    """Add necessary imported modules"""
    is_compat_v1 = util_global.get_value('is_compat_v1', False)
    correct_num_main_func = 1
    if file_name != "__init__.py":
        if not is_compat_v1:
            if util_global.get_value('need_import_npu_broadcast_func', False):
                insert_npu_broadcast_func_import(r_node)
            if util_global.get_value('need_import_npu_callbacks_func', False):
                insert_npu_callbacks_func_import(r_node)
            if util_global.get_value('need_import_experimental_loss_scale_optimizer', False):
                insert_npu_exprimental_loss_scale_optimizer_import(r_node)
            insert_npu_import(r_node)
        else:
            insert_npu_import(r_node)
            insert_compat_init_import(r_node)
            insert_enable_v1(r_node)
    if (util_global.get_value('num_main_found', False) != correct_num_main_func) and \
        not util_global.get_value('main', ""):
        log_warning_main_arg_not_set()
    if util_global.get_value('is_main_file', False) or \
        (not util_global.get_value('main', "") and util_global.get_value('has_main_func', False)):
        if not is_compat_v1:
            insert_npu_device_init(r_node)
    
    if is_compat_v1:
        distributed_mode = util_global.get_value('distributed_mode', "")
        if distributed_mode == "horovod" and util_global.get_value('is_main_file', False):
            insert_npu_resource_init(r_node)
            insert_npu_resource_shutdown(r_node)
        if util_global.get_value('is_main_file', False) and util_global.get_value('is_keras_net', False):
            insert_keras_sess_npu_config(r_node)
            insert_keras_sess_close(r_node)
    dst_content = pasta.dump(r_node)
    write_output_after_conver(os.path.join(util_global.get_value('output'), out_path_dst, file_name), dst_content)
