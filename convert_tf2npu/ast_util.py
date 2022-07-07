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

"""abstract syntax tree utils function"""

import ast
from inspect import signature
import pasta
import util_global
import config
from util import log_msg
from util import log_warning
from util import log_success_report
from util import log_hvd_distributed_mode_error


def call_name_match(call_func, call_name):
    """Judge if match call name"""
    return (isinstance(call_func, ast.Attribute) and (call_func.attr == call_name)) or \
           (isinstance(call_func, ast.Name) and call_func.id == call_name)


def check_func_arguments(origin_func, node_args, node_keywords, is_class_func):
    """Check function arguments"""
    func_args = [] if not is_class_func else [origin_func]
    func_keywords = {}
    for node_arg in node_args:
        func_args.append(pasta.dump(node_arg))
    for node_keyword in node_keywords:
        key = node_keyword.arg
        value = pasta.dump(node_keyword.value)
        func_keywords[key] = value
    try:
        signature(origin_func).bind(*func_args, **func_keywords)
    except TypeError:
        return False
    else:
        return True


def add_npu_func_to_params(node, param_index, org_func_name, param_name, npu_func, npu_func_args):
    """Add npu function to parameters"""
    param_node = None
    if ((not util_global.get_value("distributed_mode", "") or
         util_global.get_value("distributed_mode", "") == "horovod") and
            (param_name in ("callbacks", "hooks", "optimizer"))):
        return node
    log_param_msg = "".join([org_func_name, " add npu ", param_name])
    log_msg(getattr(node, "lineno", "None"), log_param_msg)
    for index, _ in enumerate(node.args):
        if param_index is not None and index == param_index:
            param_node = node.args.pop(param_index)

    for keyword in node.keywords:
        if keyword.arg == param_name:
            param_node = keyword

    if param_node:
        if isinstance(param_node, ast.keyword):
            new_value = ast.Call(func=ast.Name(id=npu_func, ctx=ast.Load()), args=[],
                                 keywords=[ast.keyword(arg=npu_func_args, value=param_node.value)])
            ast.copy_location(new_value, param_node.value)
            param_node.value = new_value
        else:
            node.keywords.append(ast.keyword(arg=param_name,
                                             value=ast.Call(func=ast.Name(id=npu_func, ctx=ast.Load()), args=[],
                                                            keywords=[
                                                                ast.keyword(arg=npu_func_args, value=param_node)])))
    else:
        node.keywords.append(ast.keyword(arg=param_name, value=pasta.parse("".join([npu_func, "()"]))))
    return node


def match_func_params_and_convert(node, origin_func, org_func_name, param_name, is_class_func):
    """Check whether function parameters is matching"""
    npu_func_map = config.npu_func_map
    param_index = None
    for index, param in enumerate(signature(origin_func).parameters):
        if param == param_name:
            param_index = index if not is_class_func else index - 1
    if param_index is not None:
        node = add_npu_func_to_params(node, param_index, org_func_name, param_name, npu_func_map.get(param_name)[0],
                                      npu_func_map.get(param_name)[1])
    return node


def convert_origin_func_to_npu(node, origin_func, org_func_name, params_list, is_class_func=None):
    """Convert original Tensorflow function to NPU function"""
    if not check_func_arguments(origin_func, node.args, node.keywords, is_class_func):
        return node
    if org_func_name == "Estimator.train":
        content = "".join([util_global.get_value('path'), ":", str(getattr(node, "lineno", "None"))])
        while True:
            message = input("Check if the train function in " + content + " is the Estimator train function. If yes, "
                            "enter 'y' to perform distributed porting on the train function. if no, enter 'n': ")
            if message == "y":
                break
            if message == "n":
                log_warning("".join(["The train func in ", content,
                                     " is user-defined functions, will not perform distributed porting"]))
                return node
            print("Input is error, Please enter 'y' or 'n'.")
    for param_name in params_list:
        node = match_func_params_and_convert(node, origin_func, org_func_name, param_name, is_class_func)

    util_global.set_value('need_conver', True)
    return node


def replace_tf_strategy_to_npu(node, new_func_node, not_tf_strategy):
    """Replace call node of tf.distribute.MirroredStrategy or tf.distribute.MultiWorkerMirroredStrategy"""
    if not_tf_strategy:
        log_hvd_distributed_mode_error(node)
        return node
    log_success_report(getattr(node, "lineno", "None"), node.func.attr)
    new_func = new_func_node
    ast.copy_location(new_func, node.func)
    node.func = new_func
    node.keywords = []
    node.args = []
    util_global.set_value('need_conver', True)
    return node
