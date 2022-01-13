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

"""NPU implemented abstract syntax tree"""

import ast
import copy
from inspect import signature
import pasta
import util_global
from util import log_msg
from util import log_warning
from util import log_success_report
from util import log_hvd_distributed_mode_error
from util import log_strategy_distributed_mode_error
from tf_func_def import TrainSpec
from tf_func_def import EvalSpec
from tf_func_def import Estimator
from tf_func_def import Model
from tf_func_def import Session
from tf_func_def import InteractiveSession
from tf_func_def import Supervisor
from tf_func_def import MonitoredTrainingSession

tf_func_map = {"tf.estimator.TrainSpec": TrainSpec,
               "tf.estimator.EvalSpec": EvalSpec,
               "tf.estimator.Estimator.train": Estimator.train,
               "tf.keras.Model.compile": Model.compile,
               "tf.keras.Model.fit": Model.fit,
               "tf.keras.Model.fit_generator": Model.fit_generator,
               "tf.Session": Session,
               "tf.InteractiveSession": InteractiveSession,
               "tf.train.Supervisor.managed_session": Supervisor.managed_session,
               "tf.train.MonitoredTrainingSession": MonitoredTrainingSession}


def attribute(node):
    """Modify node attribute"""
    log_success_report(getattr(node, "lineno", "None"), node.attr)
    node = ast.Name(id=util_global.get_value(node.attr)[0], ctx=ast.Load())
    util_global.set_value('need_conver', True)
    return node


def import_from(node):
    """Modify node based on import module"""
    if node.module:
        values = node.module.split(".")
        if "keras" in values:
            util_global.set_value('is_keras_net', True)
        if "horovod" in values:
            log_msg(getattr(node, "lineno", "None"), "remove horovod import line to None")
            util_global.set_value('has_hvd_api', True)
            new_node = ast.Expr(value=ast.NameConstant(value=None))
            ast.copy_location(new_node, node)
            util_global.set_value('need_conver', True)
            return new_node
    for value in node.names:
        if isinstance(value, ast.alias):
            values = value.name.split(".")
            if "keras" in values:
                util_global.set_value('is_keras_net', True)
            if "horovod" in values:
                log_msg(getattr(node, "lineno", "None"), "remove horovod import line to None")
                util_global.set_value('has_hvd_api', True)
                new_node = ast.Expr(value=ast.NameConstant(value=None))
                ast.copy_location(new_node, node)
                util_global.set_value('need_conver', True)
                return new_node
    util_global.set_value('need_conver', True)
    return node


def ast_import(node):
    """Modify import module"""
    for value in node.names:
        if isinstance(value, ast.alias):
            values = value.name.split(".")
            if "keras" in values:
                util_global.set_value('is_keras_net', True)
            if "horovod" in values:
                log_msg(getattr(node, "lineno", "None"), "remove horovod import line to None")
                util_global.set_value('has_hvd_api', True)
                new_node = ast.Expr(value=ast.NameConstant(value=None))
                ast.copy_location(new_node, node)
                util_global.set_value('need_conver', True)
                return new_node
    util_global.set_value('need_conver', True)
    return node


def ast_function_def(node):
    """Modify node based on function_def"""
    log_success_report(getattr(node, "lineno", "None"), node.name)
    arg_name = node.args.args[0].arg
    node.body = [ast.Return(value=ast.Call(
        func=ast.Attribute(value=ast.Name(id=util_global.get_value(node.name)[0],
                                          ctx=ast.Load()), attr='gelu', ctx=ast.Load()),
        args=[ast.Name(id=arg_name, ctx=ast.Load())],
        keywords=[]))]

    util_global.set_value('need_conver', True)
    return node


def ast_if(node):
    """Modify node based on if statement"""
    if isinstance(node.test, ast.Compare):
        if len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Str):
            if node.test.comparators[0].s == "__main__":
                util_global.set_value("is_main_file", False)
                util_global.set_value("has_main_func", True)
                if util_global.get_value("is_keras_net", False):
                    log_msg(getattr(node, "lineno", "None"), "add keras session npu config")
                    close_sess_call = ast.Call(func=ast.Name(id="close_session", ctx=ast.Load()),
                                               args=[ast.Name(id="npu_keras_sess", ctx=ast.Load())], keywords=[])
                    keras_sess_assign = ast.Assign(targets=[ast.Name(id="npu_keras_sess", ctx=ast.Store())],
                                                   value=ast.Call(
                                                       func=ast.Name(id="set_keras_session_npu_config", ctx=ast.Load()),
                                                       args=[], keywords=[]))
                    node.body = [keras_sess_assign] + node.body + [ast.Expr(value=close_sess_call)]
                    util_global.set_value('need_conver', True)
                if util_global.get_value("distributed_mode", "") == "horovod":
                    log_msg(getattr(node, "lineno", "None"), "add npu resource init api")
                    close_sess_call = ast.Call(func=ast.Name(id="close_session", ctx=ast.Load()),
                                               args=[ast.Name(id="npu_sess", ctx=ast.Load())], keywords=[])
                    init_assign = ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id="npu_sess", ctx=ast.Store()),
                                                                      ast.Name(id="npu_shutdown", ctx=ast.Store())],
                                                                ctx=ast.Store())],
                                             value=ast.Call(func=ast.Name(id="init_resource", ctx=ast.Load()), args=[],
                                                            keywords=[]))
                    shutdown_call = ast.Call(func=ast.Name(id="shutdown_resource", ctx=ast.Load()),
                                             args=[ast.Name(id="npu_sess", ctx=ast.Load()),
                                                   ast.Name(id="npu_shutdown", ctx=ast.Load())],
                                             keywords=[])
                    node.body = [init_assign] + node.body + [ast.Expr(value=shutdown_call),
                                                             ast.Expr(value=close_sess_call)]
                    util_global.set_value('need_conver', True)
                return node


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
    npu_func_map = {"config": ["npu_config_proto", "config_proto"],
                    "hooks": ["npu_hooks_append", "hooks_list"],
                    "callbacks": ["npu_callbacks_append", "callbacks_list"],
                    "optimizer": ["npu_distributed_optimizer_wrapper", "optimizer"]}
    param_index = None
    for index, param in enumerate(signature(origin_func).parameters):
        if param == param_name:
            param_index = index if not is_class_func else index - 1
    if param_index is not None:
        node = add_npu_func_to_params(node, param_index, org_func_name, param_name, npu_func_map[param_name][0],
                                      npu_func_map[param_name][1])
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


def convert_dynamic_loss_scale(node):
    """Convert dynamic loss scale related Tensorflow APIs"""
    log_msg(getattr(node, 'lineno', 'None'), "change tf.train.experimental.DynamicLossScale"
                                             " to ExponentialUpdateLossScaleManager")
    node.func = ast.Name(id="ExponentialUpdateLossScaleManager", ctx=ast.Load())

    def check_arg(node):
        initial_loss_scale = None
        increment_period = None
        multiplier = None
        for index, arg in enumerate(node.args):
            if index == 0:
                initial_loss_scale = arg
            if index == 1:
                increment_period = arg
            if index == 2:
                multiplier = arg
        for keyword in node.keywords:
            if keyword.arg == "initial_loss_scale":
                keyword.arg = "init_loss_scale"
                initial_loss_scale = keyword
            if keyword.arg == "increment_period":
                keyword.arg = "incr_every_n_steps"
                increment_period = keyword
            if keyword.arg == "multiplier":
                keyword.arg = "incr_ratio"
                multiplier = keyword
        return (initial_loss_scale, increment_period, multiplier)

    (initial_loss_scale, increment_period, multiplier) = check_arg(node)
    if initial_loss_scale:
        if not isinstance(initial_loss_scale, ast.keyword):
            node.keywords.append(ast.keyword(arg="init_loss_scale", value=initial_loss_scale))
    else:
        node.keywords.append(ast.keyword(arg="init_loss_scale", value=pasta.parse("2**15")))
    if increment_period:
        if not isinstance(increment_period, ast.keyword):
            node.keywords.append(ast.keyword(arg="incr_every_n_steps", value=increment_period))
    else:
        node.keywords.append(ast.keyword(arg="incr_every_n_steps", value=pasta.parse("2000")))
    if multiplier:
        if not isinstance(multiplier, ast.keyword):
            node.keywords.append(ast.keyword(arg="incr_ratio", value=multiplier))
    else:
        node.keywords.append(ast.keyword(arg="incr_ratio", value=pasta.parse("2")))
    node.args = []
    util_global.set_value('need_conver', True)
    return node


def convert_loss_scale_api(node):
    """Convert loss scale related Tensorflow APIs"""
    if isinstance(node.func, ast.Attribute):
        if node.func.attr == "FixedLossScale":
            log_msg(getattr(node, 'lineno', 'None'), "change tf.train.experimental.FixedLossScale"
                                                     " to FixedLossScaleManager")
            node.func = ast.Name(id="FixedLossScaleManager", ctx=ast.Load())
            if len(node.keywords) == 1:
                node.keywords[0].arg = "loss_scale"
            util_global.set_value('need_conver', True)
            return node
        if node.func.attr == "DynamicLossScale":
            return convert_dynamic_loss_scale(node)
        if node.func.attr == "MixedPrecisionLossScaleOptimizer":
            log_msg(getattr(node, 'lineno', 'None'), "change tf.train.experimental.MixedPrecisionLossScaleOptimizer"
                                                     " to NPULossScaleOptimizer")
            node.func = ast.Name(id="NPULossScaleOptimizer", ctx=ast.Load())
            for keyword in node.keywords:
                if keyword.arg == "loss_scale":
                    keyword.arg = "loss_scale_manager"
            util_global.set_value('need_conver', True)
            return node


def convert_hvd_distributed_api(node):
    """Convert horovod distributed APIs"""
    log_msg(getattr(node, "lineno", "None"), 'change hvd.DistributedOptimizer to npu_distributed_optimizer_wrapper')
    node.func = ast.Name(id="npu_distributed_optimizer_wrapper", ctx=ast.Load())
    opt_keyword = None
    for keyword in node.keywords:
        if keyword.arg == "optimizer":
            opt_keyword = keyword
    node.keywords.clear()
    if opt_keyword is None:
        opt_arg = node.args[0]
        node.args.clear()
        node.args.append(opt_arg)
    else:
        node.keywords.append(opt_keyword)
    util_global.set_value('need_conver', True)
    return node


def convert_tf_gradient_distributed(node):
    """Convert Tensorflow gradient APIs in distributed mode"""
    content = "".join([util_global.get_value('path'), ":", str(getattr(node, "lineno", "None")),
                       " is tf.gradient api, tool inserts npu_allreduce after computing grads by default.",
                       " You can adjust the allreduce position according to the algorithm"])
    log_warning(content)
    new_node = ast.Call(func=ast.Name(id="npu_allreduce", ctx=ast.Load()), args=[node], keywords=[])
    ast.copy_location(new_node, node)
    util_global.set_value("need_conver", True)
    return new_node


def convert_distributed_strategy_apis(node):
    """Convert distributed strategy API"""
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
        if ("Optimizer" in node.func.attr and node.func.attr != "ScipyOptimizerInterface" and
                node.func.attr != "MixedPrecisionLossScaleOptimizer"):
            log_msg(getattr(node, "lineno", "None"), "add npu distribute optimizer to tensorflow optimizer")
            new_node = ast.Call(func=ast.Name(id="npu_distributed_optimizer_wrapper", ctx=ast.Load()), args=[node],
                                keywords=[])
            ast.copy_location(new_node, node)
            util_global.set_value('need_conver', True)
            return new_node
    if isinstance(node.func, ast.Name) and "Optimizer" in node.func.id and node.func.id != "NPULossScaleOptimizer":
        log_msg(getattr(node, "lineno", "None"), "add npu distribute optimizer to tensorflow optimizer")
        new_node = ast.Call(func=ast.Name(id="npu_distributed_optimizer_wrapper", ctx=ast.Load()), args=[node],
                            keywords=[])
        ast.copy_location(new_node, node)
        util_global.set_value('need_conver', True)
        return new_node
    if _call_name_match(node.func, "TrainSpec"):
        return convert_origin_func_to_npu(node, tf_func_map["tf.estimator.TrainSpec"], "TrainSpec", ["hooks"])
    if _call_name_match(node.func, "EvalSpec"):
        return convert_origin_func_to_npu(node, tf_func_map["tf.estimator.EvalSpec"], "EvalSpec", ["hooks"])
    if isinstance(node.func, ast.Attribute) and node.func.attr == "train":
        if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "learning":
            return node
        return convert_origin_func_to_npu(node, tf_func_map["tf.estimator.Estimator.train"],
                                          "Estimator.train", ["hooks"], True)
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'compile'):
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "re":
            return node
        return convert_origin_func_to_npu(node, tf_func_map["tf.keras.Model.compile"],
                                          "Model.compile", ["optimizer"], True)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
        return convert_origin_func_to_npu(node, tf_func_map["tf.keras.Model.fit"], "Model.fit", ["callbacks"], True)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "fit_generator":
        return convert_origin_func_to_npu(node, tf_func_map["tf.keras.Model.fit_generator"],
                                          "Model.fit_generator", ["callbacks"], True)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "gradients" and \
            isinstance(node.func.value, ast.Name) and node.func.value.id == "tf":
        return convert_tf_gradient_distributed(node)
    return node


def ast_call(node):
    """Visit and transform ast call node"""
    distributed_mode = util_global.get_value("distributed_mode", "")
    is_not_strategy = distributed_mode in ("horovod", "")
    is_not_horovod = distributed_mode in ("tf_strategy", "")
    convert_loss_scale_api(node)
    if _call_name_match(node.func, "set_experimental_options"):
        log_msg(getattr(node, 'lineno', 'None'),
                'change set_experimental_options(*) to set_experimental_options(experimental_options)')
        node.args = [ast.Name(id='experimental_options', ctx=ast.Load())]
        node.keywords = []
        util_global.set_value('need_conver', True)
    if isinstance(node.func, ast.Name) and node.func.id == 'check_available_gpus':
        log_msg(getattr(node, 'lineno', 'None'), "change check_available_gpus() to ['/device:CPU:0']")
        util_global.set_value('need_conver', True)
        return ast.List(elts=[ast.Str(s="/device:CPU:0")], ctx=ast.Load())
    if ((isinstance(node.func, ast.Name) and node.func.id == 'GraphOptions') or
            (isinstance(node.func, ast.Attribute) and node.func.attr == 'GraphOptions')):
        log_success_report(getattr(node, 'lineno', 'None'), 'GraphOptions()')
        src = copy.deepcopy(node)
        node.func = ast.Name(id='npu_graph_options', ctx=ast.Load())
        node.args = []
        node.keywords = []
        node.keywords.append(ast.keyword(arg='graph_options', value=src))
        util_global.set_value('need_conver', True)
        return node
    if (isinstance(node.func, ast.Name) and node.func.id == 'OptimizerOptions') or \
            (isinstance(node.func, ast.Attribute) and node.func.attr == 'OptimizerOptions'):
        log_success_report(getattr(node, 'lineno', 'None'), 'OptimizerOptions()')
        src = copy.deepcopy(node)
        node.func = ast.Name(id='npu_optimizer_options', ctx=ast.Load())
        node.args = []
        node.keywords = []
        node.keywords.append(ast.keyword(arg='optimizer_options', value=src))
        util_global.set_value('need_conver', True)
        return node
    if _call_name_match(node.func, "Session"):
        return convert_origin_func_to_npu(node, tf_func_map["tf.Session"], "tf.Session", ["config"])
    if _call_name_match(node.func, "InteractiveSession"):
        return convert_origin_func_to_npu(node, tf_func_map["tf.InteractiveSession"],
                                          "tf.InteractiveSession", ["config"])
    if isinstance(node.func, ast.Attribute) and node.func.attr == "BroadcastGlobalVariablesHook":
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "hvd":
            if is_not_horovod:
                log_strategy_distributed_mode_error(node)
                return node
            log_msg(getattr(node, "lineno", "None"),
                    'change hvd.BroadcastGlobalVariablesHook to NPUBroadcastGlobalVariablesHook')
            node = pasta.parse("NPUBroadcastGlobalVariablesHook(0, int(os.getenv('RANK_ID', '0')))")
            util_global.set_value('need_conver', True)
            return node
    if isinstance(node.func, ast.Attribute) and node.func.attr == "BroadcastGlobalVariablesCallback":
        if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "callbacks":
            if is_not_horovod:
                log_strategy_distributed_mode_error(node)
                return node
            log_msg(getattr(node, "lineno", "None"),
                    'change hvd.callbacks.BroadcastGlobalVariablesCallback to NPUBroadcastGlobalVariablesCallback')
            node = pasta.parse("NPUBroadcastGlobalVariablesCallback(root_rank=0)")
            util_global.set_value('need_conver', True)
            return node
    if isinstance(node.func, ast.Attribute) and node.func.attr == "DistributedOptimizer":
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "hvd":
            if is_not_horovod:
                log_strategy_distributed_mode_error(node)
                return node
            return convert_hvd_distributed_api(node)
    if isinstance(node.func, ast.Attribute) and node.func.attr == 'shard':
        log_success_report(getattr(node, "lineno", "None"), 'shard')
        node.args = [pasta.parse("int(os.getenv('RANK_SIZE', '1'))"),
                     pasta.parse("int(os.getenv('RANK_ID', '0'))")]
        node.keywords.clear()
        util_global.set_value('need_conver', True)
        return node
    if isinstance(node.func, ast.Attribute) and node.func.attr == 'dropout':
        if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'nn':
            for index, _ in enumerate(node.args):
                if index == 2:
                    return node
            for keyword in node.keywords:
                if keyword.arg == "noise_shape":
                    return node
            log_success_report(getattr(node, "lineno", "None"), 'dropout')
            node.func = ast.Attribute(value=ast.Name(id='npu_ops', ctx=ast.Load()), attr='dropout', ctx=ast.Load())
            keywords_new = []
            for keyword in node.keywords:
                if keyword.arg != 'rate':
                    keywords_new.append(keyword)
                else:
                    keywords_new.append(ast.keyword(arg='keep_prob', value=ast.BinOp(left=ast.Num(n=1), op=ast.Sub(),
                                                                                     right=keyword.value)))
            node.keywords = keywords_new
            util_global.set_value('need_conver', True)
        return node
    if isinstance(node.func, ast.Attribute) and \
            ((node.func.attr == 'map_and_batch') or
             (node.func.attr == 'batch' and (not isinstance(node.func.value, ast.Attribute) or (
                     isinstance(node.func.value, ast.Attribute) and node.func.value.attr != 'train')))):
        exist = False
        for keyword in node.keywords:
            if keyword.arg == 'drop_remainder':
                exist = True
                if ((isinstance(keyword.value, ast.NameConstant) and not keyword.value.value) or
                        (not isinstance(keyword.value, ast.NameConstant))):
                    log_success_report(getattr(node, "lineno", "None"), node.func.attr)
                    keyword.value = pasta.parse('True')
                    util_global.set_value('need_conver', True)
        if not exist:
            log_success_report(getattr(node, "lineno", "None"), node.func.attr)
            keyword = ast.keyword(arg='drop_remainder', value=pasta.parse('True'))
            node.keywords.insert(0, keyword)
            util_global.set_value('need_conver', True)
        return node
    if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'tf' and node.func.attr == 'device'):
        log_success_report(getattr(node, "lineno", "None"), node.func.attr)
        node.args = [ast.Str(s='/cpu:0')]
        util_global.set_value('need_conver', True)
        return node
    if isinstance(node.func, ast.Attribute) and \
            (node.func.attr == "get_distribution_strategy" or
             node.func.attr == "MirroredStrategy" or
             node.func.attr == "MultiWorkerMirroredStrategy"):
        if is_not_strategy:
            log_hvd_distributed_mode_error(node)
            return node
        log_success_report(getattr(node, "lineno", "None"), node.func.attr)
        new_func = ast.Attribute(value=ast.Name(id="npu_strategy", ctx=ast.Load()),
                                 attr="NPUStrategy", ctx=ast.Load())
        ast.copy_location(new_func, node.func)
        node.func = new_func
        node.keywords = []
        node.args = []
        util_global.set_value('need_conver', True)
        return node
    if (isinstance(node.func, ast.Attribute) and (node.func.attr == 'RunConfig')) and \
            (_call_name_match(node.func.value, 'estimator') or _call_name_match(node.func.value, 'tpu')):
        if node.keywords.count("train_distribute") or node.keywords.count("eval_distribute"):
            if is_not_strategy:
                log_hvd_distributed_mode_error(node)
        save_summary_steps = None
        for keyword in node.keywords:
            if keyword.arg == 'save_summary_steps':
                save_summary_steps = keyword
                break
        if len(node.args) < 3 and not save_summary_steps:
            log_msg(getattr(node, 'lineno'), 'RunConfig() add save_summary_steps=0')
            util_global.set_value('need_conver', True)
            node.keywords.append(ast.keyword(arg='save_summary_steps', value=pasta.parse('0')))
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'TPUEstimator') and \
            ((isinstance(node.func.value, ast.Attribute) and (node.func.value.attr == 'tpu')) or
             (isinstance(node.func.value, ast.Name) and (node.func.value.id == 'tpu'))):
        add_eval_on_tpu = True
        add_use_tpu = True
        add_export_to_tpu = True
        for keyword in node.keywords:
            if (keyword.arg == 'eval_on_tpu') or (keyword.arg == 'use_tpu') or (keyword.arg == 'export_to_tpu'):
                if (not isinstance(keyword.value, ast.NameConstant)) or \
                        (isinstance(keyword.value, ast.NameConstant) and (keyword.value.value)):
                    log_success_report(getattr(node, 'lineno', 'None'), 'TPUEstimator(' + keyword.arg + '=*)')
                    keyword.value = pasta.parse('False')
                    util_global.set_value('need_conver', True)
                if add_eval_on_tpu and (keyword.arg == 'eval_on_tpu'):
                    add_eval_on_tpu = False
                if add_use_tpu and (keyword.arg == 'use_tpu'):
                    add_use_tpu = False
                if add_export_to_tpu and (keyword.arg == 'export_to_tpu'):
                    add_export_to_tpu = False
        if add_eval_on_tpu:
            log_success_report(getattr(node, 'lineno', 'None'), 'TPUEstimator(eval_on_tpu=*)')
            node.keywords.append(ast.keyword(arg='eval_on_tpu', value=pasta.parse('False')))
            util_global.set_value('need_conver', True)
        if add_use_tpu:
            log_success_report(getattr(node, 'lineno', 'None'), 'TPUEstimator(use_tpu=*)')
            node.keywords.append(ast.keyword(arg='use_tpu', value=pasta.parse('False')))
            util_global.set_value('need_conver', True)
        if add_export_to_tpu:
            log_success_report(getattr(node, 'lineno', 'None'), 'TPUEstimator(export_to_tpu=*)')
            node.keywords.append(ast.keyword(arg='export_to_tpu', value=pasta.parse('False')))
            util_global.set_value('need_conver', True)
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'VirtualDeviceConfiguration'):
        log_success_report(getattr(node, 'lineno', 'None'), 'VirtualDeviceConfiguration')
        util_global.set_value('need_conver', True)
        memory_limit = None
        for keyword in node.keywords:
            if keyword.arg == 'memory_limit':
                memory_limit = keyword
                break
        if memory_limit:
            memory_limit.value = ast.NameConstant(value=None)
        else:
            node.keywords.append(ast.keyword(arg='memory_limit', value=ast.NameConstant(value=None)))
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'set_soft_device_placement'):
        log_success_report(getattr(node, 'lineno', 'None'), 'set_soft_device_placement')
        util_global.set_value('need_conver', True)
        node.args = []
        node.keywords = [ast.keyword(arg='enabled', value=ast.NameConstant(value=True))]
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'set_memory_growth'):
        log_success_report(getattr(node, 'lineno', 'None'), 'set_memory_growth')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'set_virtual_device_configuration'):
        log_success_report(getattr(node, 'lineno', 'None'), 'set_virtual_device_configuration')
        util_global.set_value('need_conver', True)
        node = ast.NameConstant(value=None)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'jit_scope'):
        if isinstance(node.func.value, ast.Attribute) and (node.func.value.attr == 'experimental'):
            if isinstance(node.func.value.value, ast.Attribute) and (node.func.value.value.attr == 'xla'):
                log_success_report(getattr(node, 'lineno', 'None'), '*.xla.experimental.jit_scope')
                util_global.set_value('need_conver', True)
                compile_ops = None
                for keyword in node.keywords:
                    if keyword.arg == 'compile_ops':
                        compile_ops = keyword
                        break
                if compile_ops:
                    compile_ops.value = pasta.parse('False')
                else:
                    node.keywords.append(ast.keyword(arg='compile_ops', value=pasta.parse('False')))
                return node
    for estimator in util_global.get_value('Estimators', []):
        if (isinstance(node.func, ast.Attribute) and (node.func.attr == estimator)) \
                or (isinstance(node.func, ast.Name) and (node.func.id == estimator)):
            log_msg(getattr(node, 'lineno'), "".join([estimator, '() add config=npu_run_config_init()']))
            config = None
            for keyword in node.keywords:
                if keyword.arg == 'config':
                    config = keyword
                    break
            if config:
                new_value = ast.Call(func=ast.Name(id='npu_run_config_init', ctx=ast.Load()),
                                     args=[],
                                     keywords=[ast.keyword(arg='run_config', value=config.value)])
                ast.copy_location(new_value, config.value)
                config.value = new_value
            else:
                node.keywords.append(ast.keyword(arg='config',
                                                 value=pasta.parse('npu_run_config_init()')))
            util_global.set_value('need_conver', True)
            return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'clear_session'):
        log_msg(getattr(node, 'lineno'), "change keras.clear_session() to npu_clear_session()")
        node = ast.Call(func=ast.Name(id='npu_clear_session', ctx=ast.Load()),
                        args=[], keywords=[])
        util_global.set_value('need_conver', True)
    if _call_name_match(node.func, "MonitoredTrainingSession"):
        return convert_origin_func_to_npu(node, tf_func_map["tf.train.MonitoredTrainingSession"],
                                          "MonitoredTrainingSession", ["config", "hooks"])
    if isinstance(node.func, ast.Attribute) and node.func.attr == "managed_session":
        return convert_origin_func_to_npu(node, tf_func_map["tf.train.Supervisor.managed_session"],
                                          "managed_session", ["config"], True)
    if distributed_mode == "tf_strategy":  # this cond should be placed at the end of the Call function.
        return convert_distributed_strategy_apis(node)
    return node


def _call_name_match(call_func, call_name):
    return (isinstance(call_func, ast.Attribute) and (call_func.attr == call_name)) or \
           (isinstance(call_func, ast.Name) and call_func.id == call_name)


def insert_npu_import(r_node):
    """Add NPU import modules"""
    npu_alias = ast.alias(name='*', asname=None)
    npu_import = ast.ImportFrom(module='npu_bridge.npu_init', names=[npu_alias], level=0)
    num = 5 if len(r_node.body) >= 5 else len(r_node.body)
    import_index = 0
    is_insert = False
    for i in range(0, num):
        if isinstance(r_node.body[i], ast.Import):
            r_node.body.insert(i, npu_import)
            log_msg(i, "from npu_bridge.npu_init import *")
            is_insert = True
            break
        if isinstance(r_node.body[i], ast.ImportFrom):
            if r_node.body[i].module != "__future__":
                r_node.body.insert(i, npu_import)
                log_msg(i, "from npu_bridge.npu_init import *")
                is_insert = True
                break
            import_index = i + 1
    if not is_insert:
        r_node.body.insert(import_index, npu_import)
        log_msg(import_index, "from npu_bridge.npu_init import *")


def insert_keras_dropout_import(r_node):
    """Add keras dropout import module"""
    npu_alias = ast.alias(name='npu_convert_dropout', asname=None)
    npu_import = ast.ImportFrom(module='npu_bridge.estimator.npu', names=[npu_alias], level=0)
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and (isinstance(r_node.body[n], ast.ImportFrom)):
        n += 1

    r_node.body.insert(n, npu_import)
    log_msg(n, "from npu_bridge.estimator.npu import npu_convert_dropout")


def insert_npu_resource_init(r_node):
    """Add NPU resource initial module"""
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and isinstance(r_node.body[n], (ast.ImportFrom, ast.Import)):
        n += 1

    if n < lenline:
        init_assign = ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id="npu_sess", ctx=ast.Store()),
                                                          ast.Name(id="npu_shutdown", ctx=ast.Store())],
                                                    ctx=ast.Store())],
                                 value=ast.Call(func=ast.Name(id="init_resource", ctx=ast.Load()), args=[],
                                                keywords=[]))
        r_node.body.insert(n, init_assign)


def insert_npu_resource_shutdown(r_node):
    """Add NPU resource shutdown module"""
    shutdown_call = ast.Expr(value=ast.Call(func=ast.Name(id="shutdown_resource", ctx=ast.Load()),
                                            args=[ast.Name(id="npu_sess", ctx=ast.Load()),
                                                  ast.Name(id="npu_shutdown", ctx=ast.Load())],
                                            keywords=[]))
    close_sess_call = ast.Expr(value=ast.Call(func=ast.Name(id="close_session", ctx=ast.Load()),
                                              args=[ast.Name(id="npu_sess", ctx=ast.Load())], keywords=[]))
    r_node.body.append(shutdown_call)
    r_node.body.append(close_sess_call)


def insert_keras_sess_npu_config(r_node):
    """Add NPU configuration for keras session"""
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and isinstance(r_node.body[n], (ast.ImportFrom, ast.Import)):
        n += 1

    if n < lenline:
        keras_sess_assign = ast.Assign(targets=[ast.Name(id="npu_keras_sess", ctx=ast.Store())],
                                       value=ast.Call(func=ast.Name(id="set_keras_session_npu_config", ctx=ast.Load()),
                                                      args=[], keywords=[]))
        r_node.body.insert(n, keras_sess_assign)


def insert_keras_sess_close(r_node):
    """Add closing for keras session"""
    close_sess_call = ast.Expr(value=ast.Call(func=ast.Name(id="close_session", ctx=ast.Load()),
                                              args=[ast.Name(id="npu_keras_sess", ctx=ast.Load())], keywords=[]))
    r_node.body.append(close_sess_call)


def node_tree(node: str):
    """Format printing for locate"""
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
