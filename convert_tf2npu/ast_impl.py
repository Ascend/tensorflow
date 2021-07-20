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
import ast
import copy
import pasta
import util_global
from util import *

def attribute(node):
    log_success_report(getattr(node, "lineno", "None"), node.attr)
    node = ast.Name(id=util_global.get_value(node.attr)[0], ctx=ast.Load())
    util_global.set_value('need_conver', True)
    return node

def import_from(node):
    if node.module != "":
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
    log_success_report(getattr(node, "lineno", "None"), node.name)
    arg_name = node.args.args[0].arg
    node.body = [ast.Return(value=ast.Call(
                                            func=ast.Attribute(value=ast.Name(id=util_global.get_value(node.name)[0],
                                                               ctx=ast.Load()), attr='gelu',
                                                               ctx=ast.Load()),
                                            args=[ast.Name(id=arg_name, ctx=ast.Load())],
                                            keywords=[]))]

    util_global.set_value('need_conver', True)
    return node

def ast_if(node):
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
                                                   value=ast.Call(func=ast.Name(id="set_keras_session_npu_config", ctx=ast.Load()),
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
                                             value=ast.Call(func=ast.Name(id="init_resource", ctx=ast.Load()), args=[], keywords=[]))
                    shutdown_call = ast.Call(func=ast.Name(id="shutdown_resource", ctx=ast.Load()),
                                             args=[ast.Name(id="npu_sess", ctx=ast.Load()), ast.Name(id="npu_shutdown", ctx=ast.Load())],
                                             keywords=[])
                    node.body = [init_assign] + node.body + [ast.Expr(value=shutdown_call), ast.Expr(value=close_sess_call)]
                    util_global.set_value('need_conver', True)
                return node

def convert_loss_scale_api(node):
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
            log_msg(getattr(node, 'lineno', 'None'), "change tf.train.experimental.DynamicLossScale"
                    " to ExponentialUpdateLossScaleManager")
            node.func = ast.Name(id="ExponentialUpdateLossScaleManager", ctx=ast.Load())
            initial_loss_scale = None
            increment_period = None
            multiplier = None
            for index, arg in enumerate(node.args):
                if index == 0: initial_loss_scale = arg
                if index == 1: increment_period = arg
                if index == 2: multiplier = arg
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
        if node.func.attr == "MixedPrecisionLossScaleOptimizer":
            log_msg(getattr(node, 'lineno', 'None'), "change tf.train.experimental.MixedPrecisionLossScaleOptimizer"
                    " to NPULossScaleOptimizer")
            node.func = ast.Name(id="NPULossScaleOptimizer", ctx=ast.Load())
            for keyword in node.keywords:
                if keyword.arg == "loss_scale": keyword.arg = "loss_scale_manager"
            util_global.set_value('need_conver', True)
            return node

def convert_monitor_session_api(node):
    log_msg(getattr(node, "lineno", "None"), 'MonitoredTrainingSession add npu_config_proto func')
    config = None
    hooks = None
    for index, _ in enumerate(node.args):
        if index == 4: # The index of hooks in the MonitoredTrainingSession is 4
            hooks = node.args.pop(4)
        if index == 9: # The index of config in the MonitoredTrainingSession is 9
            config = node.args.pop(9)
            break
    for keyword in node.keywords:
        if keyword.arg == 'hooks':
            hooks = keyword
        if keyword.arg == "config":
            config = keyword
    if config:
        if isinstance(config, ast.keyword):
            new_value = ast.Call(func=ast.Name(id='npu_config_proto', ctx=ast.Load()), args=[],
                                    keywords=[ast.keyword(arg='config_proto', value=config.value)])
            ast.copy_location(new_value, config.value)
            config.value = new_value
        else:
            node.keywords.append(ast.keyword(arg='config', value=ast.Call(
                func=ast.Name(id='npu_config_proto', ctx=ast.Load()), args=[],
                keywords=[ast.keyword(arg='config_proto', value=config)])))
    else:
        node.keywords.append(ast.keyword(arg='config', value=pasta.parse('npu_config_proto()')))

    if util_global.get_value("distributed_mode", "") == "tf_strategy":
        log_msg(getattr(node, "lineno", "None"), 'add npu broadcast hook to hooks list')
        if not hooks:
            node.keywords.append(ast.keyword(arg='hooks', value=pasta.parse('npu_hooks_append()')))
        elif isinstance(hooks, ast.keyword):
            new_value = ast.Call(func=ast.Name(id='npu_hooks_append', ctx=ast.Load()), args=[],
                                 keywords=[ast.keyword(arg='hooks_list', value=hooks.value)])
            ast.copy_location(new_value, hooks.value)
            hooks.value = new_value
        else:
            node.keywords.append(ast.keyword(arg='hooks', value=ast.Call(func=ast.Name(id='npu_hooks_append', ctx=ast.Load()),
                                                                         args=[], keywords=[ast.keyword(arg='hooks_list', value=hooks)])))

    util_global.set_value('need_conver', True)
    return node

def convert_managed_session_api(node):
    log_msg(getattr(node, 'lineno', 'None'), "add npu_config_proto func in managed_session")
    config = None
    for index, _ in enumerate(node.args):
        if index == 1: # The index of config in the managed_session is 1
            config = node.args.pop(1)
            break
    for keyword in node.keywords:
        if keyword.arg == "config":
            config = keyword

    if config:
        if isinstance(config, ast.keyword):
            new_value = ast.Call(func=ast.Name(id='npu_config_proto', ctx=ast.Load()), args=[],
                                 keywords=[ast.keyword(arg='config_proto', value=config.value)])
            ast.copy_location(new_value, config.value)
            config.value = new_value
        else:
            node.keywords.append(ast.keyword(arg='config', value=ast.Call(
                func=ast.Name(id='npu_config_proto', ctx=ast.Load()), args=[],
                keywords=[ast.keyword(arg='config_proto', value=config)])))
    else:
        node.keywords.append(ast.keyword(arg='config', value=pasta.parse('npu_config_proto()')))
    util_global.set_value('need_conver', True)
    return node

def convert_hvd_distributed_api(node):
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

def convert_estimator_train_api(node):
    if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "learning":
        return node
    train_keywords = ["input_fn", "hooks", "steps", "max_steps", "saving_listeners"]
    if len(node.keywords) == 0:
        if len(node.args) > len(train_keywords):
            return node
        else:
            content = "".join([util_global.get_value('path'), ":", str(getattr(node, "lineno", "None"))])
            while True:
                message = input("Check if the train function in " + content + " is the Estimator train function. If yes, "
                                "enter 'y' to perform distributed porting on the train function. if no, enter 'n': " )
                if message == "y":
                    break
                elif message == "n":
                    log_warning("".join(["The train func in ", content, " is user-defined functions, will not perform distributed porting"]))
                    return node
                else:
                    print("Input is error, Please enter 'y' or 'n'.")
    else:
        for keyword in node.keywords:
            if keyword.arg not in train_keywords:
                return node

    hooks = None
    for index, _ in enumerate(node.args):
        if index == 1: # The index of hooks in the train is 1
            hooks = node.args.pop(1)
    for keyword in node.keywords:
        if keyword.arg not in train_keywords:
            return node
        elif keyword.arg == 'hooks':
            hooks = keyword
    log_msg(getattr(node, "lineno", "None"), 'add npu broadcast hook to hooks list')
    if not hooks:
        node.keywords.append(ast.keyword(arg='hooks', value=pasta.parse('npu_hooks_append()')))
    elif isinstance(hooks, ast.keyword):
        new_value = ast.Call(func=ast.Name(id='npu_hooks_append', ctx=ast.Load()), args=[],
                             keywords=[ast.keyword(arg='hooks_list', value=hooks.value)])
        ast.copy_location(new_value, hooks.value)
        hooks.value = new_value
    else:
        node.keywords.append(ast.keyword(arg='hooks',
                                         value=ast.Call(func=ast.Name(id='npu_hooks_append', ctx=ast.Load()),
                                                        args=[], keywords=[ast.keyword(arg='hooks_list', value=hooks)])))
    util_global.set_value('need_conver', True)
    return node

def convert_model_compile_api(node):
    opt_map = {"adadelta": "tf.keras.optimizers.Adadelta()",
               "adagrad": "tf.keras.optimizers.Adagrad()",
               "adam": "tf.keras.optimizers.Adam()",
               "adamax": "tf.keras.optimizers.Adamax()",
               "ftrl": "tf.keras.optimizers.Ftrl()",
               "nadam": "tf.keras.optimizers.Nadam()",
               "rmsprop": "tf.keras.optimizers.RMSprop()",
               "sgd": "tf.keras.optimizers.SGD()"}
    optimizer = None
    for keyword in node.keywords:
        if keyword.arg == "optimizer":
            optimizer = keyword
            log_msg(getattr(node, 'lineno', 'None'), 'add npu distribute optimizer to keras optimizer')
            if isinstance(keyword.value, ast.Str):
                npu_keras_opt = "".join(["npu_distributed_optimizer_wrapper(", opt_map[keyword.value.s], ")"])
                keyword.value = pasta.parse(npu_keras_opt)
            else:
                new_value = ast.Call(func=ast.Name(id="npu_distributed_optimizer_wrapper", ctx=ast.Load()), args=[keyword.value], keywords=[])
                ast.copy_location(new_value, keyword.value)
                keyword.value = new_value
            util_global.set_value('need_conver', True)
            break
    if optimizer is None:
        optimizer = node.args[0] # The index of optimizer in the compile is 0
        log_msg(getattr(node, 'lineno', 'None'), 'add npu distribute optimizer to keras optimizer')
        if isinstance(optimizer, ast.Str):
            npu_keras_opt = "".join(["npu_distributed_optimizer_wrapper(", opt_map[optimizer.s], ")"])
            node.args[0] = pasta.parse(npu_keras_opt)
        else:
            new_value = ast.Call(func=ast.Name(id="npu_distributed_optimizer_wrapper", ctx=ast.Load()), args=[optimizer], keywords=[])
            ast.copy_location(new_value, node.args[0])
            node.args[0] = new_value
    return node

def convert_model_fit_apis(node, callbacks_index):
    callbacks = None
    for index, _ in enumerate(node.args):
        if index == callbacks_index:
            callbacks = node.args.pop(callbacks_index)
            break
    for keyword in node.keywords:
        if keyword.arg == 'callbacks':
            callbacks = keyword
            break

    log_msg(getattr(node, "lineno", "None"), 'add npu broadcast callback to callbacks list')
    if not callbacks:
        node.keywords.append(ast.keyword(arg='callbacks', value=pasta.parse('npu_callbacks_append()')))
    elif isinstance(callbacks, ast.keyword):
        new_value = ast.Call(func=ast.Name(id='npu_callbacks_append', ctx=ast.Load()), args=[], keywords=[
            ast.keyword(arg='callbacks_list', value=callbacks.value)])
        ast.copy_location(new_value, callbacks.value)
        callbacks.value = new_value
    else:
        node.keywords.append(ast.keyword(arg='callbacks', value=ast.Call(func=ast.Name(id='npu_callbacks_append', ctx=ast.Load()),
                                                                     args=[], keywords=[ast.keyword(arg='callbacks_list', value=callbacks)])))
    util_global.set_value('need_conver', True)
    return node

def convert_train_eval_spec_api(node, hooks_index):
    hooks = None
    for index, _ in enumerate(node.args):
        if index == hooks_index:
            hooks = node.args.pop(hooks_index)
            break
    for keyword in node.keywords:
        if keyword.arg == 'hooks':
            hooks = keyword
            break

    log_msg(getattr(node, "lineno", "None"), 'add npu broadcast hook to hooks list')
    if not hooks:
        node.keywords.append(ast.keyword(arg='hooks', value=pasta.parse('npu_hooks_append()')))
    elif isinstance(hooks, ast.keyword):
        new_value = ast.Call(func=ast.Name(id='npu_hooks_append', ctx=ast.Load()), args=[], keywords=[
            ast.keyword(arg='hooks_list', value=hooks.value)])
        ast.copy_location(new_value, hooks.value)
        hooks.value = new_value
    else:
        node.keywords.append(ast.keyword(arg='hooks', value=ast.Call(func=ast.Name(id='npu_hooks_append', ctx=ast.Load()),
                                                                     args=[], keywords=[ast.keyword(arg='hooks_list', value=hooks)])))
    util_global.set_value('need_conver', True)
    return node

def convert_tf_gradient_distributed(node):
    content = "".join([util_global.get_value('path'), ":", str(getattr(node, "lineno", "None")),
                       " is tf.gradient api, tool inserts allreduce op after computing grads by default.",
                       " You can adjust the allreduce position according to the algorithm"])
    log_warning(content)
    new_node = ast.Call(func=ast.Name(id="npu_allreduce", ctx=ast.Load()), args=[node], keywords=[])
    ast.copy_location(new_node, node)
    util_global.set_value("need_conver", True)
    return new_node

def convert_distributed_strategy_apis(node):
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
        if (node.func.attr.find("Optimizer") != -1 and node.func.attr != "ScipyOptimizerInterface" and
            node.func.attr != "MixedPrecisionLossScaleOptimizer"):
            log_msg(getattr(node, "lineno", "None"), "add npu distribute optimizer to tensorflow optimizer")
            new_node = ast.Call(func=ast.Name(id="npu_distributed_optimizer_wrapper", ctx=ast.Load()), args=[node], keywords=[])
            ast.copy_location(new_node, node)
            util_global.set_value('need_conver', True)
            return new_node
    if isinstance(node.func, ast.Name) and node.func.id.find("Optimizer") != -1 and node.func.id != "NPULossScaleOptimizer":
        log_msg(getattr(node, "lineno", "None"), "add npu distribute optimizer to tensorflow optimizer")
        new_node = ast.Call(func=ast.Name(id="npu_distributed_optimizer_wrapper", ctx=ast.Load()), args=[node], keywords=[])
        ast.copy_location(new_node, node)
        util_global.set_value('need_conver', True)
        return new_node
    specs = {'TrainSpec': 2, 'EvalSpec': 3} # the num is hooks index in api
    for spec, hooks_index in specs.items():
        if _call_name_match(node.func, spec):
            return convert_train_eval_spec_api(node, hooks_index)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "train":
        return convert_estimator_train_api(node)
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'compile'):
        return convert_model_compile_api(node)
    callbacks_indexs = {'fit': 5, 'fit_generator': 4} # the num is callbacks index in api
    for fit_api, callbacks_index in callbacks_indexs.items():
        if isinstance(node.func, ast.Attribute) and node.func.attr == fit_api:
            return convert_model_fit_apis(node, callbacks_index)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "gradients" and \
       isinstance(node.func.value, ast.Name) and node.func.value.id == "tf":
        return convert_tf_gradient_distributed(node)
    return node

def ast_call(node):
    distributed_mode = util_global.get_value("distributed_mode", "")
    is_not_strategy = (distributed_mode == "horovod" or distributed_mode == "")
    is_not_horovod = (distributed_mode == "tf_strategy" or distributed_mode == "")
    convert_loss_scale_api(node)
    if _call_name_match(node.func, "set_experimental_options"):
        log_msg(getattr(node, 'lineno', 'None'), 'change set_experimental_options(*) to set_experimental_options(experimental_options)')
        node.args = [ast.Name(id='experimental_options', ctx=ast.Load())]
        node.keywords = []
        util_global.set_value('need_conver', True)
    if isinstance(node.func, ast.Name) and node.func.id == 'check_available_gpus':
        log_msg(getattr(node, 'lineno', 'None'), "change check_available_gpus() to ['/device:CPU:0']")
        util_global.set_value('need_conver', True)
        return ast.List(elts=[ast.Str(s="/device:CPU:0")], ctx=ast.Load())
    if (isinstance(node.func, ast.Name) and node.func.id == 'GraphOptions') or \
       (isinstance(node.func, ast.Attribute) and node.func.attr == 'GraphOptions'):
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
    if (isinstance(node.func, ast.Name) and (node.func.id == 'Session' or node.func.id == 'InteractiveSession')) or \
       (isinstance(node.func, ast.Attribute) and (node.func.attr == 'Session' or node.func.attr == 'InteractiveSession')):
        log_success_report(getattr(node, 'lineno', 'None'), 'Session()')
        config = None
        for index, _ in enumerate(node.args):
            if index == 2:
                config = node.args.pop(2)
                break
        for keyword in node.keywords:
            if keyword.arg == 'config':
                config = keyword
                break
        if config:
            if isinstance(config, ast.keyword):
                config.value = ast.Call(
                    func=ast.Name(id='npu_config_proto', ctx=ast.Load()),
                    args=[],
                    keywords=[ast.keyword(arg='config_proto', value=config.value)])
            else:
                node.keywords.append(ast.keyword(arg='config', value=ast.Call(
                    func=ast.Name(id='npu_config_proto', ctx=ast.Load()),
                    args=[],
                    keywords=[ast.keyword(arg='config_proto', value=config)])))
        else:
            node.keywords.append(ast.keyword(arg='config', value=ast.Call(
                func=ast.Name(id='npu_config_proto', ctx=ast.Load()),
                args=[],
                keywords=[])))
        util_global.set_value('need_conver', True)
        return node
    if isinstance(node.func, ast.Attribute) and node.func.attr == "BroadcastGlobalVariablesHook":
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "hvd":
            if is_not_horovod:
                log_strategy_distributed_mode_error(node)
                return node
            log_msg(getattr(node, "lineno", "None"), 'change hvd.BroadcastGlobalVariablesHook to NPUBroadcastGlobalVariablesHook')
            node = pasta.parse("NPUBroadcastGlobalVariablesHook(0, int(os.getenv('RANK_ID', '0')))")
            util_global.set_value('need_conver', True)
            return node
    if isinstance(node.func, ast.Attribute) and node.func.attr == "BroadcastGlobalVariablesCallback":
        if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "callbacks":
            if is_not_horovod:
                log_strategy_distributed_mode_error(node)
                return node
            log_msg(getattr(node, "lineno", "None"), 'change hvd.callbacks.BroadcastGlobalVariablesCallback to NPUBroadcastGlobalVariablesCallback')
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
            node.func=ast.Attribute(value=ast.Name(id='npu_ops', ctx=ast.Load()), attr='dropout', ctx=ast.Load())
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
    if isinstance(node.func, ast.Attribute) and ((node.func.attr == 'map_and_batch') or (node.func.attr == 'batch'
        and (not isinstance(node.func.value, ast.Attribute) or (isinstance(node.func.value, ast.Attribute) and node.func.value.attr != 'train')))):
        exist = False
        for keyword in node.keywords:
            if keyword.arg == 'drop_remainder':
                exist = True
                if ((isinstance(keyword.value, ast.NameConstant) and keyword.value.value != True) or
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
    if isinstance(node.func, ast.Attribute) and (node.func.attr == "get_distribution_strategy" or
        node.func.attr == "MirroredStrategy" or node.func.attr == "MultiWorkerMirroredStrategy"):
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
                if (not isinstance(keyword.value, ast.NameConstant)) or (isinstance(keyword.value, ast.NameConstant) and (keyword.value.value != False)):
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
            log_msg(getattr(node, 'lineno'), "".join([estimator , '() add config=npu_run_config_init()']))
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
    if (isinstance(node.func, ast.Attribute) and (node.func.attr == 'MonitoredTrainingSession')) or \
       (isinstance(node.func, ast.Name) and (node.func.id == 'MonitoredTrainingSession')):
        return convert_monitor_session_api(node)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "managed_session":
        return convert_managed_session_api(node)
    if distributed_mode == "tf_strategy": # this cond should be placed at the end of the Call function.
        return convert_distributed_strategy_apis(node)
    return node

def _call_name_match(call_func, call_name):
    return (isinstance(call_func, ast.Attribute) and (call_func.attr == call_name)) or \
           (isinstance(call_func, ast.Name) and call_func.id == call_name)

def insert_npu_import(r_node):
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
        elif isinstance(r_node.body[i], ast.ImportFrom):
            if r_node.body[i].module != "__future__":
                r_node.body.insert(i, npu_import)
                log_msg(i, "from npu_bridge.npu_init import *")
                is_insert = True
                break
            else:
                import_index = i + 1
    if not is_insert:
        r_node.body.insert(import_index, npu_import)
        log_msg(import_index, "from npu_bridge.npu_init import *")

def insert_npu_resource_init(r_node):
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and (isinstance(r_node.body[n], ast.ImportFrom) or isinstance(r_node.body[n], ast.Import)):
        n += 1

    if n < lenline:
        init_assign = ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id="npu_sess", ctx=ast.Store()),
                                                          ast.Name(id="npu_shutdown", ctx=ast.Store())],
                                                    ctx=ast.Store())],
                                 value=ast.Call(func=ast.Name(id="init_resource", ctx=ast.Load()), args=[], keywords=[]))
        r_node.body.insert(n, init_assign)

def insert_npu_resource_shutdown(r_node):
    shutdown_call = ast.Expr(value=ast.Call(func=ast.Name(id="shutdown_resource", ctx=ast.Load()),
                                            args=[ast.Name(id="npu_sess", ctx=ast.Load()), ast.Name(id="npu_shutdown", ctx=ast.Load())],
                                            keywords=[]))
    close_sess_call = ast.Expr(value=ast.Call(func=ast.Name(id="close_session", ctx=ast.Load()),
                                              args=[ast.Name(id="npu_sess", ctx=ast.Load())], keywords=[]))
    r_node.body.append(shutdown_call)
    r_node.body.append(close_sess_call)

def insert_keras_sess_npu_config(r_node):
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and (isinstance(r_node.body[n], ast.ImportFrom) or isinstance(r_node.body[n], ast.Import)):
        n += 1

    if n < lenline:
        keras_sess_assign = ast.Assign(targets=[ast.Name(id="npu_keras_sess", ctx=ast.Store())],
                                       value=ast.Call(func=ast.Name(id="set_keras_session_npu_config", ctx=ast.Load()),
                                                      args=[], keywords=[]))
        r_node.body.insert(n, keras_sess_assign)

def insert_keras_sess_close(r_node):
    close_sess_call = ast.Expr(value=ast.Call(func=ast.Name(id="close_session", ctx=ast.Load()),
                                              args=[ast.Name(id="npu_keras_sess", ctx=ast.Load())], keywords=[]))
    r_node.body.append(close_sess_call)

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