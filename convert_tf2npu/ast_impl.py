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
from util import log_success_report
from util import log_migration_report
from util import log_msg

def attribute(node):
    log_success_report(getattr(node, "lineno", "None"), node.attr)
    node = ast.Name(id=util_global.get_value(node.attr)[0], ctx=ast.Load())
    util_global.set_value('need_conver', True)
    return node

def import_from(node):
    if node.module != None:
        values = node.module.split(".")
        if "keras" in values:
            util_global.set_value('is_keras_net', True)
        if "horovod" in values:
            log_msg(getattr(node, "lineno", "None"), "remove horovod import line to None")
            util_global.set_value('has_hccl_api', True)
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
                util_global.set_value('has_hccl_api', True)
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
                util_global.set_value('has_hccl_api', True)
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
                    log_msg(getattr(node, "lineno", "None"), " add keras session npu config")
                    close_sess_call = ast.Call(func=ast.Name(id="close_session", ctx=ast.Load()),
                                               args=[ast.Name(id="npu_keras_sess", ctx=ast.Load())], keywords=[])
                    keras_sess_assign = ast.Assign(targets=[ast.Name(id="npu_keras_sess", ctx=ast.Store())],
                                                   value=ast.Call(func=ast.Name(id="set_keras_session_npu_config", ctx=ast.Load()),
                                                                  args=[], keywords=[]))
                    node.body = [keras_sess_assign] + node.body + [ast.Expr(value=close_sess_call)]
                    util_global.set_value('need_conver', True)
                log_msg(getattr(node, "lineno", "None"), " add npu resource init api")
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
    if (isinstance(node.func, ast.Attribute) and (node.func.attr == 'MonitoredTrainingSession')) or \
       (isinstance(node.func, ast.Name) and (node.func.id == 'MonitoredTrainingSession')):
        log_success_report(getattr(node, "lineno", "None"), 'MonitoredTrainingSession')
        hooks = None
        config = None
        for index, _ in enumerate(node.args):
            if index == 4:
                hooks = node.args.pop(4)
            if index == 9:
                config = node.args.pop(9)
                break
        for keyword in node.keywords:
            if keyword.arg == 'hooks':
                hooks = keyword
            if keyword.arg == "config":
                config = keyword
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

def convert_managed_session_api(node):
    if isinstance(node.func, ast.Attribute) and node.func.attr == "managed_session":
        log_msg(getattr(node, 'lineno', 'None'), "add npu_config_proto func in managed_session")
        config = None
        for index, _ in enumerate(node.args):
            if index == 1:
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

def ast_call(node):
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
        log_success_report(getattr(node, "lineno", "None"), 'BroadcastGlobalVariablesHook')
        node.func = ast.Name(id="NpuEmptyHook", ctx=ast.Load())
        node.args = []
        node.keywords = []
        util_global.set_value('need_conver', True)
        return node
    if isinstance(node.func, ast.Attribute) and node.func.attr == "DistributedOptimizer":
        log_msg(getattr(node, "lineno", "None"), 'change hvd.DistributedOptimizer to the input key optimzier')
        opt_keyword = None
        for keyword in node.keywords:
            if keyword.arg == "optimizer":
                opt_keyword = keyword
        if opt_keyword is None:
            return node.args[0]
        else:
            return opt_keyword.value
    if isinstance(node.func, ast.Attribute) and node.func.attr == 'shard':
        log_success_report(getattr(node, "lineno", "None"), 'shard')
        node.args = [pasta.parse("int(os.getenv('RANK_SIZE', '1'))"),
                     pasta.parse("int(os.getenv('RANK_ID', '0'))")]
        node.keywords.clear()
        util_global.set_value('need_conver', True)
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
    if isinstance(node.func, ast.Attribute) and ((node.func.attr == 'map_and_batch') or (node.func.attr == 'batch' \
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
    if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
        node.func.value.id == 'tf' and node.func.attr == 'device'):
        log_success_report(getattr(node, "lineno", "None"), node.func.attr)
        node.args = [ast.Str(s='/cpu:0')]
        util_global.set_value('need_conver', True)
    if isinstance(node.func, ast.Attribute) and (node.func.attr == "get_distribution_strategy" or
        node.func.attr == "MirroredStrategy" or node.func.attr == "MultiWorkerMirroredStrategy"):
        log_success_report(getattr(node, "lineno", "None"), node.func.attr)
        new_func = ast.Attribute(value=ast.Name(id="npu_strategy", ctx=ast.Load()),
                                  attr="NPUStrategy", ctx=ast.Load())
        ast.copy_location(new_func, node.func)
        node.func = new_func
        node.keywords = []
        node.args = []
        util_global.set_value('need_conver', True)
    if (isinstance(node.func, ast.Attribute) and (node.func.attr == 'RunConfig')) and \
        (_call_name_match(node.func.value, 'estimator') or _call_name_match(node.func.value, 'tpu')):
        save_summary_steps = None
        for keyword in node.keywords:
            if (keyword.arg == 'save_summary_steps'):
                save_summary_steps = keyword
                break
        if len(node.args) < 3 and not save_summary_steps:
            log_msg(getattr(node, 'lineno'), 'RunConfig() add save_summary_steps=0')
            util_global.set_value('need_conver', True)
            node.keywords.append(ast.keyword(arg='save_summary_steps', value=pasta.parse('0')))
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'TPUEstimator') and \
        ((isinstance(node.func.value, ast.Attribute) and (node.func.value.attr == 'tpu')) or \
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
            log_msg(getattr(node, 'lineno'), estimator + '() add config=npu_run_config_init()')
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
    for estimator_func in util_global.get_value('EstimatorFunc', []):
        if isinstance(node.func, ast.Attribute) and (node.func.attr == estimator_func):
            if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "learning":
                return node
            train_keywords = ["input_fn", "hooks", "steps", "max_steps", "saving_listeners"]
            if len(node.keywords) == 0 and len(node.args) != 5:
                return node
            input_fn = None
            hooks = None
            for index, _ in enumerate(node.args):
                if index == 0:
                    input_fn = node.args[0]
                elif index == 1:
                    hooks = node.args.pop(1)
            for keyword in node.keywords:
                if keyword.arg not in train_keywords:
                    return node
                if keyword.arg == 'input_fn':
                    input_fn = keyword
                elif keyword.arg == 'hooks':
                    hooks = keyword
            if not input_fn:
                break
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
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'compile'):
        opt_map = {"adadelta": "tf.keras.optimizers.Adadelta()",
                   "adagrad": "tf.keras.optimizers.Adagrad()",
                   "adam": "tf.keras.optimizers.Adam()",
                   "adamax": "tf.keras.optimizers.Adamax()",
                   "ftrl": "tf.keras.optimizers.Ftrl()",
                   "nadam": "tf.keras.optimizers.Nadam()",
                   "rmsprop": "tf.keras.optimizers.RMSprop()",
                   "sgd": "tf.keras.optimizers.SGD()"}
        for keyword in node.keywords:
            if keyword.arg == "optimizer":
                log_success_report(getattr(node, 'lineno', 'None'), 'KerasDistributeOptimizer')
                if isinstance(keyword.value, ast.Str):
                    keras_opt = opt_map[keyword.value.s]
                    npu_keras_opt = "npu_keras_optimizer(" + keras_opt + ")"
                    keyword.value = pasta.parse(npu_keras_opt)
                util_global.set_value('need_conver', True)
                return node
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
        if (node.func.attr.find("Optimizer") != -1) and (node.func.attr != 'ScipyOptimizerInterface'):
            log_msg(getattr(node, "lineno", "None"), "add NPUDistributedOptimizer()")
            new_node = ast.Call(func=ast.Name(id="npu_tf_optimizer", ctx=ast.Load()), args=[node], keywords=[])
            ast.copy_location(new_node, node)
            util_global.set_value('need_conver', True)
            return new_node
    if isinstance(node.func, ast.Attribute):
        opt_list = ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop", "SGD"]
        if node.func.attr in opt_list:
            log_success_report(getattr(node, "lineno", "None"), "KerasDistributeOptimizer")
            new_node = ast.Call(func=ast.Name(id="npu_keras_optimizer", ctx=ast.Load()), args=[node], keywords=[])
            ast.copy_location(new_node, node)
            util_global.set_value('need_conver', True)
            return new_node
    convert_monitor_session_api(node)
    convert_managed_session_api(node)
    specs = {'TrainSpec': 2, 'EvalSpec': 3}
    for spec, hooks_index in specs.items():
        if _call_name_match(node.func, spec):
            log_success_report(getattr(node, "lineno", "None"), spec)
            hooks = None
            for index, _ in enumerate(node.args):
                if index == hooks_index:
                    hooks = node.args.pop(hooks_index)
                    break
            for keyword in node.keywords:
                if keyword.arg == 'hooks':
                    hooks = keyword
                    break
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

def _call_name_match(call_func, call_name):
    return (isinstance(call_func, ast.Attribute) and (call_func.attr == call_name)) or \
           (isinstance(call_func, ast.Name) and (call_func.id) == call_name)

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