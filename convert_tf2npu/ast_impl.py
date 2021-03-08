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
import copy
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
            util_global.set_value('is_hvd_net', True)


def ast_import(node):
    for value in node.names:
        if isinstance(value, ast.alias):
            values = value.name.split(".")
            if "keras" in values:
                util_global.set_value('is_keras_net', True)
            if "horovod" in values:
                util_global.set_value('is_hvd_net', True)

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
                if util_global.get_value("is_keras_net", False):
                    log_msg(getattr(node, "lineno", "None"), " add keras session npu config")
                    close_sess_call = ast.Call(func=ast.Name(id="close_session", ctx=ast.Load()),
                                               args=[ast.Name(id="npu_keras_sess", ctx=ast.Load())], keywords=[])
                    keras_sess_assign = ast.Assign(targets=[ast.Name(id="npu_keras_sess", ctx=ast.Store())],
                                                   value=ast.Call(func=ast.Name(id="set_keras_session_npu_config", ctx=ast.Load()),
                                                                  args=[], keywords=[]))
                    try_node = ast.Try(body=[keras_sess_assign, node.body], handlers=[], orelse=[],
                                       finalbody=[ast.Expr(value=close_sess_call)])
                    node.body = [try_node]
                    util_global.set_value('need_conver', True)
                if util_global.get_value("is_hvd_net", False):
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
                    try_node = ast.Try(body=[init_assign, node.body], handlers=[], orelse=[],
                                       finalbody=[ast.Expr(value=shutdown_call), ast.Expr(value=close_sess_call)])
                    node.body = [try_node]
                    util_global.set_value('need_conver', True)
                return node

def ast_call(node):
    if (isinstance(node.func, ast.Name) and node.func.id == 'ConfigProto') or \
       (isinstance(node.func, ast.Attribute) and node.func.attr == 'ConfigProto'):
        log_success_report(getattr(node, 'lineno', 'None'), 'ConfigProto()')
        src = copy.deepcopy(node)
        node.func = ast.Name(id='npu_config_proto', ctx=ast.Load())
        node.args = []
        node.keywords = []
        node.keywords.append(ast.keyword(arg='config_proto', value=src))
        util_global.set_value('need_conver', True)
        return node
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
    if (isinstance(node.func, ast.Name) and node.func.id == 'Session') or \
       (isinstance(node.func, ast.Attribute) and node.func.attr == 'Session'):
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
                    func=ast.Name(id='npu_session_config_init', ctx=ast.Load()),
                    args=[],
                    keywords=[ast.keyword(arg='session_config', value=config.value)])
            else:
                node.keywords.append(ast.keyword(arg='config', value=ast.Call(
                    func=ast.Name(id='npu_session_config_init', ctx=ast.Load()),
                    args=[],
                    keywords=[ast.keyword(arg='session_config', value=config)])))
        else:
            node.keywords.append(ast.keyword(arg='config', value=ast.Call(
                func=ast.Name(id='npu_session_config_init', ctx=ast.Load()),
                args=[],
                keywords=[])))
        util_global.set_value('need_conver', True)
        util_global.set_value('insert_npu_session_config_func', True)
        return node
    if isinstance(node.func, ast.Attribute) and node.func.attr == "BroadcastGlobalVariablesHook":
        log_success_report(getattr(node, "lineno", "None"), 'BroadcastGlobalVariablesHook')
        node.func = ast.Name(id="NpuEmptyHook", ctx=ast.Load())
        node.args = []
        node.keywords = []
        util_global.set_value('need_conver', True)
        util_global.set_value('insert_empty_hook', True)
        return node
    if isinstance(node.func, ast.Attribute) and node.func.attr == "DistributedOptimizer":
        log_success_report(getattr(node, "lineno", "None"), 'DistributedOptimizer')
        return node.args[0]
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
    if isinstance(node.func, ast.Attribute) and ((node.func.attr == 'map_and_batch') or (node.func.attr == 'batch' \
        and (not isinstance(node.func.value, ast.Attribute) or (isinstance(node.func.value, ast.Attribute) and node.func.value.attr != 'train')))):
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
        add_eval_on_tpu = True
        add_use_tpu = True
        add_export_to_tpu = True
        for keyword in node.keywords:
            if (keyword.arg == 'eval_on_tpu') or (keyword.arg == 'use_tpu') or (keyword.arg == 'export_to_tpu'):
                if (not isinstance(keyword.value, ast.NameConstant)) or (isinstance(keyword.value, ast.NameConstant) and (keyword.value.value != False)):
                    log_success_report(getattr(node, 'lineno', 'None'), 'TPUEstimator(' + keyword.arg + '=*)')
                    keyword.value = ast.NameConstant(value=False)
                    util_global.set_value('need_conver', True)
                if add_eval_on_tpu and (keyword.arg == 'eval_on_tpu'):
                    add_eval_on_tpu = False
                if add_use_tpu and (keyword.arg == 'use_tpu'):
                    add_use_tpu = False
                if add_export_to_tpu and (keyword.arg == 'export_to_tpu'):
                    add_export_to_tpu = False
        if add_eval_on_tpu:
            log_success_report(getattr(node, 'lineno', 'None'), 'TPUEstimator(eval_on_tpu=*)')
            node.keywords.append(ast.keyword(arg='eval_on_tpu', value=ast.NameConstant(value=False)))
            util_global.set_value('need_conver', True)
        if add_use_tpu:
            log_success_report(getattr(node, 'lineno', 'None'), 'TPUEstimator(use_tpu=*)')
            node.keywords.append(ast.keyword(arg='use_tpu', value=ast.NameConstant(value=False)))
            util_global.set_value('need_conver', True)
        if add_export_to_tpu:
            log_success_report(getattr(node, 'lineno', 'None'), 'TPUEstimator(export_to_tpu=*)')
            node.keywords.append(ast.keyword(arg='export_to_tpu', value=ast.NameConstant(value=False)))
            util_global.set_value('need_conver', True)
        return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'RunConfig') \
        and ((isinstance(node.func.value, ast.Name) and node.func.value.id == 'tpu') \
            or (isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'tpu')):
        for keyword in node.keywords:
            if keyword.arg == 'session_config':
                log_success_report(getattr(node, 'lineno', 'None'), 'add_npu_config')
                keyword.value = ast.Call(func=ast.Name(id='npu_init', ctx=ast.Load()), args=[keyword.value], keywords=[])
                util_global.set_value('insert_npu_init_func', True)
                util_global.set_value('need_conver', True)
                return node

        node.keywords.append(ast.keyword(
            arg='session_config',
            value = ast.Call(keywords=[],
                args=[
                    ast.Call(args=[], keywords=[], func=ast.Attribute(value=ast.Name(id='config_pb2', ctx=ast.Load()), attr='ConfigProto', ctx=ast.Load()))
                ],
                func=ast.Name(id='npu_init', ctx=ast.Load()))))
        log_success_report(getattr(node, 'lineno', 'None'), 'add_npu_config')
        util_global.set_value('insert_npu_init_func', True)
        util_global.set_value('import_config_pb2', True)
        util_global.set_value('need_conver', True)
        return node
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
                    compile_ops.value = ast.NameConstant(value=False)
                else:
                    node.keywords.append(ast.keyword(arg='compile_ops', value=ast.NameConstant(value=False)))
                return node
    for estimator in util_global.get_value('Estimators', []):
        if (isinstance(node.func, ast.Attribute) and (node.func.attr == estimator)) \
            or (isinstance(node.func, ast.Name) and (node.func.id == estimator)):
            config = None
            for keyword in node.keywords:
                if keyword.arg == 'config':
                    config = keyword
                    break
            if config:
                config.value = ast.Call(
                    func=ast.Name(id='npu_run_config_init', ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(arg='run_config', value=config.value)
                    ]
                )
            else:
                node.keywords.append(
                    ast.keyword(
                        arg='config',
                        value=ast.Call(func=ast.Name(id='npu_run_config_init', ctx=ast.Load()), args=[], keywords=[])
                    )
                )
            util_global.set_value('insert_npu_run_config_func', True)
            util_global.set_value('need_conver', True)
            return node
    for estimator_func in util_global.get_value('EstimatorFunc', []):
        if isinstance(node.func, ast.Attribute) and (node.func.attr == estimator_func):
            input_fn = None
            hooks = None
            for keyword in node.keywords:
                if keyword.arg == 'input_fn':
                    input_fn = keyword
                elif keyword.arg == 'hooks':
                    hooks = keyword
            if not input_fn:
                break
            if not hooks:
                node.keywords.append(
                    ast.keyword(arg='hooks', value=ast.Call(func=ast.Name(id='npu_hooks_append', ctx=ast.Load()), args=[], keywords=[])))
            else:
                hooks.value = ast.Call(func=ast.Name(id='npu_hooks_append', ctx=ast.Load()), args=[], keywords=[
                    ast.keyword(arg='hooks_list', value=hooks.value)])
            util_global.set_value('insert_npu_hooks_append', True)
            util_global.set_value('need_conver', True)
            return node
    if isinstance(node.func, ast.Attribute) and (node.func.attr == 'compile'):
        opt_map = {"adadelta": "tf.keras.optimizers.Adadelta",
                   "adagrad": "tf.keras.optimizers.Adagrad",
                   "adam": "tf.keras.optimizers.Adam",
                   "adamax": "tf.keras.optimizers.Adamax",
                   "ftrl": "tf.keras.optimizers.Ftrl",
                   "nadam": "tf.keras.optimizers.Nadam",
                   "rmsprop": "tf.keras.optimizers.RMSprop",
                   "sgd": "tf.keras.optimizers.SGD"}
        for keyword in node.keywords:
            if keyword.arg == "optimizer":
                log_success_report(getattr(node, 'lineno', 'None'), 'KerasDistributeOptimizer')
                opt_func_name = ast.Name(id="npu_keras_optimizer", ctx=ast.Load())
                if isinstance(keyword.value, ast.Str):
                    keras_opt = opt_map[keyword.value.s].split(".")
                    tf_opt_func = ast.Attribute(value=ast.Attribute(value=ast.Attribute(value=ast.Name(id=keras_opt[0], ctx=ast.Load()),
                                                attr=keras_opt[1], ctx=ast.Load()), attr=keras_opt[2], ctx=ast.Load()),
                                                attr=keras_opt[3], ctx=ast.Load())
                    keyword.value = ast.Call(func=opt_func_name, args=[ast.Call(func=tf_opt_func, args=[], keywords=[])], keywords=[])
                util_global.set_value('need_conver', True)
                util_global.set_value('insert_npu_keras_opt_func', True)
                return node
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
        if node.func.attr.find("Optimizer") != -1:
            log_success_report(getattr(node, "lineno", "None"), "NPUDistributedOptimizer")
            node = ast.Call(func=ast.Name(id="npu_tf_optimizer", ctx=ast.Load()), args=[node], keywords=[])
            util_global.set_value('need_conver', True)
            util_global.set_value('insert_npu_tf_opt_func', True)
            return node
    if isinstance(node.func, ast.Attribute):
        opt_list = ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop", "SGD"]
        if node.func.attr in opt_list:
            log_success_report(getattr(node, "lineno", "None"), "KerasDistributeOptimizer")
            node = ast.Call(func=ast.Name(id="npu_keras_optimizer", ctx=ast.Load()), args=[node], keywords=[])
            util_global.set_value('need_conver', True)
            util_global.set_value('insert_npu_keras_opt_func', True)
            return node
    return node

def insert_npu_import(r_node):
    npu_alias = ast.alias(name='*', asname=None)
    npu_import = ast.ImportFrom(module='npu_bridge.npu_init', names=[npu_alias], level=0)
    num = 5 if len(r_node.body) >= 5 else len(r_node.body)
    import_index = 0
    is_insert = False
    for i in range(0, num):
        if isinstance(r_node.body[i], ast.Import):
            r_node.body.insert(i, npu_import)
            log_success_report(i, "import")
            is_insert = True
            break
        elif isinstance(r_node.body[i], ast.ImportFrom):
            if r_node.body[i].module != "__future__":
                r_node.body.insert(i, npu_import)
                log_success_report(i, "import")
                is_insert = True
                break
            else:
                import_index = i + 1
    if not is_insert:
        r_node.body.insert(import_index, npu_import)
        log_success_report(import_index, "import")

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
                                value=ast.Name(id='config', ctx=ast.Load())))))))

        custom_op_name_assign_node = ast.Assign(
            targets=[
                ast.Attribute(value=ast.Name(id='custom_op', ctx=ast.Load()), attr='name', ctx=ast.Store())
            ],
            value=ast.Str(s='NpuOptimizer'))

        util_global.set_value('import_RewriterConfig', True)
        remapping_assign_node = ast.Assign(
            targets=[
                ast.Attribute(attr='remapping', ctx=ast.Store(),
                    value=ast.Attribute(attr='rewrite_options', ctx=ast.Load(),
                        value=ast.Attribute(attr='graph_options', ctx=ast.Load(),
                            value=ast.Name(id='config', ctx=ast.Load()))))
            ],
            value=ast.Attribute(attr='OFF', ctx=ast.Load(), value=ast.Name(id='RewriterConfig')))

        return_node = ast.Return(value=ast.Name(id='config', ctx=ast.Load()))

        r_node.body.insert(n, ast.FunctionDef(
            name='npu_init',
            args=ast.arguments(
                args=[
                    ast.arg(arg='config', annotation=None)
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]),
            body=[
                custom_op_assign_node,
                custom_op_name_assign_node,
                remapping_assign_node,
                return_node
            ],
            decorator_list=[],
            returns=None))
def insert_NPUBroadcastGlobalVariablesHook_import(r_node):
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
        log_success_report(n, 'import NPUBroadcastGlobalVariablesHook')
        r_node.body.insert(n, ast.ImportFrom(module='npu_bridge.estimator.npu.npu_hook', names=[ast.alias(name='NPUBroadcastGlobalVariablesHook', asname=None)], level=0))
def insert_npu_hooks_append_func(r_node):
    n = 0
    lenline = len(r_node.body)
    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1
    while n < lenline and (isinstance(r_node.body[n], ast.ImportFrom) or isinstance(r_node.body[n], ast.Import)):
        n += 1
    if_not_list_node = ast.If(
        test=ast.UnaryOp(
            op=ast.Not(),
            operand=ast.Call(
                func=ast.Name(id='isinstance', ctx=ast.Load()),
                args=[ast.Name(id='hooks_list', ctx=ast.Load()), ast.Name(id='list', ctx=ast.Load())],
                keywords=[]
            )
        ),
        body=[ast.Assign(targets=[ast.Name(id='hooks_list', ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))],
        orelse=[]
    )
    list_add_node = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name('hooks_list', ctx=ast.Load()),
                attr='append',
                ctx=ast.Load()
            ),
            args=[
                ast.Call(
                    func=ast.Name(id='NPUBroadcastGlobalVariablesHook', ctx=ast.Load()),
                    args=[
                        ast.Num(n=0),
                        ast.Call(
                            func=ast.Name(id='int', ctx=ast.Load()),
                            args=[
                                ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id='os', ctx=ast.Load()),
                                        attr='getenv',
                                        ctx=ast.Load()
                                    ),
                                    args=[
                                        ast.Str(s='RANK_ID'), ast.Str(s='0')
                                    ],
                                    keywords=[]
                                )
                            ],
                            keywords=[]
                        )
                    ],
                    keywords=[]
                )
            ],
            keywords=[]
        )
    )
    return_node = ast.Return(value=ast.Name(id='hooks_list', ctx=ast.Load()))
    util_global.set_value('import_NPUBroadcastGlobalVariablesHook', True)
    util_global.set_value('import_os', True)
    r_node.body.insert(n, ast.FunctionDef(
        name='npu_hooks_append',
        args=ast.arguments(
            args=[
                ast.arg(arg='hooks_list', annotation=None)
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[ast.List(elts=[], ctx=ast.Load())]
        ),
        body=[if_not_list_node, list_add_node, return_node],
        decorator_list=[],
        returns=None
    ))
def insert_npu_session_config_func(r_node):
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and (isinstance(r_node.body[n], ast.ImportFrom) or isinstance(r_node.body[n], ast.Import)):
        n += 1
    if_not_session_config_node = ast.If(
        test=ast.BoolOp(
            op=ast.And(),
            values=[
                ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Call(
                        func=ast.Name(id='isinstance', ctx=ast.Load()),
                        args=[
                            ast.Name(id='session_config'),
                            ast.Attribute(value=ast.Name(id='config_pb2', ctx=ast.Load()), attr='ConfigProto', ctx=ast.Load())
                        ],
                        keywords=[]
                    )
                ),
                ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Call(
                        func=ast.Name(id='issubclass', ctx=ast.Load()),
                        args=[
                            ast.Call(
                                func=ast.Name(id='type', ctx=ast.Load()),
                                args=[ast.Name(id='session_config', ctx=ast.Load())],
                                keywords=[]
                            ),
                            ast.Attribute(value=ast.Name(id='config_pb2', ctx=ast.Load()), attr='ConfigProto', ctx=ast.Load())
                        ],
                        keywords=[]
                    )
                )
            ]
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id='session_config', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='config_pb2', ctx=ast.Load()),
                        attr='ConfigProto',
                        ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[]
                )
            )
        ],
        orelse=[]
    )
    if_session_config_node = ast.If(
        test=ast.BoolOp(
            op=ast.Or(),
            values=[
                ast.Call(
                    func=ast.Name(id='isinstance', ctx=ast.Load()),
                    args=[
                        ast.Name(id='session_config', ctx=ast.Load()),
                        ast.Attribute(
                            value=ast.Name(id='config_pb2', ctx=ast.Load()),
                            attr='ConfigProto',
                            ctx=ast.Load()
                        )
                    ],
                    keywords=[]
                ),
                ast.Call(
                    func=ast.Name(id='issubclass', ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Name(id='type', ctx=ast.Load()),
                            args=[ast.Name(id='session_config', ctx=ast.Load())],
                            keywords=[]
                        ),
                        ast.Attribute(
                            value=ast.Name(id='config_pb2', ctx=ast.Load()),
                            attr='ConfigProto',
                            ctx=ast.Load()
                        )
                    ],
                    keywords=[]
                )
            ]
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id='custom_op', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id='session_config', ctx=ast.Load()),
                                    attr='graph_options',
                                    ctx=ast.Load()
                                ),
                                attr='rewrite_options',
                                ctx=ast.Load()
                            ),
                            attr='custom_optimizers',
                            ctx=ast.Load()
                        ),
                        attr='add',
                        ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[]
                )
            ),
            ast.Assign(
                targets=[ast.Attribute(value=ast.Name(id='custom_op', ctx=ast.Load()), attr='name', ctx=ast.Store())],
                value=ast.Str(s='NpuOptimizer')
            ),
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(
                                    id='session_config',
                                    ctx=ast.Load()
                                ),
                                attr='graph_options',
                                ctx=ast.Load()
                            ),
                            attr='rewrite_options',
                            ctx=ast.Load()
                        ),
                        attr='remapping',
                        ctx=ast.Store()
                    )
                ],
                value=ast.Attribute(
                    value=ast.Name(id='RewriterConfig', ctx=ast.Load()),
                    attr='OFF',
                    ctx=ast.Load()
                )
            )
        ],
        orelse=[]
    )
    return_node = ast.Return(value=ast.Name(id='session_config', ctx=ast.Load()))
    util_global.set_value('import_RewriterConfig', True)
    util_global.set_value('import_config_pb2', True)
    r_node.body.insert(n, ast.FunctionDef(
        name='npu_session_config_init',
        args=ast.arguments(
            args=[
                ast.arg(arg='session_config', annotation=None)
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[ast.NameConstant(value=None)]
        ),
        body=[if_not_session_config_node, if_session_config_node, return_node],
        decorator_list=[],
        returns=None
    ))

def insert_npu_run_config_func(r_node):
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and (isinstance(r_node.body[n], ast.ImportFrom) or isinstance(r_node.body[n], ast.Import)):
        n += 1
    if_not_run_config_node = ast.If(
        test=ast.BoolOp(
            op=ast.And(),
            values=[
                ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Call(
                        func=ast.Name(id='isinstance', ctx=ast.Load()),
                        args=[
                            ast.Name(id='run_config', ctx=ast.Load()),
                            ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id='tf', ctx=ast.Load()),
                                    attr='estimator',
                                    ctx=ast.Load()
                                ),
                                attr='RunConfig',
                                ctx=ast.Load()
                            )
                        ],
                        keywords=[]
                    )
                ),
                ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Call(
                        func=ast.Name(id='issubclass', ctx=ast.Load()),
                        args=[
                            ast.Call(
                                func=ast.Name(id='type', ctx=ast.Load()),
                                args=[ast.Name(id='run_config', ctx=ast.Load())],
                                keywords=[]
                            ),
                            ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id='tf', ctx=ast.Load()),
                                    attr='estimator',
                                    ctx=ast.Load()
                                ),
                                attr='RunConfig',
                                ctx=ast.Load()
                            )
                        ],
                        keywords=[]
                    )
                )
            ]
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id='run_config', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id='tf', ctx=ast.Load()),
                            attr='estimator',
                            ctx=ast.Load()
                        ),
                        attr='RunConfig',
                        ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[]
                )
            )
        ],
        orelse=[]
    )
    if_run_config_node = ast.If(
        test=ast.BoolOp(
            op=ast.Or(),
            values=[
                ast.Call(
                    func=ast.Name(id='isinstance', ctx=ast.Load()),
                    args=[
                        ast.Name(id='run_config', ctx=ast.Load()),
                        ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id='tf', ctx=ast.Load()),
                                attr='estimator',
                                ctx=ast.Load()
                            ),
                            attr='RunConfig',
                            ctx=ast.Load()
                        )
                    ],
                    keywords=[]
                ),
                ast.Call(
                    func=ast.Name(id='issubclass', ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Name(id='type', ctx=ast.Load()),
                            args=[
                                ast.Name(id='run_config', ctx=ast.Load())
                            ],
                            keywords=[]
                        ),
                        ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id='tf', ctx=ast.Load()),
                                attr='estimator',
                                ctx=ast.Load()
                            ),
                            attr='RunConfig',
                            ctx=ast.Load()
                        )
                    ],
                    keywords=[]
                )
            ]
        ),
        body=[
            ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Attribute(
                            value=ast.Name(id='run_config', ctx=ast.Load()),
                            attr='__dict__',
                            ctx=ast.Load()
                        ),
                        slice=ast.Index(
                            value=ast.Str(s='_session_config')
                        ),
                        ctx=ast.Store()
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id='npu_session_config_init', ctx=ast.Load()),
                    args=[
                        ast.Attribute(
                            value=ast.Name(id='run_config', ctx=ast.Load()),
                            attr='session_config',
                            ctx=ast.Load()
                        )
                    ],
                    keywords=[]
                )
            )
        ],
        orelse=[]
    )
    return_node = ast.Return(value=ast.Name(id='run_config', ctx=ast.Load()))
    util_global.set_value('insert_npu_session_config_func', True)
    r_node.body.insert(n, ast.FunctionDef(
        name='npu_run_config_init',
        args=ast.arguments(
            args=[
                ast.arg(arg='run_config', annotation=None)
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[ast.NameConstant(value=None)]
        ),
        body=[if_not_run_config_node, if_run_config_node, return_node],
        decorator_list=[],
        returns=None
    ))
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
        r_node.body.insert(n, ast.ImportFrom(module='tensorflow.core.protobuf', names=[ast.alias(name='config_pb2', asname=None)], level=0))

def insert_RewriterConfig_import(r_node):
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
        log_success_report(n, 'import RewriterConfig')
        r_node.body.insert(n, ast.ImportFrom(module='tensorflow.core.protobuf.rewriter_config_pb2', names=[ast.alias(name='RewriterConfig', asname=None)], level=0))

def insert_npu_tf_opt_func(r_node):
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and (isinstance(r_node.body[n], ast.ImportFrom) or isinstance(r_node.body[n], ast.Import)):
        n += 1

    if n < lenline:
        npu_func = ast.Name(id="NPUDistributedOptimizer", ctx=ast.Load())
        assign_target = ast.Name(id="npu_opt", ctx=ast.Store())
        assign_args = ast.Name(id="opt", ctx=ast.Load())
        npu_opt = ast.Assign(targets=[assign_target], value=ast.Call(func=npu_func, args=[assign_args], keywords=[]))
        return_node = ast.Return(value=ast.Name(id='npu_opt', ctx=ast.Load()))

        r_node.body.insert(n, ast.FunctionDef(
            name='npu_tf_optimizer',
            args=ast.arguments(
                args=[
                    ast.arg(arg='opt', annotation=None)
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]),
            body=[
                npu_opt,
                return_node
            ],
            decorator_list=[],
            returns=None))

def insert_npu_keras_opt_func(r_node):
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and (isinstance(r_node.body[n], ast.ImportFrom) or isinstance(r_node.body[n], ast.Import)):
        n += 1

    if n < lenline:
        npu_func = ast.Name(id="KerasDistributeOptimizer", ctx=ast.Load())
        assign_target = ast.Name(id="npu_opt", ctx=ast.Store())
        assign_args = ast.Name(id="opt", ctx=ast.Load())
        npu_opt = ast.Assign(targets=[assign_target], value=ast.Call(func=npu_func, args=[assign_args], keywords=[]))
        return_node = ast.Return(value=ast.Name(id='npu_opt', ctx=ast.Load()))

        r_node.body.insert(n, ast.FunctionDef(
            name='npu_keras_optimizer',
            args=ast.arguments(
                args=[
                    ast.arg(arg='opt', annotation=None)
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]),
            body=[
                npu_opt,
                return_node
            ],
            decorator_list=[],
            returns=None))

def insert_empty_hook(r_node):
    n = 0
    lenline = len(r_node.body)

    while n < lenline and not isinstance(r_node.body[n], ast.ImportFrom) and not isinstance(r_node.body[n], ast.Import):
        n += 1

    while n < lenline and (isinstance(r_node.body[n], ast.ImportFrom) or isinstance(r_node.body[n], ast.Import)):
        n += 1

    if n < lenline:
        hook_attr = ast.Attribute(value=ast.Attribute(value=ast.Name(id="tf", ctx=ast.Load()), attr="train", ctx=ast.Load()),
                                  attr="SessionRunHook", ctx=ast.Load())
        class_def = ast.ClassDef(name="NpuEmptyHook", bases=[hook_attr], keywords=[],
                                 body=[ast.Pass()], decorator_list=[])
        r_node.body.insert(n, class_def)

def insert_os_import(r_node):
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
        log_success_report(n, 'import os')
        r_node.body.insert(n, ast.Import(names=[ast.alias(name='os', asname=None)]))

def ast_assign(node):
    for target in node.targets:
        if (isinstance(target, ast.Name) and target.id == 'global_jit_level') or (isinstance(target, ast.Attribute) and target.attr == 'global_jit_level'):
            log_success_report(getattr(node, 'lineno', 'None'), '*.global_jit_level')
            util_global.set_value('import_config_pb2', True)
            util_global.set_value('need_conver', True)
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
                    inputs = None
                    pool_size = None
                    strides = None
                    padding = None
                    data_format = None
                    name = None
                    for index, arg in enumerate(node.value.args):
                        if index == 0: inputs = arg
                        elif index == 1: pool_size = arg
                        elif index == 2: strides = arg
                        elif index == 3: padding = arg
                        elif index == 4: data_format = arg
                        elif index == 5: name = arg
                    for keyword in node.value.keywords:
                        if keyword.arg == 'inputs': inputs = keyword.value
                        elif keyword.arg == 'pool_size': pool_size = keyword.value
                        elif keyword.arg == 'strides': strides = keyword.value
                        elif keyword.arg == 'padding': padding = keyword.value
                        elif keyword.arg == 'data_format': data_format = keyword.value
                        elif keyword.arg == 'name': name = keyword.value
                    node.value.func = ast.Attribute(value=ast.Attribute(
                        value=ast.Name(id='tf', ctx=ast.Load()), attr='nn', ctx=ast.Load()), attr='max_pool_with_argmax', ctx=ast.Load())
                    node.value.args=[]
                    node.value.keywords=[]
                    if inputs:
                        node.value.keywords.append(ast.keyword(arg='input', value=inputs))
                    if pool_size:
                        if isinstance(pool_size, ast.Num):
                            node.value.keywords.append(ast.keyword(
                                arg='ksize', value=ast.List(elts=[ast.Num(n=1), pool_size, pool_size, ast.Num(n=1)], ctx=ast.Load())))
                        elif (isinstance(pool_size, ast.List) or isinstance(pool_size, ast.Tuple)) and len(pool_size.elts) >= 2:
                            node.value.keywords.append(ast.keyword(
                                arg='ksize', value=ast.List(elts=[ast.Num(n=1), pool_size.elts[0], pool_size.elts[1], ast.Num(n=1)], ctx=ast.Load())))
                    if strides:
                        if isinstance(pool_size, ast.Num):
                            node.value.keywords.append(ast.keyword(
                                arg='strides', value=ast.List(elts=[ast.Num(n=1), strides, strides, ast.Num(n=1)], ctx=ast.Load())))
                        elif (isinstance(strides, ast.List) or isinstance(strides, ast.Tuple)) and len(strides.elts) >= 2:
                            node.value.keywords.append(ast.keyword(
                                arg='strides', value=ast.List(elts=[ast.Num(n=1), strides.elts[0], strides.elts[1], ast.Num(n=1)], ctx=ast.Load())))
                    if padding:
                        padding.s = padding.s.upper()
                        node.value.keywords.append(ast.keyword(arg='padding', value=padding))
                    else:
                        node.value.keywords.append(ast.keyword(arg='padding', value=ast.Str(s='SAME')))
                    if data_format:
                        node.value.keywords.append(ast.keyword(arg='data_format', value=data_format))
                    if name:
                        node.value.keywords.append(ast.keyword(arg='name', value=name))
                    node.targets.append(ast.Name(id='argmax', ctx=ast.Store()))
                    node.targets = [ast.Tuple(elts=node.targets, ctx=ast.Store())]
                    return node
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