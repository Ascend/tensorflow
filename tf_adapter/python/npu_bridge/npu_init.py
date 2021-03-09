from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_config import ProfilingConfig
from npu_bridge.estimator.npu.npu_config import DumpConfig
from npu_bridge.estimator.npu.npu_config import DynamicInputConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_estimator import NPUEstimatorSpec
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu.npu_hook import NPUCheckpointSaverHook
from npu_bridge.estimator.npu.npu_hook import NPUOutputTensorHook
from npu_bridge.estimator.npu.npu_hook import NPUBroadcastGlobalVariablesHook
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu.npu_optimizer import NPUOptimizer
from npu_bridge.estimator.npu.npu_optimizer import KerasDistributeOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu import npu_rnn
from npu_bridge.estimator.npu import npu_scope
from npu_bridge.estimator.npu import util
from npu_bridge.estimator.npu import keras_to_npu
from npu_bridge.estimator.npu import npu_strategy
from npu_bridge.estimator.npu_unary_ops import npu_unary_ops
from npu_bridge.hccl import hccl_ops

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.client import session

from hccl.manage.api import create_group
from hccl.manage.api import destroy_group
from hccl.manage.api import get_rank_size
from hccl.manage.api import get_local_rank_size
from hccl.manage.api import get_rank_id
from hccl.manage.api import get_local_rank_id
from hccl.manage.api import get_world_rank_from_group_rank
from hccl.manage.api import get_group_rank_from_world_rank
from hccl.split.api import set_split_strategy_by_idx
from hccl.split.api import set_split_strategy_by_size

import tensorflow as tf
import os

def _npu_config_proto(config_proto):
    config_proto.allow_soft_placement = True
    config_proto.log_device_placement = False
    config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config_proto.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF

def npu_session_config_init(session_config=None):
    if ((not isinstance(session_config, config_pb2.ConfigProto)) and (not issubclass(type(session_config), config_pb2.ConfigProto))):
        session_config = config_pb2.ConfigProto()
    if (isinstance(session_config, config_pb2.ConfigProto) or issubclass(type(session_config), config_pb2.ConfigProto)):
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        _npu_config_proto(session_config)
    return session_config

def npu_run_config_init(run_config=None):
    if ((not isinstance(run_config, tf.estimator.RunConfig)) and (not issubclass(type(run_config), tf.estimator.RunConfig))):
        run_config = tf.estimator.RunConfig()
    if (isinstance(run_config, tf.estimator.RunConfig) or issubclass(type(run_config), tf.estimator.RunConfig)):
        run_config.__dict__['_session_config'] = npu_session_config_init(run_config.session_config)
    return run_config

def npu_hooks_append(hooks_list=[]):
    if (not isinstance(hooks_list, list)):
        hooks_list = []
    hooks_list.append(NPUBroadcastGlobalVariablesHook(0, int(os.getenv('RANK_ID', '0'))))
    return hooks_list

def npu_config_proto(config_proto = None):
    if (not isinstance(config_proto, config_pb2.ConfigProto)) or (not issubclass(type(config_proto), config_pb2.ConfigProto)):
        config_proto = config_pb2.ConfigProto()
    _npu_config_proto(config_proto)
    return config_proto

def npu_graph_options(graph_options = None):
    if (not isinstance(graph_options, config_pb2.GraphOptions)) or (not issubclass(type(graph_options), config_pb2.GraphOptions)):
        graph_options = config_pb2.GraphOptions()
    graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
    return graph_options

def npu_optimizer_options(optimizer_options = None):
    if (not isinstance(optimizer_options, config_pb2.OptimizerOptions)) or (not issubclass(type(optimizer_options), config_pb2.OptimizerOptions)):
        optimizer_options = config_pb2.OptimizerOptions()
    optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
    return optimizer_options

def set_keras_session_npu_config():
    from tensorflow.python.keras import backend
    config = config_pb2.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess = session.Session(config=config)
    backend.set_session(sess)
    return sess

def init_resource():
    npu_init = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()
    config = config_pb2.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess = session.Session(config=config)
    sess.run(npu_init)
    return sess, npu_shutdown

def shutdown_resource(sess, npu_shutdown):
    sess.run(npu_shutdown)

def close_session(sess):
    sess.close()
