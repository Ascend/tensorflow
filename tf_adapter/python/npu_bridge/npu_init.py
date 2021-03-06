from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_config import ProfilingConfig
from npu_bridge.estimator.npu.npu_config import DumpConfig
from npu_bridge.estimator.npu.npu_config import DynamicInputConfig

from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_estimator import NPUEstimatorSpec
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

from npu_bridge.estimator.npu.npu_hook import NPUCheckpointSaverHook
from npu_bridge.estimator.npu.npu_hook import NPUOutputTensorHook

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