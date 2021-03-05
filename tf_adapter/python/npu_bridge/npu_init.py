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

def npu_config_proto(config_proto = None):
    if (not isinstance(config_proto, config_pb2.ConfigProto)) or (not issubclass(type(config_proto), config_pb2.ConfigProto)):
        config_proto = config_pb2.ConfigProto()
    config_proto.allow_soft_placement = True
    config_proto.log_device_placement = False
    config_proto.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF
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