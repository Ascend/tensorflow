import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


class SessionConfigBuilder:
    def __init__(self, config=None):
        if isinstance(config, tf.compat.v1.ConfigProto):
            self._session_config = config
        else:
            self._session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self._session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        self._npu_config = self._session_config.graph_options.rewrite_options.custom_optimizers.add()
        self._npu_config.name = 'NpuOptimizer'

    @property
    def parameter_map(self):
        return self._npu_config.parameter_map

    @property
    def session_config(self):
        return self._session_config
