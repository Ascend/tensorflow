import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import one_device_strategy

from hccl.manage.api import get_rank_size
from hccl.manage.api import get_rank_id

class NPUExtended(one_device_strategy.OneDeviceExtended):
  def __init__(self, container_strategy, device):
    super(NPUExtended, self).__init__(container_strategy, device)

  def _experimental_distribute_dataset(self, dataset):
    return dataset.shard(get_rank_size(), get_rank_id())

class NPUStrategy(distribute_lib.StrategyV1):
  def __init__(self, device="/cpu:0"):
    super(NPUStrategy, self).__init__(NPUExtended(self, device))