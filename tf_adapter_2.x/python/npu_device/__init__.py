import os
import tensorflow as tf
npu_ops = tf.load_op_library(os.path.dirname(__file__) + "/_npu_ops.so")

from npu_device.npu_device import *
from npu_device._api import distribute