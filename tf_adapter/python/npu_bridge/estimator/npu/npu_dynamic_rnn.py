# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import tensorflow as tf
from tensorflow.python.framework import dtypes
from npu_bridge.helper import helper
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops

gen_npu_ops = helper.get_gen_ops()

DYNAMIC_RNN_UNIDIRECTION = "UNIDIRECTIONAL"
DYNAMIC_RNN_BIDIRECTION = "BIDIRECTIONAL"


_npu_gru_doc_string = """
  Cudnn RNN has an opaque parameter buffer that can be used for inference and
  training. But it is possible that the layout of the parameter buffers
  changes between generations. So it is highly recommended to use
  CudnnOpaqueParamsSaveable to save and restore weights and biases in a
  canonical format.
  This is a typical use case:
    * The user creates a CudnnRNN model.
    * The user query that parameter buffer size.
    * The user creates a variable of that size that serves as the parameter
        buffers.
    * The user either initialize the parameter buffer, or load the canonical
        weights into the parameter buffer.
    * The user calls the model with the parameter buffer for inference, or
        training.
    * If training, the user creates a Saver object.
    * If training, the user creates a CudnnOpaqueParamsSaveable object from the
        parameter buffer for it to be later saved in the canonical format. When
        creating a CudnnOpaqueParamsSaveable object, a name could be provided,
        which is useful in distinguishing the names of multiple
        CudnnOpaqueParamsSaveable objects (e.g. for an encoder-decoder model).
    * Once a while, the user saves the parameter buffer into model checkpoints
        with Saver.save().
    * When restoring, the user creates a CudnnOpaqueParamsSaveable object and
      uses Saver.restore() to restore the parameter buffer from the canonical
      format to a user-defined format, as well as to restore other savable
      objects in the checkpoint file.
"""

class _DynamicBasic(base_layer.Layer):
  """Create a basic class for dynamic using Layer."""

  def __init__(self,
               hidden_size,
               dtype,
               direction=DYNAMIC_RNN_UNIDIRECTION,
               cell_depth=1,
               keep_prob=1.0,
               cell_clip=-1.0,
               num_proj=0,
               time_major=True,
               activation="tanh",
               is_training=True):
    super(_DynamicBasic, self).__init__()
    self._direction = direction
    self._cell_depth = cell_depth
    self._keep_prob = keep_prob
    self._cell_clip = cell_clip
    self._num_proj = num_proj
    self._time_major = time_major
    self._activation = activation
    self._is_training = is_training
    self._hidden_size = hidden_size
    self._dtype = dtype
    self._args = {
      "direction": self._direction,
      "cell_depth": self._cell_depth,
      "keep_prob": self._keep_prob,
      "cell_clip": self._cell_clip,
      "num_proj": self._num_proj,
      "time_major": self._time_major,
      "activation": self._activation,
      "is_training": self._is_training
    }
    self._seq_length = None
    self._init_h = None


  @property
  def direction(self):
    return self._direction

  @property
  def cell_depth(self):
    return self._cell_depth

  @property
  def keep_prob(self):
    return self._keep_prob

  @property
  def cell_clip(self):
    return self._cell_clip

  @property
  def num_proj(self):
    return self._num_proj

  @property
  def time_major(self):
    return self._time_major

  @property
  def activation(self):
    return self._activation

  @property
  def is_training(self):
    return self._is_training

  def check_direction(self):
    """Check validity of direction."""
    if self._direction not in (DYNAMIC_RNN_UNIDIRECTION, DYNAMIC_RNN_BIDIRECTION):
        raise ValueError("Invalid direction: %s, expecting %s or %s" %
                        (self._direction, DYNAMIC_RNN_UNIDIRECTION, DYNAMIC_RNN_BIDIRECTION))

  def build(self, input_shape):
    if input_shape[1].value is None:
      raise ValueError("Expected input_shape[1] to be known, saw shape: batch_size.")
    batch_size = input_shape[1].value
    self._seq_length = self.add_variable(
      "dynamicbase/seq_length",
      shape=[batch_size],
      dtype=dtypes.int32,
      initializer=init_ops.zeros_initializer(dtype=dtypes.int32),
      trainable=False)
    self._init_h = array_ops.zeros([batch_size, self._hidden_size], dtype=self._dtype)
    super(_DynamicBasic, self).build(input_shape)

  def call(self,
           x,
           seq_length=None,
           init_h=None):
    """Dynamic GRU.
    Args:
        inputs: the input sequence to the RNN model. If `time_major` is True
        (default), the Tensor shape is [max_time, batch_size, input_size]. If
        `time_major` is False, the shape is [batch_size, max_time, input_size].
        input_h: the initial hidden state for h. If `time_major` is True (default),
        the Tensor shape is [num_layers, batch_size, num_units]. If `time_major`
        is False, the shape is [batch_size, num_layers, num_units].
        input_c: the initial hidden state for c. This is only relevant for LSTM. A
        Tensor of the same shape as input_h.
        params: the parameter buffer created for this model.
        is_training: whether this operation will be used in training or inference
        rnn_mode: one of ('lstm', 'gru', 'rnn_relu', 'rnn_tanh').
        sequence_lengths: an int32 array representing the variable sequence lengths
        in a batch. The size of the array has to equal the batch_size. Default to
        None, in which case sequences in the batch are assumed to have the same
        length, which is inferred from inputs.
        time_major: The shape format of the `inputs` and `outputs` Tensors. If true,
        these Tensors must be shaped ['max_time', 'batch_size', 'depth']. If
        false, these Tensors must be shaped ['batch_size', 'max_time', 'depth'].
        By default this function accepts input and emits output in time-major
        form. This param is only effective when 'sequence_lengths' is used.
        input_mode: indicate whether there is a linear projection between the input
        and the actual computation before the first layer. It could be
        'linear_input', 'skip_input' or 'auto_select'. 'linear_input' (default)
        always applies a linear projection of input onto RNN hidden state.
        (standard RNN behavior). 'skip_input' is only allowed when input_size ==
        num_units; 'auto_select' implies 'skip_input' when input_size ==
        num_units; otherwise, it implies 'linear_input'.
        direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
        dropout: whether to enable dropout. With it is 0, dropout is disabled.
        seed: the op seed used for initializing dropout. See
        `tf.compat.v1.set_random_seed` for behavior.
        num_proj: The output dimensionality for the projection matrices.
        If None or 0, no projection is performed.
        name: name of the operation.
    Returns:
        if is_training is true
          y, output_h, update, reset, new, hidden_new
        else 
          y, output_h
    """
    self.check_direction()
    self._args["x"] = x
    if seq_length is None:
      seq_length = self._seq_length
    if init_h is None:
      init_h = self._init_h
    self._args["seq_length"] = seq_length
    self._args["init_h"] = init_h

class DynamicGRUV2(_DynamicBasic):
  """Create a basic class for dynamic using Layer."""

  def __init__(self,
               hidden_size,
               dtype,
               direction=DYNAMIC_RNN_UNIDIRECTION,
               cell_depth=1,
               keep_prob=1.0,
               cell_clip=-1.0,
               num_proj=0,
               time_major=True,
               activation="tanh",
               gate_order="zrh",
               reset_after=True,
               is_training=True):
    super(DynamicGRUV2, self).__init__(
      hidden_size,
      dtype,
      direction=direction,
      cell_depth=cell_depth,
      keep_prob=keep_prob,
      cell_clip=cell_clip,
      num_proj=num_proj,
      activation=activation,
      time_major=time_major,
      is_training=is_training)
    self._gate_order = gate_order
    self._reset_after = reset_after
    self._args["bias_type"] = "single_bias"
    self._args["gate_order"] = self._gate_order
    self._args["reset_after"] = self._reset_after
    self._gruv2_weight_input = None
    self._gruv2_weight_hidden = None
    self._bias_input = None
    self._bias_hidden = None


  @property
  def gate_order(self):
    return self._gate_order

  @property
  def reset_after(self):
    return self._reset_after

  def build(self, input_shape):
    if input_shape[2].value is None:
      raise ValueError("Expected input_shape[2] to be known, saw shape: input_size.")
    input_size = input_shape[2].value
    stdv = 1.0 / math.sqrt(self._hidden_size)
    self._gruv2_weight_input = self.add_variable(
      "dynamicgruv2/weight_input",
      shape=[input_size, 3 * self._hidden_size],
      dtype=self._dtype,
      initializer=init_ops.random_uniform_initializer(-stdv, stdv))
    self._gruv2_weight_hidden = self.add_variable(
      "dynamicgruv2/weight_hidden",
      shape=[self._hidden_size, 3 * self._hidden_size],
      dtype=self._dtype,
      initializer=init_ops.random_uniform_initializer(-stdv, stdv))
    self._bias_input = self.add_variable(
      "dynamicgruv2/bias_input",
      shape=[3 * self._hidden_size],
      dtype=self._dtype,
      initializer=init_ops.random_uniform_initializer(-stdv, stdv))
    self._bias_hidden = self.add_variable(
      "dynamicgruv2/bias_hidden",
      shape=[3 * self._hidden_size],
      dtype=self._dtype,
      initializer=init_ops.random_uniform_initializer(-stdv, stdv))
    super(DynamicGRUV2, self).build(input_shape)

  def call(self,
           x,
           weight_input=None,
           weight_hidden=None,
           bias_input=None,
           bias_hidden=None,
           seq_length=None,
           init_h=None):
    """Dynamic GRU.
    Args:
        inputs: the input sequence to the RNN model. If `time_major` is True
        (default), the Tensor shape is [max_time, batch_size, input_size]. If
        `time_major` is False, the shape is [batch_size, max_time, input_size].
        input_h: the initial hidden state for h. If `time_major` is True (default),
        the Tensor shape is [num_layers, batch_size, num_units]. If `time_major`
        is False, the shape is [batch_size, num_layers, num_units].
        input_c: the initial hidden state for c. This is only relevant for LSTM. A
        Tensor of the same shape as input_h.
        params: the parameter buffer created for this model.
        is_training: whether this operation will be used in training or inference
        rnn_mode: one of ('lstm', 'gru', 'rnn_relu', 'rnn_tanh').
        sequence_lengths: an int32 array representing the variable sequence lengths
        in a batch. The size of the array has to equal the batch_size. Default to
        None, in which case sequences in the batch are assumed to have the same
        length, which is inferred from inputs.
        time_major: The shape format of the `inputs` and `outputs` Tensors. If true,
        these Tensors must be shaped ['max_time', 'batch_size', 'depth']. If
        false, these Tensors must be shaped ['batch_size', 'max_time', 'depth'].
        By default this function accepts input and emits output in time-major
        form. This param is only effective when 'sequence_lengths' is used.
        input_mode: indicate whether there is a linear projection between the input
        and the actual computation before the first layer. It could be
        'linear_input', 'skip_input' or 'auto_select'. 'linear_input' (default)
        always applies a linear projection of input onto RNN hidden state.
        (standard RNN behavior). 'skip_input' is only allowed when input_size ==
        num_units; 'auto_select' implies 'skip_input' when input_size ==
        num_units; otherwise, it implies 'linear_input'.
        direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
        dropout: whether to enable dropout. With it is 0, dropout is disabled.
        seed: the op seed used for initializing dropout. See
        `tf.compat.v1.set_random_seed` for behavior.
        num_proj: The output dimensionality for the projection matrices.
        If None or 0, no projection is performed.
        name: name of the operation.
    Returns:
        if is_training is true
          y, output_h, update, reset, new, hidden_new
        else 
          y, output_h
    """
    super(DynamicGRUV2, self).call(x,
                                   seq_length=seq_length,
                                   init_h=init_h)
    if weight_input is None:
      weight_input = self._gruv2_weight_input
    if weight_hidden is None:
      weight_hidden = self._gruv2_weight_hiddens
    if bias_input is None:
      bias_input = self._bias_input
    if bias_hidden is None:
      bias_hidden = self._bias_hidden
    self._args["weight_input"] = weight_input
    self._args["weight_hidden"] = weight_hidden
    self._args["bias_input"] = bias_input
    self._args["bias_hidden"] = bias_hidden
    return gen_npu_ops.dynamic_gru_v2(**self._args)

class DynamicRNN(_DynamicBasic):
  """Create a basic class for dynamic using Layer."""

  def __init__(self,
               hidden_size,
               dtype,
               cell_type="LSTM",
               direction=DYNAMIC_RNN_UNIDIRECTION,
               cell_depth=1,
               use_peephole=False,
               keep_prob=1.0,
               cell_clip=-1.0,
               num_proj=0,
               time_major=True,
               activation="tanh",
               forget_bias=0.0,
               is_training=True):
    super(DynamicRNN, self).__init__(
      hidden_size,
      dtype,
      direction=direction,
      cell_depth=cell_depth,
      keep_prob=keep_prob,
      cell_clip=cell_clip,
      num_proj=num_proj,
      activation=activation,
      time_major=time_major,
      is_training=is_training)
    self._cell_type = cell_type
    self._use_peephole = use_peephole
    self._forget_bias = forget_bias
    self._args["cell_type"] = self._cell_type
    self._args["use_peephole"] = self._use_peephole
    self._args["forget_bias"] = self._forget_bias
    self._rnn_w = None
    self._rnn_b = None
    self._init_c = None


  @property
  def cell_type(self):
    return self._cell_type

  @property
  def use_peephole(self):
    return self._use_peephole

  @property
  def forget_bias(self):
    return self._forget_bias

  def build(self, input_shape):
    if input_shape[1].value is None:
      raise ValueError("Expected input_shape[0] to be known, saw shape: batch_size.")
    batch_size = input_shape[1].value
    if input_shape[2].value is None:
      raise ValueError("Expected input_shape[2] to be known, saw shape: input_size.")
    input_size = input_shape[2].value
    self._rnn_w = self.add_variable(
      "dynamicrnn/w",
      shape=[input_size + self._hidden_size, 4 * self._hidden_size],
      dtype=self._dtype,
      initializer=init_ops.glorot_uniform_initializer(seed=10, dtype=self._dtype))
    self._rnn_b = self.add_variable(
      "dynamicrnn/b",
      shape=[4 * self._hidden_size],
      dtype=self._dtype,
      initializer=init_ops.zeros_initializer(dtype=self._dtype))
    self._init_c = array_ops.zeros([batch_size, self._hidden_size], dtype=self._dtype)
    super(DynamicRNN, self).build(input_shape)

  def call(self,
           x,
           w=None,
           b=None,
           seq_length=None,
           init_h=None,
           init_c=None):
    """Dynamic GRU.
    Args:
        inputs: the input sequence to the RNN model. If `time_major` is True
        (default), the Tensor shape is [max_time, batch_size, input_size]. If
        `time_major` is False, the shape is [batch_size, max_time, input_size].
        input_h: the initial hidden state for h. If `time_major` is True (default),
        the Tensor shape is [num_layers, batch_size, num_units]. If `time_major`
        is False, the shape is [batch_size, num_layers, num_units].
        input_c: the initial hidden state for c. This is only relevant for LSTM. A
        Tensor of the same shape as input_h.
        params: the parameter buffer created for this model.
        is_training: whether this operation will be used in training or inference
        rnn_mode: one of ('lstm', 'gru', 'rnn_relu', 'rnn_tanh').
        sequence_lengths: an int32 array representing the variable sequence lengths
        in a batch. The size of the array has to equal the batch_size. Default to
        None, in which case sequences in the batch are assumed to have the same
        length, which is inferred from inputs.
        time_major: The shape format of the `inputs` and `outputs` Tensors. If true,
        these Tensors must be shaped ['max_time', 'batch_size', 'depth']. If
        false, these Tensors must be shaped ['batch_size', 'max_time', 'depth'].
        By default this function accepts input and emits output in time-major
        form. This param is only effective when 'sequence_lengths' is used.
        input_mode: indicate whether there is a linear projection between the input
        and the actual computation before the first layer. It could be
        'linear_input', 'skip_input' or 'auto_select'. 'linear_input' (default)
        always applies a linear projection of input onto RNN hidden state.
        (standard RNN behavior). 'skip_input' is only allowed when input_size ==
        num_units; 'auto_select' implies 'skip_input' when input_size ==
        num_units; otherwise, it implies 'linear_input'.
        direction: the direction model that the model operates. Could be either
        'unidirectional' or 'bidirectional'
        dropout: whether to enable dropout. With it is 0, dropout is disabled.
        seed: the op seed used for initializing dropout. See
        `tf.compat.v1.set_random_seed` for behavior.
        num_proj: The output dimensionality for the projection matrices.
        If None or 0, no projection is performed.
        name: name of the operation.
    Returns:
        if is_training is true
          y, output_h, update, reset, new, hidden_new
        else 
          y, output_h
    """
    super(DynamicRNN, self).call(x,
                                 seq_length=seq_length,
                                 init_h=init_h)
    if w is None:
      w = self._rnn_w
    if b is None:
      b = self._rnn_b
    if init_c is None:
      init_c = self._init_c
    self._args["w"] = w
    self._args["b"] = b
    self._args["init_c"] = init_c
    return gen_npu_ops.dynamic_rnn(**self._args)
