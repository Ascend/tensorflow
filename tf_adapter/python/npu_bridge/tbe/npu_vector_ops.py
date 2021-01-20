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
#
# Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
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
"""Ops for aicore cube."""
from tensorflow import Tensor
from tensorflow.python.eager import context
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from npu_bridge.helper import helper
from npu_bridge.estimator.npu_aicore_ops import prelu
gen_npu_ops = helper.get_gen_ops()

def lamb_apply_optimizer_assign(input0, input1, input2, input3, mul0_x, mul1_x,
                                mul2_x, mul3_x, add2_y, steps, do_use_weight, weight_decay_rate, name=None):
    if context.executing_eagerly():
      raise RuntimeError("tf.lamb_apply_optimizer_assign() is not compatible with "
                        "eager execution.")
    update, nextv, nextm = gen_npu_ops.lamb_apply_optimizer_assign(input0, input1, input2, input3, mul0_x, mul1_x, mul2_x,
                                                     mul3_x, add2_y, steps, do_use_weight, weight_decay_rate, name)
    return update, nextv, nextm

def lamb_apply_weight_assign(input0, input1, input2, input3, input4, name=None):
    if context.executing_eagerly():
      raise RuntimeError("tf.lamb_apply_weight_assign() is not compatible with "
                        "eager execution.")
    result = gen_npu_ops.lamb_apply_weight_assign(input0, input1, input2, input3, input4, name)
    return result

class PReLU(Layer):
    """Parametric Rectified Linear Unit.

    It follows:

    ```
      f(x) = alpha * x for x < 0
      f(x) = x for x >= 0
    ```
    where `alpha` is a learned array with the same shape as x.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Args:
      alpha_initializer: Initializer function for the weights.
      alpha_regularizer: Regularizer for the weights.
      alpha_constraint: Constraint for the weights.
      shared_axes: The axes along which to share learnable
        parameters for the activation function.
        For example, if the incoming feature maps
        are from a 2D convolution
        with output shape `(batch, height, width, channels)`,
        and you wish to share parameters across space
        so that each filter only has one set of parameters,
        set `shared_axes=[1, 2]`.
    """

    def __init__(self,
                 alpha_initializer='zeros',
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        param_shape = [1,]
        self.alpha = self.add_weight(
            shape=param_shape,
            name='alpha',
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint)
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        return prelu(inputs, self.alpha)

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(PReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
