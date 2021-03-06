# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""All bert ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops


from npu_bridge.helper import helper
npu_aicore_ops = helper.get_gen_ops();

@ops.RegisterGradient("FastGelu")
def _fast_gelu_grad(op, grad):
  """The gradient for `fast_gelu`.

  Args:
      op: The `fast_gelu` `Operation` that we are differentiating, which we can use
          to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `fast_gelu` op.

  Returns:
      Gradients with respect to the input of `fast_gelu`.
  """
  return [npu_aicore_ops.fast_gelu_grad(grad, op.inputs[0])]  # List of one Tensor, since we have one input


def centralization(x, axes, name=None):
    """
    centralization op
        return x - reduce_mean(x, axes)
    """
    x = ops.convert_to_tensor(x, name="x")
    result = npu_aicore_ops.centralization(x, axes, name=name)
    return result

@ops.RegisterGradient("PRelu")
def prelu_grad(op, grad):
    dx, da = npu_aicore_ops.p_relu_grad(grad, op.inputs[0], op.inputs[1])
    return [dx, da]

def prelu(x, weight):
    return npu_aicore_ops.p_relu(x, weight)

# go/tf-wildcard-import