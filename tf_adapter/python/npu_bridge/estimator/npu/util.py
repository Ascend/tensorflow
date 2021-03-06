from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.util import compat
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.python.ops import resource_variable_ops

_NPU_RUNCONFIG = 'npu_runconfig'
_ITERATIONS_PER_LOOP_VAR = 'iterations_per_loop'
_LOOP_COND_VAR = 'loop_cond'
_CONST_ZERO = 'zero'
_CONST_ONE = 'one'

def check_not_none(value, name):
  """Checks whether `value` is not None."""
  if value is None:
    error_msg = '{} must not be None'.format(name)
    raise ValueError(error_msg)


def check_positive_integer(value, name):
  """Checks whether `value` is a positive integer."""
  if not isinstance(value, six.integer_types):
    error_msg = '{} must be int, got {}'.format(name, type(value))
    raise TypeError(error_msg)
  if value <= 0:
    error_msg = '{} must be positive, got {}'.format(name, value)
    raise ValueError(error_msg)


def check_nonnegative_integer(value, name):
  """Checks whether `value` is a nonnegative integer."""
  if not isinstance(value, six.integer_types):
    error_msg = '{} must be int, got {}'.format(name, type(value))
    raise TypeError(error_msg)

  if value < 0:
    error_msg = '{} must be nonnegative, got {}'.format(name, value)
    raise ValueError(error_msg)

def check_bool_type(value, name):
  """Checks whether `value` is True or false."""
  if not isinstance(value, bool):
    error_msg = '{} must be True or False, got {}'.format(name, value)
    raise TypeError(error_msg)

def convert_bool_to_int(value):
  """convert True/False to 1/0."""
  if value is True:
    return int(1)

  if value is False:
    return int(0)

  return int(-1)

def format_string(value, name):
  """fomat input to string type"""
  if value is None:
    return ""

  if not isinstance(value, six.string_types):
    error_msg = '{} must be string, got {}'.format(name, type(value))
    raise TypeError(error_msg)

  return str(value)

def check_profiling_options(profiling_options=[]):
  """Check profiling options .
  Args:
      profiling_options: Profiling options.
  Return:
      Valid options
  Raise:
      If profiling_options is null or option is not `training_trace` or `task_trace`, `op_trace`'.
  """

  error_mag = 'profiling options must be in `training_trace`, `task_trace` or `op_trace`'

  if len(profiling_options) == 0:
      raise ValueError(error_mag)

  profiling_types = ["training_trace", "task_trace", "op_trace"]
  for option in profiling_options :
      if option not in profiling_types:
          raise ValueError(error_mag)

  result = ":".join(profiling_options)
  return result

def check_path(path):
  """Check path.
  Args:
      path: path.
  Return:
      real path
  Raise:
      if path is valid or not read and write permissions.
  """
  if os.path.exists(path):
    real_path = os.path.realpath(path)
    if not os.path.isdir(real_path):
      raise ValueError("path:%s is not directory." %(path))
    if not os.access(real_path, os.R_OK | os.W_OK):
      raise ValueError("path:%s is not read and write permissions." %(path))
  else:
    raise ValueError("path:%s is not exists." %(path))
  return real_path

def check_mstune_mode(mstune_mode):
  """Check mstune mode .
  Args:
      mstune_mode: mstune_mode: Optimization Task Type."1": model tune; "2": optune;
                                "3": model tune & optune; "4": gradient split tune.
  Return:
      mstune_mode
  Raise:
      If mstune_mode is null or not in ['1', '2', '3', '4'].
  """
  mstune_modes = ['1', '2', '3', '4']
  if mstune_mode not in mstune_modes:
    raise ValueError("mstune_mode is valid, should be in ['1', '2', '3', '4']")

def register_func(var_name):
  ops.register_proto_function(
    '{}_{}'.format(_NPU_RUNCONFIG, var_name),
    proto_type=variable_pb2.VariableDef,
    to_proto=resource_variable_ops._to_proto_fn,
    from_proto=resource_variable_ops._from_proto_fn)

def create_or_get_var(var_name):
  graph = ops.get_default_graph()
  collection_name = '{}_{}'.format(_NPU_RUNCONFIG, var_name)
  iter_vars = graph.get_collection(collection_name)
  if len(iter_vars) == 1:
      return iter_vars[0]
  elif len(iter_vars) > 1:
      raise RuntimeError('Multiple var in collection.')
  ignore_existing = False
  if training_util.get_global_step() is None:
      ignore_existing = True
  with ops.colocate_with(training_util.get_global_step(), ignore_existing=ignore_existing):
      with variable_scope.variable_scope(_NPU_RUNCONFIG, reuse=variable_scope.AUTO_REUSE):
        return variable_scope.get_variable(
          var_name,
          initializer=init_ops.zeros_initializer(),
          shape=[],
          dtype=dtypes.int64,
          trainable=False,
          collections=[collection_name, ops.GraphKeys.LOCAL_VARIABLES],
          use_resource=True)

def set_iteration_per_loop(sess, train_op, iterations_per_loop=1):
  """
  Constructs a set_iteration_per_loop.
  Args:
  sess: A TensorFlow Session that has been created.
  train_op: An Operation that updates the variables
    or applies the specified gradients.
  iterations_per_loop： This is the number of train steps running in NPU
    system before returning to CPU host for each `Session.run`.

  Returns:
  An Operation named IterationOp that executes all its inputs.
  """
  if not isinstance(train_op, ops.Operation):
    raise ValueError(
          "The incoming 'train_op' type is '%s', "
          "and the need type is 'Operation'" %(train_op.dtype.name))
  check_positive_integer(iterations_per_loop, "iterations_per_loop")
  if iterations_per_loop == 1 :
    return train_op

  iterations_per_loop_var = create_or_get_var(_ITERATIONS_PER_LOOP_VAR)
  loop_cond_var = create_or_get_var(_LOOP_COND_VAR)
  const_zero = create_or_get_var(_CONST_ZERO)
  const_one = create_or_get_var(_CONST_ONE)

  iterations_per_loop_var.load(iterations_per_loop - 1, session=sess)
  loop_cond_var.load(0, session=sess)
  const_zero.load(0, session=sess)
  const_one.load(1, session=sess)

  # Add IterationOp denpend on train_op
  group_train_op = tf.group(train_op, name="IterationOp")

  return group_train_op

class IterationPerLoop():
  """
  An object provide two API to create and set iterations_per_loop
  """
  def create_iteration_per_loop_var(self, train_op):
    """
    Constructs a set_iteration_per_loop.
    Args:
        train_op: An Operation that updates the variables
          or applies the specified gradients.
        iterations_per_loop: This is the number of train steps running in NPU
          system before returning to CPU host for each `Session.run`.

    Returns:
    An Operation named IterationOp that executes all its inputs.
    """
    if not isinstance(train_op, ops.Operation):
      raise ValueError(
            "The incoming 'train_op' type is '%s', "
            "and the need type is 'Operation'" %(train_op.dtype.name))

    self._iterations_per_loop_var = create_or_get_var(_ITERATIONS_PER_LOOP_VAR)
    self._loop_cond_var = create_or_get_var(_LOOP_COND_VAR)
    self._const_zero = create_or_get_var(_CONST_ZERO)
    self._const_one = create_or_get_var(_CONST_ONE)

    # Add IterationOp denpend on train_op
    group_train_op = tf.group(train_op, name="IterationOp")

    return group_train_op

  def load_iteration_per_loop_var(self, sess, iterations_per_loop=1):
    """
    Constructs a load_iteration_per_loop_var.
    Args:
    sess: A TensorFlow Session that has been created.
    iterations_per_loop: This is the number of train steps running in NPU
      system before returning to CPU host for each `Session.run`.
    """
    check_positive_integer(iterations_per_loop, "iterations_per_loop")
    self._iterations_per_loop_var.load(iterations_per_loop - 1, session=sess)
    self._loop_cond_var.load(0, session=sess)
    self._const_zero.load(0, session=sess)
    self._const_one.load(1, session=sess)

def variable_initializer_in_host(var_list):
  """Returns an Op that initializes a list of variables.
  If `var_list` is empty, however, the function still returns an Op that can
  be run. That Op just has no effect.
  Args:
    var_list: List of `Variable` objects to initialize.
    name: Optional name for the returned operation.
  Returns:
    An Op that run the initializers of all the specified variables.
  """
  return tf.initializers.variables(var_list, name='var_in_host')

def fair_division(input, number):
    def get_sum(list):
        res = 0
        for item in list:
            res += item.size
        return res

    def get_left_input_sum(list):
        res = 0
        for item in list:
            if item.root_rank_id < 0:
                res += item.size
        return res

    def get_average(list, size):
        large_number_list = []
        average_size = 0
        res = 0
        if size == 1:
            for item in list:
                if item.root_rank_id < 0:
                    res += item.size
            return res
        while True:
            res = 0
            find_large_number = False
            for item in list:
                if item not in large_number_list and item.root_rank_id < 0:
                    res += item.size
            average_size = res // (size - len(large_number_list))
            for item in list:
                if item not in large_number_list and item.root_rank_id < 0 and item.size > res - item.size:
                    find_large_number = True
                    large_number_list.append(item)
            if not find_large_number:
                break
        return average_size

    if number > len(input) or number < 0:
        raise ValueError("'number' is greater than the number of inputs or 'number' is less than 0. ")
    elif number == len(input):
        for i in range(len(input)):
            input[i].root_rank_id = i
        return input

    j = -1
    last_index = 0
    while True:
        j = j+1
        total_number = number - j
        if total_number == 0:
            break
        average_size = get_average(input, total_number)
        tmp_list = []
        tmp_last_index = last_index
        for i in range(tmp_last_index, len(input) - total_number + 1):
            if get_sum(tmp_list) + input[i].size <= average_size:
                input[i].root_rank_id = j
                tmp_list.append(input[i])
                last_index = i+1
            else:
                if len(tmp_list) <= 0:
                    input[i].root_rank_id = j
                    tmp_list.append(input[i])
                    last_index = i+1
                elif (get_sum(tmp_list) + input[i].size - average_size) <= (average_size - get_sum(tmp_list)):
                    input[i].root_rank_id = j
                    tmp_list.append(input[i])
                    last_index = i+1
                break

    return input

class GradDivisionItem():
    def __init__(self, grad, var):
        self.grad = grad
        self.var = var
        self.size = self.__get_size()
        self.root_rank_id = -1

    def __get_size(self):
        size = 1
        grad_shape = self.grad.shape
        if len(grad_shape) <= 0:
            return 0
        for i in range(len(grad_shape)):
            size = size * int(grad_shape[i])
        size = size * self.grad.dtype.size
        return size

_GRADIENTS_AND_VARS = []

def add_grads_and_vars(grads_and_vars, rank_size):
    global _GRADIENTS_AND_VARS
    _GRADIENTS_AND_VARS.clear()
    for grad, var in grads_and_vars:
        if grad is not None:
            item = GradDivisionItem(grad, var)
            _GRADIENTS_AND_VARS.append(item)
    _GRADIENTS_AND_VARS = fair_division(_GRADIENTS_AND_VARS, rank_size)

def get_gid_by_grad(grad):
    gid = -1
    global _GRADIENTS_AND_VARS
    for item in _GRADIENTS_AND_VARS:
        if item.grad.name == grad.name:
            gid = item.root_rank_id
    return gid

def get_gid_by_weight(weight):
    gid = -1
    global _GRADIENTS_AND_VARS
    for item in _GRADIENTS_AND_VARS:
        if item.var.name == weight.name:
            gid = item.root_rank_id
    return gid

def get_all_grad_item():
    global _GRADIENTS_AND_VARS
    return _GRADIENTS_AND_VARS

def set_graph_exec_config(fetch, dynamic_input = False,
                                  dynamic_graph_execute_mode = "dynamic_execute",
                                  dynamic_inputs_shape_range = None):
  """
  add dynamic exec config to operation or tensor.
  Args:
    fetch:
    dynamic_input:Whether Input is dynamic.
    dynamic_graph_execute_mode: Dynamic graph execute mode.
    dynamic_inputs_shape_range: Inputs shape range. In dynamic_execute mode, should be set.
  Returns:
  An fetch that includes dynamic exec config.
  """
  def _set_op_attr(fetch, dynamic_input_attr, dynamic_graph_execute_mode_attr,
                  dynamic_inputs_shape_range_attr):
    if isinstance(fetch, ops.Operation):
      fetch._set_attr("_graph_dynamic_input", dynamic_input_attr)
      fetch._set_attr("_graph_dynamic_graph_execute_mode", dynamic_graph_execute_mode_attr)
      fetch._set_attr("_graph_dynamic_inputs_shape_range", dynamic_inputs_shape_range_attr)
    else:
      fetch.op._set_attr("_graph_dynamic_input", dynamic_input_attr)
      fetch.op._set_attr("_graph_dynamic_graph_execute_mode", dynamic_graph_execute_mode_attr)
      fetch.op._set_attr("_graph_dynamic_inputs_shape_range", dynamic_inputs_shape_range_attr)

  if dynamic_graph_execute_mode != "lazy_recompile" and dynamic_graph_execute_mode != "dynamic_execute":
    raise ValueError("dynamic_graph_execute_mode should be lazy_recompile or dynamic_execute")
  dynamic_input_attr = attr_value_pb2.AttrValue(b = dynamic_input)
  dynamic_graph_execute_mode_attr = attr_value_pb2.AttrValue(s = compat.as_bytes(dynamic_graph_execute_mode))
  if dynamic_inputs_shape_range is None:
    dynamic_inputs_shape_range = ""
  dynamic_inputs_shape_range_attr = attr_value_pb2.AttrValue(s = compat.as_bytes(dynamic_inputs_shape_range))
  if isinstance(fetch, (ops.Operation, ops.Tensor)):
    _set_op_attr(fetch, dynamic_input_attr, dynamic_graph_execute_mode_attr,
                 dynamic_inputs_shape_range_attr)
  elif isinstance(fetch, (tuple, list)):
    for tensor in fetch:
      tensor = set_graph_exec_config(tensor, dynamic_input, dynamic_graph_execute_mode,
                                             dynamic_inputs_shape_range)
  elif isinstance(fetch, str):
    tensor = set_graph_exec_config(ops.get_default_graph().get_tensor_by_name(fetch),
                dynamic_input, dynamic_graph_execute_mode, dynamic_inputs_shape_range)
    return tensor
  else:
    raise ValueError("fetch is invalid, should be op, tensor, list, tuple or tensor name.")
  return fetch

def npu_compile(sess, *fetches):
  sess.run(fetches)