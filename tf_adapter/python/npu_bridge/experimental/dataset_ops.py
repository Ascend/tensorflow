#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
"""Python wrappers for Datasets."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.dataset_ops import StructuredFunctionWrapper

from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import function_utils
from tensorflow.python.eager import function as eager_function
from tensorflow.python.training.tracking import tracking

from npu_bridge.helper import helper
gen_npu_cpu_ops = helper.get_gen_ops()


class NpuUnaryDataset(dataset_ops.UnaryDataset):
    """Abstract class representing a dataset with one input."""

    def __init__(self, input_dataset, variant_tensor, output_device="cpu"):
        self._input_dataset = input_dataset
        if output_device not in ["cpu", "npu"]:
            raise ValueError("Invalid type for output_device: %s , available value: cpu or npu." % output_device)
        self._output_device = output_device
        super(dataset_ops.UnaryDataset, self).__init__(variant_tensor)

    def check_output_device(self, func_name=""):
        if self._output_device == "cpu":
            return True
        else:
            print("Current dataset whose output is %s is not support %s() function."
                % (self._output_device, func_name))
            return False

    def map(self,
            map_func,
            num_parallel_calls=None,
            deterministic=None,
            name=None):
        """Maps `map_func` across the elements of this dataset."""

        if self.check_output_device("map"):
            return super(NpuUnaryDataset, self).map(map_func,
                                                    num_parallel_calls=num_parallel_calls,
                                                    deterministic=deterministic,
                                                    name=name)
        else:
            return None

    def concatenate(self, dataset, name=None):
        if self.check_output_device("concatenate"):
            return super(NpuUnaryDataset, self).concatenate(dataset, name=name)
        else:
            return None

    def prefetch(self, buffer_size, name=None):
        if self.check_output_device("prefetch"):
            return super(NpuUnaryDataset, self).prefetch(buffer_size, name=name)
        else:
            return None

    def repeat(self, count=None, name=None):
        if self.check_output_device("repeat"):
            return super(NpuUnaryDataset, self).repeat(count=count, name=name)
        else:
            return None

    def enumerate(self, start=0, name=None):
        if self.check_output_device("enumerate"):
            return super(NpuUnaryDataset, self).enumerate(start=start, name=name)
        else:
            return None

    def shuffle(self,
                buffer_size,
                seed=None,
                reshuffle_each_iteration=None,
                name=None):
        if self.check_output_device("shuffle"):
            return super(NpuUnaryDataset, self).shuffle(buffer_size,
                                                        seed=seed,
                                                        reshuffle_each_iteration=reshuffle_each_iteration,
                                                        name=name)
        else:
            return None

    def cache(self, filename="", name=None):
        if self.check_output_device("cache"):
            return super(NpuUnaryDataset, self).cache(filename=filename, name=name)
        else:
            return None

    def take(self, count, name=None):
        if self.check_output_device("take"):
            return super(NpuUnaryDataset, self).take(count, name=name)
        else:
            return None

    def skip(self, count, name=None):
        if self.check_output_device("skip"):
            return super(NpuUnaryDataset, self).skip(count, name=name)
        else:
            return None

    def shard(self, num_shards, index, name=None):
        if self.check_output_device("shard"):
            return super(NpuUnaryDataset, self).shard(num_shards, index, name=name)
        else:
            return None

    def batch(self,
              batch_size,
              drop_remainder=False,
              num_parallel_calls=None,
              deterministic=None,
              name=None):
        if self.check_output_device("batch"):
            return super(NpuUnaryDataset, self).batch(batch_size,
                                                      drop_remainder=drop_remainder,
                                                      num_parallel_calls=num_parallel_calls,
                                                      deterministic=deterministic,
                                                      name=name)
        else:
            return None

    def padded_batch(self,
                     batch_size,
                     padded_shapes=None,
                     padding_values=None,
                     drop_remainder=False,
                     name=None):
        if self.check_output_device("padded_batch"):
            return super(NpuUnaryDataset, self).padded_batch(batch_size,
                                                             padded_shapes=padded_shapes,
                                                             padding_values=padding_values,
                                                             drop_remainder=drop_remainder,
                                                             name=name)
        else:
            return None

    def flat_map(self, map_func, name=None):
        if self.check_output_device("flat_map"):
            return super(NpuUnaryDataset, self).flat_map(map_func, name=name)
        else:
            return None

    def interleave(self,
                   map_func,
                   cycle_length=None,
                   block_length=None,
                   deterministic=None,
                   name=None):
        if self.check_output_device("interleave"):
            return super(NpuUnaryDataset, self).interleave(map_func,
                                                           cycle_length=cycle_length,
                                                           block_length=block_length,
                                                           deterministic=deterministic,
                                                           name=name)
        else:
            return None

    def filter(self, predicate, name=None):
        if self.check_output_device("filter"):
            return super(NpuUnaryDataset, self).filter(predicate, name=name)
        else:
            return None

    def apply(self, transformation_func):
        if self.check_output_device("apply"):
            return super(NpuUnaryDataset, self).apply(transformation_func)
        else:
            return None

    def window(self, size, shift=None, stride=1, drop_remainder=False, name=None):
        if self.check_output_device("window"):
            return super(NpuUnaryDataset, self).window(size,
                                                       shift=shift,
                                                       stride=stride,
                                                       drop_remainder=drop_remainder,
                                                       name=name)
        else:
            return None

    def unbatch(self, name=None):
        if self.check_output_device("unbatch"):
            return super(NpuUnaryDataset, self).unbatch(name=name)
        else:
            return None

    def with_options(self, options, name=None):
        if self.check_output_device("with_options"):
            return super(NpuUnaryDataset, self).with_options(options, name=name)
        else:
            return None


class MapDataset(NpuUnaryDataset):
    """A `Dataset` that maps a function over elements in its input in parallel."""

    def __init__(self,
                 input_dataset,
                 map_func,
                 num_parallel_npu,
                 deterministic=False,
                 output_device="cpu",
                 preserve_cardinality=False,
                 use_legacy_function=False,
                 name=None):
        """See `Dataset.map()` for details."""
        self._input_dataset = input_dataset
        self._map_func = StructuredFunctionWrapper(
            map_func,
            self._transformation_name(),
            dataset=input_dataset,
            use_legacy_function=use_legacy_function)
        self._deterministic = deterministic
        self._preserve_cardinality = preserve_cardinality
        self._num_parallel_npu = ops.convert_to_tensor(
            num_parallel_npu, dtype=dtypes.int64, name="num_parallel_npu")
        self._name = name
        variant_tensor = gen_npu_cpu_ops.npu_map_dataset(
            input_dataset._variant_tensor,  # pylint: disable=protected-access
            self._map_func.function.captured_inputs,
            f=self._map_func.function,
            num_parallel_calls=self._num_parallel_npu,
            output_device=self._output_device,
            deterministic=self._deterministic,
            preserve_cardinality=self._preserve_cardinality,
            **self._common_args)
        super(MapDataset, self).__init__(input_dataset, variant_tensor, output_device)

    @property
    def element_spec(self):
        return self._map_func.output_structure

    def _functions(self):
        return [self._map_func]

    def _transformation_name(self):
        return "Dataset.experimental.map()"


class MapAndBatchDataset(NpuUnaryDataset):
    """A `Dataset` that maps a function over a batch of elements."""

    def __init__(self, input_dataset, map_func, batch_size,
                 drop_remainder, num_parallel_npu,
                 output_device, use_legacy_function=False):
        self._input_dataset = input_dataset
        self._output_device = output_device
        self._map_func = StructuredFunctionWrapper(
            map_func,
            "tf.data.experimental.map_and_batch()",
            dataset=input_dataset,
            use_legacy_function=use_legacy_function)
        self._batch_size_t = ops.convert_to_tensor(
            batch_size, dtype=dtypes.int64, name="batch_size")
        self._drop_remainder_t = ops.convert_to_tensor(
            drop_remainder, dtype=dtypes.bool, name="drop_remainder")
        self._num_parallel_npu_t = ops.convert_to_tensor(
            num_parallel_npu, dtype=dtypes.int64, name="num_parallel_npu")

        constant_drop_remainder = tensor_util.constant_value(self._drop_remainder_t)
        # pylint: disable=protected-access
        if constant_drop_remainder:
            # pylint: disable=g-long-lambda
            self._element_spec = nest.map_structure(
                lambda component_spec: component_spec._batch(
                    tensor_util.constant_value(self._batch_size_t)),
                self._map_func.output_structure)
        else:
            self._element_spec = nest.map_structure(
                lambda component_spec: component_spec._batch(None),
                self._map_func.output_structure)
        # pylint: enable=protected-access
        variant_tensor = gen_npu_cpu_ops.npu_map_and_batch_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            self._map_func.function.captured_inputs,
            f=self._map_func.function,
            batch_size=self._batch_size_t,
            num_parallel_calls=self._num_parallel_npu_t,
            drop_remainder=self._drop_remainder_t,
            output_device=self._output_device,
            preserve_cardinality=True,
            **self._flat_structure)
        super(MapAndBatchDataset, self).__init__(input_dataset, variant_tensor, output_device)

    @property
    def element_spec(self):
        return self._element_spec

    def _functions(self):
        return [self._map_func]


#pylint: disable=redefined-builtin
def map(map_func, num_parallel_calls=None, num_parallel_npu=28,
    deterministic=None, output_device="cpu"):
    """Map function for create a MapDatasetOp."""

    def _apply_fn(dataset):
        return MapDataset(dataset,
                          map_func,
                          num_parallel_npu=num_parallel_npu,
                          deterministic=deterministic,
                          output_device=output_device)

    return _apply_fn


def map_and_batch(map_func,
                  batch_size,
                  num_parallel_batches=None,
                  drop_remainder=False,
                  num_parallel_calls=None,
                  num_parallel_npu=28,
                  output_device="cpu"):
    """Fused implementation of `map` and `batch`.

    Maps `map_func` across `batch_size` consecutive elements of this dataset
    and then combines them into a batch. Functionally, it is equivalent to `map`
    followed by `batch`. This API is temporary and deprecated since input pipeline
    optimization now fuses consecutive `map` and `batch` operations automatically.

    Args:
      map_func: A function mapping a nested structure of tensors to another
        nested structure of tensors.
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      num_parallel_batches: (Optional.) A `tf.int64` scalar `tf.Tensor`,
        representing the number of batches to create in parallel. On one hand,
        higher values can help mitigate the effect of stragglers. On the other
        hand, higher values can increase contention if CPU is scarce.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in case its size is smaller than
        desired; the default behavior is not to drop the smaller batch.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number of elements to process in parallel. If not
        specified, `batch_size * num_parallel_batches` elements will be processed
        in parallel. If the value `tf.data.AUTOTUNE` is used, then
        the number of parallel calls is set dynamically based on available CPU.
      npu_parallel_npu: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number of elements process in parallel on NPU.
      output_device: Optional. A name for the dataset executed on. ("cpu" or "npu")


    Returns:
      A `Dataset` transformation function, which can be passed to
      `tf.data.Dataset.apply`.

    Raises:
      ValueError: If both `num_parallel_batches` and `num_parallel_calls` are
        specified.
    """
    if num_parallel_batches is None and num_parallel_calls is None:
        num_parallel_calls = batch_size
    elif num_parallel_batches is not None and num_parallel_calls is None:
        num_parallel_calls = batch_size * num_parallel_batches
    elif num_parallel_batches is not None and num_parallel_calls is not None:
        raise ValueError(
            "`map_and_batch` allows only one of `num_parallel_batches` and "
            "`num_parallel_calls` to be set, but "
            f"`num_parallel_batches` was set to {num_parallel_batches} "
            f"and `num_parallel_calls` as set to {num_parallel_calls}.")

    def _apply_fn(dataset):
        return MapAndBatchDataset(dataset, map_func, batch_size,
                                  drop_remainder, num_parallel_npu, output_device)

    return _apply_fn
