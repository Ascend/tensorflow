#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""utils for APIs used in Functional API construction"""

from keras.layers import core
from keras.utils import generic_utils
from keras import backend

_API_NAME_ATTR = 'npu_api_names'
_name_to_npu_symbol_map = dict()


def npu_symbol_register(names, func):
    """
    register npu symbol by name

    Args:
      names: names of API without prefix `npu.`
      func: the decorate function or class
    """
    if not names:
        return func
    if isinstance(names, list):
        for name in names:
            _name_to_npu_symbol_map[name] = func
    else:
        _name_to_npu_symbol_map[names] = func
    setattr(func, _API_NAME_ATTR, names)
    return func


def get_npu_symbol_from_name(name):
    """return npu symbol by symbol name"""
    return _name_to_npu_symbol_map.get(name)


def get_one_name_for_npu_symbol(symbol):
    """return one name of the npu symbol"""
    if not hasattr(symbol, _API_NAME_ATTR):
        return None
    api_names = getattr(symbol, _API_NAME_ATTR)
    if isinstance(api_names, list):
        return api_names[0]
    return api_names


@generic_utils.register_keras_serializable(package='npu_device.utils.npu_wrapper')
class NpuOpLambda(core.TFOpLambda):
    """
    Wraps NPU API symbols in a `TFOpLambda` object extends by `Layer`.

    It is used by the Functional API construction when users call
    a supported NPU symbol on KerasTensors to replace TF symbol.
    """
    def __init__(self, function, **kwargs):
        name = get_one_name_for_npu_symbol(function)
        if 'name' not in kwargs:
            kwargs['name'] = backend.unique_object_name(
                name, zero_based=True, avoid_observed_names=True)
        super(NpuOpLambda, self).__init__(function, **kwargs)
        self.symbol = name

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        symbol_name = config.get('function')
        function = get_npu_symbol_from_name(symbol_name)
        if not function:
            raise ValueError(
                'NPU symbol `npu.%s` could not be found.' % symbol_name)
        config['function'] = function

        return cls(**config)


def npu_functional_support(op):
    """Decorator that adds a functional API handling wrapper to an op."""
    def wrapper(*args, **kwargs):
        try:
            return op(*args, **kwargs)
        except (TypeError, ValueError):
            if any(
                isinstance(x, keras_tensor.KerasTensor)
                for x in nest.flatten([args, kwargs])):
                return NpuOpLambda(op)(*args, **kwargs)
            else:
                raise
    return wrapper