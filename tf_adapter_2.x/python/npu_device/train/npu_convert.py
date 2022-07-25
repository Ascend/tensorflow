#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

"""NPU implemented dropout"""

import functools
import absl.logging as logging

from npu_device.npu_device import global_npu_ctx
from npu_device.train import convert_dropout
from tensorflow.python.eager import context
import tensorflow as tf

_GRAPH_MODE = "graph"
_EAGER_MODE = "eager"
"""
The Version define rules as follows：
eg1: '2.6.2', Support Scope: version = 2.6.2
eg2: '2.6.*', Support Scope: 2.6.0 <= version <= 2.6.999, the max sub-version is:999
eg3: '2.*', Support Scope: 2.0.0 <= version <= 2.999.999
eg4: '*', Support Scope: 0.0.0 <= version <= 999.999.999
eg5: '2.6.1~2.6.5', Support Scope: 2.6.1 <= version <= 2.6.5
The version information defined in others is illegal
"""
_version_range = []
_DOT_NUM = 2


def parse_star_ver(ver=str):
    """
    parse version with *
    :param ver:  version strings
    :return: True
    """
    low = ""
    high = ""
    for c in ver:
        if c != '*':
            low = low + c
            high = high + c
        else:
            low = low + '0'
            high = high + '999'
    for i in range(low.count('.'), _DOT_NUM):
        low = low + '.0'
        high = high + '.999'
    _version_range.append([[int(x) for x in low.split('.')], [int(x) for x in high.split('.')]])
    return True


def parse_range_version(ver=str):
    """
    parse version with ~
    :param ver: version strings
    :return: True or False
    """
    v = ver.split('~')
    low = v[0]
    high = v[1]
    if low.count('.') != _DOT_NUM or high.count('.') != _DOT_NUM:
        return False
    low_ver = [int(x) for x in low.split('.')]
    high_ver = [int(x) for x in high.split('.')]
    if low_ver > high_ver:
        return False
    _version_range.append([low_ver, high_ver])
    return True


def parse_version(ver=str):
    """
    parse version only with number and dot
    :param ver: version strings
    :return: True
    """
    _version_range.append([[int(x) for x in ver.split('.')], [int(x) for x in ver.split('.')]])
    return True


def check_version_format(version=str):
    """
    check illegal version definitions
    :param version: version strings
    :return: True or False
    """
    if version.replace('.', '0').replace('~', '0').replace('*', '0').isdecimal() is False:
        return False
    if ('*' in version and '~' in version) \
            or version.startswith('~') or version.endswith('~') \
            or version.startswith('.') or version.endswith('.') \
            or version.count('*') > 1 or version.count('~') > 1:
        return False
    if '*' in version and version.endswith('*') is False:
        return False
    return True


def parse_version_range(version=str):
    """
    parse version string
    :param version: version strings
    :return: True or False
    """
    if '*' in version:
        parse_star_ver(version)
    elif '~' in version:
        parse_range_version(version)
    elif version.count('.') == _DOT_NUM:
        parse_version(version)
    else:
        return False
    return True


def verify_version():
    """
    check whether the version is supported
    :return: True or False
    """
    version = tf.__version__
    ver = [int(x) for x in version.split('.')]
    for ver_range in _version_range:
        if ver_range[0] <= ver <= ver_range[1]:
            return True
    return False


def judge_version_support(version_list=list):
    """
    parse version and check whether is supported
    :param version_list:
    :return:
    """
    for ver in version_list:
        if check_version_format(ver) is False:
            logging.error(f"Parse version error, {ver}")
            return False
        if parse_version_range(ver) is False:
            logging.error(f"Parse version error, {ver}")
            return False
    return verify_version()


def api_func_convert(wrapfunc, **kwargs):
    """
    Define Decorator Function for Replace Api function
    :param wrapfunc: Wrapper Function for Api function
    :param kwargs: Wrapper parameter define
    :return: Wrapped API
    """
    if wrapfunc is None or callable(wrapfunc) is False:
        raise Exception("Input New Function is None")
    wrapper_func = wrapfunc
    support_version = kwargs.get('version', '')
    support_mode = kwargs.get('mode', '')

    def convert_func(apifunc):
        origin_func = apifunc
        if judge_version_support(support_version) is False:
            return origin_func
        if _EAGER_MODE in support_mode:
            eager_func = wrapper_func
        else:
            eager_func = origin_func
        if _GRAPH_MODE in support_mode:
            graph_func = wrapper_func
        else:
            graph_func = origin_func
        setattr(wrapper_func, '__OriginCall__', origin_func.__call__)

        def wrapper(*args, **kwargs):
            if global_npu_ctx() is None:  # NPU disable
                return origin_func(*args, **kwargs)
            if context.executing_eagerly():
                return eager_func(*args, **kwargs)
            else:
                return graph_func(*args, **kwargs)

        return wrapper

    return convert_func


dropout_convert = functools.partial(api_func_convert, ToolType='ApiConvert')


def npu_convert_api():
    """
    Convert TF API to Npu Api
    :return:
    """
    convert_dropout.dropout_api_convert()
