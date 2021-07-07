#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_contextlib
from tensorflow.core.framework import attr_value_pb2


@tf_contextlib.contextmanager
def keep_dtype_scope():
    with ops.get_default_graph()._attr_scope({'_keep_dtype': attr_value_pb2.AttrValue(b=True)}):
        yield
