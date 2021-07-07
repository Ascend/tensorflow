#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_contextlib
from tensorflow.core.framework import attr_value_pb2


@tf_contextlib.contextmanager
def keep_dtype_scope():
    with ops.get_default_graph()._attr_scope({'_keep_dtype': attr_value_pb2.AttrValue(b=True)}):
        yield
