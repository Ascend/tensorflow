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

from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow_estimator.python.estimator import run_config
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator

def model_to_npu_estimator(keras_model=None,
                           keras_model_path=None,
                           custom_objects=None,
                           model_dir=None,
                           checkpoint_format='saver',
                           config=None,
                           job_start_file=''):
    """Constructs an `NPUEstimator` instance from given keras model.
    """
    tf_estimator = model_to_estimator(keras_model=keras_model,
                                      keras_model_path=keras_model_path,
                                      custom_objects=custom_objects,
                                      model_dir=model_dir,
                                      config=run_config.RunConfig(model_dir=model_dir),
                                      checkpoint_format=checkpoint_format)

    estimator = NPUEstimator(model_fn=tf_estimator._model_fn,
                             model_dir=model_dir,
                             config=config,
                             job_start_file=job_start_file,
                             warm_start_from=tf_estimator._warm_start_settings)

    return estimator