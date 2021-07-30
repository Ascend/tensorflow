#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless REQUIRED by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

USE_DEFAULT = object()

class TrainSpec(object):
    def __new__(cls, input_fn, max_steps=None, hooks=None):
        pass

class EvalSpec(object):
    def __new__(cls,
              input_fn,
              steps=100,
              name=None,
              hooks=None,
              exporters=None,
              start_delay_secs=120,
              throttle_secs=600):
        pass

class Estimator(object):
    def __init__(self, model_fn, model_dir=None, config=None, params=None,
               warm_start_from=None):
        pass

    def train(self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):
        pass

class Model(object):
    def __init__(self, *args, **kwargs):
        pass
    def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):
        pass
    def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          **kwargs):
        pass
    def fit_generator(self,
                    generator,
                    steps_per_epoch=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    validation_steps=None,
                    validation_freq=1,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0):
        pass

class Session(object):
    def __init__(self, target='', graph=None, config=None):
        pass

class InteractiveSession(object):
    def __init__(self, target='', graph=None, config=None):
        pass

def MonitoredTrainingSession(
    master='',  # pylint: disable=invalid-name
    is_chief=True,
    checkpoint_dir=None,
    scaffold=None,
    hooks=None,
    chief_only_hooks=None,
    save_checkpoint_secs=USE_DEFAULT,
    save_summaries_steps=USE_DEFAULT,
    save_summaries_secs=USE_DEFAULT,
    config=None,
    stop_grace_period_secs=120,
    log_step_count_steps=100,
    max_wait_secs=7200,
    save_checkpoint_steps=USE_DEFAULT,
    summary_dir=None):
    pass

class Supervisor(object):
    def __init__(self,
               graph=None,
               ready_op=USE_DEFAULT,
               ready_for_local_init_op=USE_DEFAULT,
               is_chief=True,
               init_op=USE_DEFAULT,
               init_feed_dict=None,
               local_init_op=USE_DEFAULT,
               logdir=None,
               summary_op=USE_DEFAULT,
               saver=USE_DEFAULT,
               global_step=USE_DEFAULT,
               save_summaries_secs=120,
               save_model_secs=600,
               recovery_wait_secs=30,
               stop_grace_secs=120,
               checkpoint_basename="model.ckpt",
               session_manager=None,
               summary_writer=USE_DEFAULT,
               init_fn=None,
               local_init_run_options=None):
        pass
    def managed_session(self,
                      master="",
                      config=None,
                      start_standard_services=True,
                      close_summary_writer=True):
        pass