#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

"""log class"""

import os
import logging
from logging.handlers import RotatingFileHandler

def logger_create(logger_name, log_dir):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.INFO)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    rotating = RotatingFileHandler(filename=log_dir+'/'+logger_name, maxBytes=1024**2,
                                   backupCount=10, encoding='utf-8')
    rotating.setFormatter(logging.Formatter('%(message)s'))
    rotating.setLevel(logging.INFO)
    logger.addHandler(rotating)
    return logger

def init_loggers(log_dir='.'):
    global logger_success_report
    logger_success_report = logger_create('success_report.txt', log_dir)

    global logger_failed_report
    logger_failed_report = logger_create('failed_report.txt', log_dir)

    global logger_need_migration_doc
    logger_need_migration_doc = logger_create('need_migration_doc.txt', log_dir)

    global logger_api_brief_report
    logger_api_brief_report = logger_create('api_brief_report.txt', log_dir)

init_loggers()