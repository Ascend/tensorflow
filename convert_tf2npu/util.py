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
import util_global
from file_op import write_conver_report
import os

def log_msg(lineno, msg):
    content = util_global.get_value('path') + ':' + str(lineno) + ' ' + msg
    write_conver_report(content, util_global.get_value('report_file')[0])

def log_info(lineno, msg, file):
    content = (util_global.get_value('path', '') + ':' + str(lineno) +
               ' change ' + util_global.get_value(msg)[1] +
               ' to ' + util_global.get_value(msg)[2])
    print(content)
    write_conver_report(content, file)

def log_warning(msg):
    os.system("")
    print("".join(["\033[1;33mWARNING\033[0m:", msg]))
    write_conver_report(msg, util_global.get_value('report_file')[0])

def log_success_report(lineno, msg):
    content = (util_global.get_value('path', '') + ':' + str(lineno) +
               ' change ' + util_global.get_value(msg)[1] +
               ' to ' + util_global.get_value(msg)[2])
    write_conver_report(content, util_global.get_value('report_file')[0])
    util_global.set_value('report_file_status', (util_global.get_value('report_file_status') | 0b1))

def log_failed_report(lineno, msg):
    content = "".join([util_global.get_value('path'), ":", str(lineno), " ", msg, " is not support migration."])
    os.system("")
    print("".join(["\033[1;31mERROR\033[0m:", content]))
    write_conver_report(content, util_global.get_value('report_file')[1])
    util_global.set_value('report_file_status', (util_global.get_value('report_file_status') | 0b10))

def log_migration_report(lineno, msg):
    content = (util_global.get_value('path', '') + ':' + str(lineno) + ' "' + msg +
               '" feature needs to be migrated manually, Please refer to the migration guide.' +
               util_global.get_value(msg)[0])
    print(content)
    write_conver_report(content, util_global.get_value('report_file')[2])
    util_global.set_value('report_file_status', (util_global.get_value('report_file_status') | 0b100))

def ask_the_distributed_mode(node, prompt, warning_msg):
    while not util_global.get_value('already_check_distributed_mode_arg', False):
        message = input(prompt)
        if message == "c" or message == "continue":
            content = "".join([util_global.get_value('path'), ":", str(getattr(node, 'lineno')), warning_msg])
            os.system("")
            print("".join(["\033[1;33mWARNING\033[0m:", content]))
            write_conver_report(content, util_global.get_value('report_file')[1])
            util_global.set_value('report_file_status', (util_global.get_value('report_file_status') | 0b10))
            break
        elif message == "exit":
            exit(0)
        else:
            print("Input is error, please enter 'exit' or 'c' or 'continue'.")

def log_hvd_distributed_mode_error(node):
    if not util_global.get_value("distributed_mode", ""):
        prompt = ("As the '-d' option is not included, distributed porting will not be performed. "
                  "Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
        warning_msg = " is tf_strategy api. As the '-d' option is not included, distributed porting will not be performed."
        ask_the_distributed_mode(node, prompt, warning_msg)
    else:
        prompt = ("The '-d' argument conflicts with the Tensorflow distributed strategy in your script, which means that Tensorflow "
                  "distributed porting will not be performed. Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
        warning_msg = " is tf_strategy api. The '-d' argument conflicts with the Tensorflow distributed strategy"
        ask_the_distributed_mode(node, prompt, warning_msg)
    util_global.set_value('already_check_distributed_mode_arg', True)

def log_strategy_distributed_mode_error(node):
    if not util_global.get_value("distributed_mode", ""):
        prompt = ("As the '-d' option is not included, distributed porting will not be performed. "
                  "Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
        warning_msg = " is horovod api. As the '-d' option is not included, distributed porting will not be performed."
        ask_the_distributed_mode(node, prompt, warning_msg)
    else:
        prompt = ("The '-d' argument conflicts with the Horovod distributed strategy in your script, which means that Horovod "
                  "distributed porting will not be performed. Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
        warning_msg = " is horovod api. The '-d' argument conflicts with the Horovod distributed strategy in your script"
        ask_the_distributed_mode(node, prompt, warning_msg)
    util_global.set_value('already_check_distributed_mode_arg', True)

def log_warning_main_arg_not_set():
    while not util_global.get_value('already_check_main_arg', False):
        message = input("As your script contains Horovod or Keras API, ensure that the Python entry script contains the main function "
                        "and the '-m' option is included to avoid porting failures. "
                        "Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
        if message == "continue" or message == "c":
            break
        elif message == "exit":
            exit(0)
        else:
            print("Input is error, please enter 'exit' or 'c' or 'continue'.")
    util_global.set_value('already_check_main_arg', True)