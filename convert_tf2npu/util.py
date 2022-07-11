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

"""Public functions for logging messages"""

import ctypes
import sys
import platform
import os
import util_global
from file_op import write_conver_report
from file_op import mkdir
from log import logger_success_report
from log import logger_failed_report
from log import logger_need_migration_doc
import config


def log_msg(lineno, msg):
    """Log message during conversion"""
    content = util_global.get_value('path') + ':' + str(lineno) + ' ' + msg
    logger_success_report.info(content)


def log_info(lineno, msg, file):
    """Log information during conversion"""
    content = (util_global.get_value('path', '') + ':' + str(lineno) +
               ' change ' + util_global.get_value(msg)[1] +
               ' to ' + util_global.get_value(msg)[2])
    print(content)
    write_conver_report(content, file)


def log_warning(msg):
    """Log warning during conversion"""
    os.system("cd .")
    print("".join(["\033[1;33mWARNING\033[0m:", msg]))
    logger_success_report.info(msg)


def log_success_report(lineno, msg):
    """Log sussess report"""
    content = (util_global.get_value('path', '') + ':' + str(lineno) +
               ' change ' + util_global.get_value(msg)[1] +
               ' to ' + util_global.get_value(msg)[2])
    logger_success_report.info(content)
    util_global.set_value('report_file_status', (util_global.get_value('report_file_status') | 0b1))


def log_failed_report(lineno, msg):
    """Log fail report"""
    content = "".join([util_global.get_value('path'), ":", str(lineno), " ", msg, " is not support migration."])
    os.system("cd .")
    print("".join(["\033[1;31mERROR\033[0m:", content]))
    logger_failed_report.info(content)
    util_global.set_value('report_file_status', (util_global.get_value('report_file_status') | 0b10))


def log_migration_report(lineno, msg):
    """Log migration report"""
    content = (util_global.get_value('path', '') + ':' + str(lineno) + ' "' + msg +
               '" feature needs to be migrated manually, Please refer to the migration guide.' +
               util_global.get_value(msg)[0])
    print(content)
    logger_need_migration_doc.info(content)
    util_global.set_value('report_file_status', (util_global.get_value('report_file_status') | 0b100))


def ask_the_distributed_mode(node, prompt, warning_msg):
    """Ask user to specify distributed mode"""
    while not util_global.get_value('already_check_distributed_mode_arg', False):
        message = input(prompt)
        if message in ("c", "continue"):
            content = "".join([util_global.get_value('path'), ":", str(getattr(node, 'lineno')), warning_msg])
            os.system("cd .")
            print("".join(["\033[1;33mWARNING\033[0m:", content]))
            logger_failed_report.info(content)
            util_global.set_value('report_file_status', (util_global.get_value('report_file_status') | 0b10))
            break
        elif message == "exit":
            sys.exit()
        else:
            print("Input is error, please enter 'exit' or 'c' or 'continue'.")


def log_hvd_distributed_mode_error(node):
    """Log error when hvd distributed mode is not valid"""
    if not util_global.get_value("distributed_mode", ""):
        prompt = ("As the '-d' option is not included, distributed porting will not be performed. "
                  "Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
        warning_msg = " is tf_strategy api. As the '-d' option is not included, " \
                      "distributed porting will not be performed."
        ask_the_distributed_mode(node, prompt, warning_msg)
    else:
        prompt = (
            "The '-d' argument conflicts with the Tensorflow distributed strategy in your script, "
            "which means that Tensorflow distributed porting will not be performed. "
            "Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
        warning_msg = " is tf_strategy api. The '-d' argument conflicts with the Tensorflow distributed strategy"
        ask_the_distributed_mode(node, prompt, warning_msg)
    util_global.set_value('already_check_distributed_mode_arg', True)


def log_strategy_distributed_mode_error(node):
    """Log error when strategy distributed mode is not valid"""
    if not util_global.get_value("distributed_mode", ""):
        prompt = ("As the '-d' option is not included, distributed porting will not be performed. "
                  "Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
        warning_msg = " is horovod api. As the '-d' option is not included, distributed porting will not be performed."
        ask_the_distributed_mode(node, prompt, warning_msg)
    else:
        prompt = (
            "The '-d' argument conflicts with the Horovod distributed strategy in your script, "
            "which means that Horovod distributed porting will not be performed. "
            "Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
        warning_msg = " is horovod api. The '-d' argument conflicts with the Horovod distributed strategy " \
                      "in your script"
        ask_the_distributed_mode(node, prompt, warning_msg)
    util_global.set_value('already_check_distributed_mode_arg', True)


def log_warning_main_arg_not_set():
    """Log error when main file for keras script is not set"""
    while not util_global.get_value('already_check_main_arg', False):
        message = input(config.param_config.main_arg_not_set_promt)
        if message in ("continue", "c"):
            break
        elif message == "exit":
            sys.exit()
        else:
            print("Input is error, please enter 'exit' or 'c' or 'continue'.")
    util_global.set_value('already_check_main_arg', True)


def check_path_length(path):
    if platform.system().lower() == 'windows':
        return len(path) <= 256
    elif platform.system().lower() == 'linux':
        names = path.split('/')
        for name in names:
            if len(name) > 255:
                return False
        return True
    else:
        return True


def get_dir_size_and_py_num(folder):
    size = 0
    py_files_num = 0
    threshold_files_num = 5000
    for root, dirs, files in os.walk(folder):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        for file in files:
            if file.endswith(".py"):
                py_files_num += 1
                if py_files_num > threshold_files_num:
                    break
    return size, py_files_num


def get_dir_free_space(folder):
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(folder), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value
    else:
        st = os.statvfs(folder)
        return st.f_bavail * st.f_frsize


def check_input_and_output_dir(input_dir, output_dir):
    mkdir(output_dir)
    if platform.system().lower() == 'linux':
        import pwd
        current_usr = pwd.getpwuid(os.getuid())[0]
        input_dir_usr = pwd.getpwuid(os.stat(input_dir).st_uid).pw_name
        output_dir_usr = pwd.getpwuid(os.stat(output_dir).st_uid).pw_name
        if current_usr != input_dir_usr or current_usr != output_dir_usr:
            while True:
                message = input("The owner of the path set by '-i' or '-o' is inconsistent with the current user, "
                                "please check. Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
                if message in ("c", "continue"):
                    break
                elif message == "exit":
                    sys.exit()
                else:
                    print("Input is error, please enter 'exit' or 'c' or 'continue'.")
    input_dir_size, py_files_num = get_dir_size_and_py_num(input_dir)
    output_dir_free_space = get_dir_free_space(output_dir)
    if input_dir_size > output_dir_free_space:
        content = "".join(["The output path: ", output_dir, " does not have enough space."])
        os.system("cd .")
        print("".join(["\033[1;31mERROR\033[0m:", content]))
        sys.exit(1)
    threshold_files_size = 50 * 1024 * 1024 * 1024
    threshold_files_num = 5000
    if input_dir_size > threshold_files_size or py_files_num > threshold_files_num:
        while True:
            message = input("The number of files in the path set by '-i' is too large, and the conversion will "
                            "takes a long time. Enter 'continue' or 'c' to continue or enter 'exit' to exit: ")
            if message in ("c", "continue"):
                break
            elif message == "exit":
                sys.exit()
            else:
                print("Input is error, please enter 'exit' or 'c' or 'continue'.")
    if not check_path_length(input_dir) or not check_path_length(output_dir):
        print("The length of input/output dir is invalid, please change it and try again.")
        sys.exit(1)
