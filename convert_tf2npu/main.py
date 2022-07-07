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

"""Entry point to initiate script migration"""

import os
import sys
import getopt
import util_global
from conver import conver
from log import init_loggers
import util
import config


def get_para_input(arg):
    """Get input directory parameter"""
    input_dir = os.path.abspath(arg)
    if str(input_dir).endswith('/'):
        input_dir = input_dir[:-1]
    input_dir = input_dir.replace('\\', '/')
    return input_dir


def get_para_output(arg):
    """Get output directory parameter"""
    output = os.path.abspath(arg)
    if str(output).endswith('/'):
        output = output[:-1]
    output = output.replace('\\', '/')
    return output


def get_para_report(arg, report_suffix):
    """Get report directory parameter"""
    report = os.path.abspath(arg)
    if str(report).endswith('/'):
        report = report[:-1]
    report = os.path.join(report, report_suffix)
    report = report.replace('\\', '/')
    return report


def get_para_main(arg):
    """Get absolute path for main.py"""
    main_file = ""
    if os.path.isfile(arg):
        main_file = os.path.abspath(arg)
        main_path = os.path.dirname(main_file)
        select_file = os.path.basename(main_file)
        main_path = main_path.replace('\\', '/')
        main_file = os.path.join(main_path, select_file)
    else:
        raise ValueError("--main args must be existing files")
    return main_file


def get_para_distributed_mode(arg):
    """Get distributed mode parameter"""
    if arg not in ["horovod", "tf_strategy"]:
        raise ValueError("--distributed_mode or -d must be one of ['horovod', 'tf_strategy']")
    return arg


def para_check_and_set(argv):
    """Verify validation and set parameters"""
    input_dir = "npu_input"
    support_list = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.param_config.support_list_filename)
    output = "output" + util_global.get_value('timestap')
    report = "report" + util_global.get_value('timestap')
    report_suffix = report
    main_file = ""
    distributed_mode = ""

    try:
        opts, _ = getopt.getopt(argv, config.param_config.short_opts,
                                config.param_config.long_opts)
    except getopt.GetoptError:
        print(config.param_config.opt_err_prompt)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(config.param_config.opt_help)
            sys.exit()
        elif opt in ("-i", "--input"):
            input_dir = get_para_input(arg)
        elif opt in ("-l", "--list"):
            support_list = arg
        elif opt in ("-o", "--output"):
            output = get_para_output(arg)
        elif opt in ("-r", "--report"):
            report = get_para_report(arg, report_suffix)
        elif opt in ("-m", "--main"):
            main_file = get_para_main(arg)
        elif opt in ("-c", "--compat"):
            util_global.set_value('is_compat_v1', True)
        elif opt in ("-d", "--distributed_mode"):
            distributed_mode = get_para_distributed_mode(arg)

    if input_dir == "npu_input":
        raise ValueError("Please check -i or --input.")

    if input_dir + '/' in output + '/' or input_dir + '/' in report + '/':
        print("<output> or <report> could not be the subdirectory of <input>, please try another option.")
        sys.exit(2)

    if distributed_mode == "horovod" and 'c' in config.param_config.short_opts \
        and not util_global.get_value('is_compat_v1', False):
        print("Horovod distribute mode is only supported in compat.v1 mode now! Please wait for our later updates.")
        sys.exit(2)

    util.check_input_and_output_dir(input_dir, output)
    util_global.set_value('input', input_dir)
    util_global.set_value('list', support_list)
    util_global.set_value('output', output)
    util_global.set_value('report', report)
    util_global.set_value('main', main_file)
    util_global.set_value('distributed_mode', distributed_mode)
    init_loggers(report)

if __name__ == "__main__":
    util_global.init()
    util_global.set_value('already_check_distributed_mode_arg', False)
    util_global.set_value('already_check_main_arg', False)
    para_check_and_set(sys.argv[1:])
    conver()
