#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
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

"""config of tf2.x convert tool"""

import util_global

HELP_INFO = '    this tool just support to convert tf-2.6 scripts.\n' \
            '    main.py -i <input> -l <list> -o <output> -r <report> -m <main>\n' \
            'or: main.py --input=<input> --list=<list> --output=<output> --report=<report> --main=<main>\n' \
            '-i or --input:  The source script to be converted.\n' \
            '-l or --list:  The list of supported api, Default value: TF2.6_api_support_list.xlsx\n' \
            '-o or --output: The destination script after converted, Default value: output_npu_***/\n' \
            '-r or --report: Conversion report, Default value: report_npu_***/\n' \
            '-m or --main: The executed entry *.py file, default:None\n' \
            '-c or --compat: The source script used compat.v1 module\n' \
            '-d or --distributed_mode: The distribute mode to choose, including horovod distributed '\
            'and tensorflow distributed strategy. the value should be one of [horovod, tf_strategy]'

param_config = util_global.ParamConfig(
    short_opts='hi:l:o:r:m:ci:d:',
    long_opts = ["help", "input=", "list=", "output=", "report=", "main=", "compat", "distributed_mode"],
    opt_err_prompt =
        'Parameter error, please check.\n{}'.format(HELP_INFO),
    opt_help = HELP_INFO,
    support_list_filename = 'TF2.6_api_support_list.xlsx',
    main_arg_not_set_promt = "As it is necessary to initiate npu device, ensure that the Python entry script contains "\
            "the main function or the '-m' option is included to avoid porting failures. "\
            "Enter 'continue' or 'c' to continue or enter 'exit' to exit: "
)

import_list = ['tf', 'hvd']
keras_dropout_api = []

npu_func_map = {
        "config": ["npu_config_proto", "config_proto"],
        "hooks": ["npu_hooks_append", "hooks_list"],
        "callbacks": ["npu_callbacks_append", "callbacks_list"],
        "optimizer": ["npu.distribute.npu_distributed_keras_optimizer_wrapper", "optimizer"]}

TOOL_TITLE = "Tensorflow2.6 API Analysis"