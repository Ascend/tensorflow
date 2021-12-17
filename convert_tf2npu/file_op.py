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
import os
import re
import subprocess
import shutil
import util_global
import pandas as pd
from visit_by_ast import get_tf_enume
from visit_by_ast import get_unsupport_api

def before_clear():
    exit_folder = os.path.exists(util_global.get_value('output'))
    if exit_folder:
        shutil.rmtree(util_global.get_value('output'))
    exit_folder = os.path.exists(util_global.get_value('report'))
    if exit_folder:
        shutil.rmtree(util_global.get_value('report'))

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def mkdir_and_copyfile(srcfile, dstpath, file_name):
    mkdir(dstpath)
    shutil.copyfile(os.path.join(srcfile, file_name), os.path.join(dstpath, file_name))

def write_output_after_conver(out_file, dst_content):
    with open(out_file, 'w') as file:
        file.write(dst_content)

def write_report_after_conver(new_file_path, report_file, dst_content):
    mkdir(new_file_path)
    with open(os.path.join(new_file_path, report_file), 'w') as file:
        file.write(dst_content)

def get_bit_val(value, index):
    if value & (1 << index):
        return 1
    else:
        return 0

def write_report_terminator(content):
    report_path = util_global.get_value('report')
    value = util_global.get_value('report_file_status')
    times = value.bit_length()
    while times > 0:
        if get_bit_val(value, times - 1):
            file = util_global.get_value('report_file')[times - 1]
            if os.path.exists(os.path.join(report_path, file)):
                with open(os.path.join(report_path, file), 'a') as file:
                    file.write(content)
                    file.write("\r\n")
                    file.write("\r\n")
        times = times - 1
    util_global.set_value('report_file_status', 0)

def write_conver_report(content, file):
    report_path = util_global.get_value('report')
    mkdir(report_path)
    with open(os.path.join(report_path, file), 'a') as file:
        file.write(content)
        file.write("\r\n")

def check_warning(lineno, api_msg):
    # raise warning when api is related to element range check
    pattern = r'tf.*.is_finite'
    if re.match(pattern, api_msg):
        doc_msg = "{}, chapter: {}".format('"Tensorflow模型迁移和训练', '"tf.is_finite接口手工迁移" and "Loss Scale"')
        content = "".join([util_global.get_value('path', ''), ":", str(lineno), ", You used tensorflow api: ",
                           api_msg, ", It is suggested to use npu api. Please refer to the online document: ",
                           doc_msg])
        subprocess.run(["cd", "."], shell=True)
        print("".join(["\033[1;33mWARNING\033[0m:", content]), flush=True)
        write_conver_report(content, util_global.get_value('report_file')[1])

def log_failed_api(lineno, api_msg, is_third_party):
    subprocess.run(["cd", "."], shell=True)
    if is_third_party:
        content = "".join([util_global.get_value('path', ''), ":", str(lineno), ", NPU Unsupport API: ", api_msg,
                           ", Please modify user scripts manually."])
        print("".join(["\033[1;31mERROR\033[0m:", content]), flush=True)

    elif api_msg.startswith("hvd"):
        doc_msg = "{}, chapter: {}".format('"Tensorflow模型迁移和训练', '"Horovod脚本迁移示例"')
        content = "".join([util_global.get_value('path', ''), ":", str(lineno), ", NPU Unsupport API: ", api_msg,
                           ", Please refer to the online document: ", doc_msg])
        print("".join(["\033[1;33mWARNING\033[0m:", content]), flush=True)

    elif api_msg.startswith("tf.is_"):
        doc_msg = "{}, chapter: {}".format('"Tensorflow模型迁移和训练', '"tf.is_finite接口手工迁移" and "Loss Scale"')
        content = "".join([util_global.get_value('path', ''), ":", str(lineno), ", NPU Unsupport API: ", api_msg,
                           ", Please refer to the online document: ", doc_msg])
        print("".join(["\033[1;33mWARNING\033[0m:", content]), flush=True)

    else:
        content = "".join([util_global.get_value('path', ''), ":", str(lineno), ", NPU Unsupport API: ", api_msg])
        print("".join(["\033[1;31mERROR\033[0m:", content]), flush=True)
    write_conver_report(content, util_global.get_value('report_file')[1])

def abs_join(abs1, abs2):
    abs2 = os.fspath(abs2)
    abs2 = os.path.splitdrive(abs2)[1]
    abs2 = abs2.strip('\\/') or abs2
    return os.path.join(abs1, abs2)

def scan_file(path, file_name, api, lineno):
    api_list = pd.read_excel(util_global.get_value('list'), sheet_name=0)
    api_module = api_list['模块名'].values.tolist()
    api_name = api_list['API名'].values.tolist()
    api_support = api_list['工具迁移API支持度'].values.tolist()
    api_advice = api_list['说明'].values.tolist()

    script_name = []
    code_line = []
    code_module = []
    code_api = []
    support_type = []
    migrate_advice = []

    for i in range(len(api)):
        name = api[i]
        if name in api_name:
            script_name.append(file_name)
            code_api.append(name)
            code_line.append(lineno[i])
            code_module.append(api_module[api_name.index(name)])
            support_type.append(api_support[api_name.index(name)])
            migrate_advice.append(api_advice[api_name.index(name)])

            api_support_type = api_support[api_name.index(name)]
            # print warning message of npu supported api
            if api_support_type == '支持（无需迁移）' or api_support_type == '兼容类':
                check_warning(lineno[i], name)

            # print error message when api is unsupported on npu
            if api_support_type == '分析中（特性商用时不应该存在）' or \
                    api_support_type == '不支持（无迁移方案，建议用户不使用）':
                log_failed_api(lineno[i], name, is_third_party=False)

    # search for tf enumeration
    enume_list = pd.read_excel(util_global.get_value('list'), sheet_name=1)
    enume_name = enume_list['API名'].values.tolist()
    (enume, lineno) = get_tf_enume(os.path.join(path, file_name), enume_name)

    for i in range(len(enume)):
        name = enume[i]
        class_name = '.'.join(name.split('.')[:-1])
        if name not in code_api and class_name not in code_api:
            if class_name in api_name:
                script_name.append(file_name)
                code_api.append(class_name)
                code_line.append(lineno[i])
                code_module.append(api_module[api_name.index(class_name)])
                support_type.append(api_support[api_name.index(class_name)])
                migrate_advice.append(api_advice[api_name.index(class_name)])

                # print error message when api is unsupported on npu
                api_support_type = api_support[api_name.index(class_name)]
                if api_support_type == '分析中（特性商用时不应该存在）' or \
                    api_support_type == '不支持（无迁移方案，建议用户不使用）':
                    log_failed_api(lineno[i], class_name, is_third_party=False)

    # record unsupported api
    (unsupport, unsupport_module, lineno) = get_unsupport_api(os.path.join(path, file_name))
    for i in range(len(unsupport)):
        script_name.append(file_name)
        code_api.append(unsupport[i])
        code_line.append(lineno[i])
        code_module.append(unsupport_module[i])
        support_type.append('不支持（无迁移方案，建议用户不使用）')
        migrate_advice.append('第三方非TF官网API，暂不支持')

        # print error message for unsupported api
        log_failed_api(lineno[i], unsupport[i], is_third_party=True)

    analyse_result = pd.DataFrame({'脚本文件名': script_name, '代码行': code_line,
                                   '模块名': code_module, 'API名': code_api,
                                   '工具迁移API支持度': support_type, '说明': migrate_advice})

    # when there are tf apis used in script, analysis report will be generated
    report = util_global.get_value('generate_dir_report')
    if len(script_name):
        report = report.append(analyse_result)
        util_global.set_value('generate_dir_report', report)

def adjust_index():
    report = util_global.get_value('generate_dir_report')
    index_column = []
    for i in range(len(report)):
        index_column.append(i + 1)
    report.index = index_column
    report.index.name = '序号'
    util_global.set_value('generate_dir_report', report)

def get_api_statistic(analysis_report):
    code_api = analysis_report['API名'].values.tolist()
    support_type = analysis_report['工具迁移API支持度'].values.tolist()

    # eliminate duplicated data
    eliminate_dup_api = []
    eliminate_dup_type = []
    for item in code_api:
        if not item in eliminate_dup_api:
            eliminate_dup_api.append(item)
            eliminate_dup_type.append(support_type[code_api.index(item)])

    # api statistics
    api_analysis = "1.In brief: Total API: {}, in which Support: {}, " \
                   "API support after migration: {}, " \
                   "Network training support after migration: {}, " \
                   "Not support but no impact on migration: {}, " \
                   "Not support or recommended: {}, " \
                   "Compatible: {}, " \
                   "Deprecated: {}, " \
                   "Analysing: {}".format(len(code_api),
                   support_type.count('支持（无需迁移）'),
                   support_type.count('工具迁移后API功能支持'),
                   support_type.count('工具迁移后训练功能打通'),
                   support_type.count('不支持（不影响迁移，用户无需干预）'),
                   support_type.count('不支持（无迁移方案，建议用户不使用）'),
                   support_type.count('兼容类'),
                   support_type.count('废弃类'),
                   support_type.count('分析中（特性商用时不应该存在）'))

    api_eliminate_dup = "2.After eliminate duplicate: Total API: {}, in which Support: {}, " \
                        "API support after migration: {}, " \
                        "Network training support after migration: {}, " \
                        "Not support but no impact on migration: {}, " \
                        "Not support or recommended: {}, " \
                        "Compatible: {}, " \
                        "Deprecated: {}, " \
                        "Analysing: {}".format(len(eliminate_dup_api),
                        eliminate_dup_type.count('支持（无需迁移）'),
                        eliminate_dup_type.count('工具迁移后API功能支持'),
                        eliminate_dup_type.count('工具迁移后训练功能打通'),
                        eliminate_dup_type.count('不支持（不影响迁移，用户无需干预）'),
                        eliminate_dup_type.count('不支持（无迁移方案，建议用户不使用）'),
                        eliminate_dup_type.count('兼容类'),
                        eliminate_dup_type.count('废弃类'),
                        eliminate_dup_type.count('分析中（特性商用时不应该存在）'))
    content = (api_analysis + '\n' + api_eliminate_dup)
    print(content)
    write_conver_report(content, 'api_brief_report.txt')