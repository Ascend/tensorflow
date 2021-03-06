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
import os
import shutil
import util_global
import pandas as pd

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
    file = open(out_file, 'w')
    file.write(dst_content)
    file.close()

def write_report_after_conver(new_file_path, report_file, dst_content):
    mkdir(new_file_path)
    file = open(os.path.join(new_file_path, report_file), 'w')
    file.write(dst_content)
    file.close()

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
                file = open(os.path.join(report_path, file), 'a')
                file.write(content)
                file.write("\r\n")
                file.write("\r\n")
                file.close()
        times = times - 1
    util_global.set_value('report_file_status', 0)

def write_conver_report(content, file):
    report_path = util_global.get_value('report')
    mkdir(report_path)
    file = open(os.path.join(report_path, file), 'a')
    file.write(content)
    file.write("\r\n")
    file.close()

def write_analysis_report(content, file):
    report_path = util_global.get_value('report')
    mkdir(report_path)
    file = open(os.path.join(report_path, file), 'a')
    file.write(content)
    file.write("\r\n")
    file.close()

def abs_join(abs1, abs2):
    abs2 = os.fspath(abs2)
    abs2 = os.path.splitdrive(abs2)[1]
    abs2 = abs2.strip('\\/') or abs2
    return os.path.join(abs1, abs2)

def scan_file(file_name, api, lineno):
    api_list = pd.read_excel(util_global.get_value('list'))
    api_module = api_list['模块名'].values.tolist()
    api_name = api_list['API名'].values.tolist()
    api_support = api_list['支持度'].values.tolist()
    api_advice = api_list['迁移建议'].values.tolist()

    script_name = []
    code_line = []
    code_module = []
    code_api = []
    support_type = []
    migrate_advice = []
    for i in range(len(api)):
        name = api[i]
        class_name = '.'.join(name.split('.')[:-1])
        script_name.append(file_name)
        code_api.append(name)
        code_line.append(lineno[i])
        if name in api_name:
            code_module.append(api_module[api_name.index(name)])
            support_type.append(api_support[api_name.index(name)])
            migrate_advice.append(api_advice[api_name.index(name)])
        elif class_name in api_name:
            code_module.append(api_module[api_name.index(class_name)])
            support_type.append(api_support[api_name.index(class_name)])
            migrate_advice.append(api_advice[api_name.index(class_name)])
        else:
            code_module.append('Unknown')
            support_type.append('NA')
            migrate_advice.append('Unknown')
    analyse_result = pd.DataFrame({'脚本文件名': script_name, '代码行': code_line,
                                   '模块名': code_module, 'API名': code_api,
                                   '支持度': support_type, '迁移建议': migrate_advice})

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
    support_type = analysis_report['支持度'].values.tolist()

    # eliminate duplicated data
    eliminate_dup_api = []
    eliminate_dup_type = []
    for item in code_api:
        if not item in eliminate_dup_api:
            eliminate_dup_api.append(item)
            eliminate_dup_type.append(support_type[code_api.index(item)])

    # api statistics
    api_analysis = "1.In brief: Total API: {}, in which Support: {}, " \
                   "Support after migrated by tool: {}, Support after migrated manually: {}, " \
                   "Analysing: {}, Unsupport: {}, Deprecated: {}, NA: {}".format(len(code_api),
                   support_type.count('支持'), support_type.count('支持但需工具迁移'),
                   support_type.count('支持但需手工迁移'), support_type.count('分析中'),
                   support_type.count('不支持'), support_type.count('废弃'), support_type.count('NA'))

    api_eliminate_dup = "2.After eliminate duplicate: Total API: {}, in which Support: {}, " \
                        "Support after migrated by tool: {}, Support after migrated manually: {}, " \
                        "Analysing: {}, Unsupport: {}, Deprecated: {}, NA: {}".format(len(eliminate_dup_api),
                        eliminate_dup_type.count('支持'), eliminate_dup_type.count('支持但需工具迁移'),
                        eliminate_dup_type.count('支持但需手工迁移'), eliminate_dup_type.count('分析中'),
                        eliminate_dup_type.count('不支持'), eliminate_dup_type.count('废弃'),
                        eliminate_dup_type.count('NA'))
    content = (api_analysis + '\n' + api_eliminate_dup)
    print(content)
    write_analysis_report(content, 'api_brief_report.txt')