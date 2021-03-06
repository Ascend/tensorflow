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
import util_global
from conver_by_ast import conver_ast
from file_op import mkdir
from file_op import mkdir_and_copyfile
from file_op import write_report_terminator
from file_op import abs_join
import pandas as pd
from file_op import get_api_statistic
from file_op import adjust_index

def conver():
    print("Begin conver, input file: " + util_global.get_value('input'))
    out_path = util_global.get_value('output')
    dst_path = os.path.split(util_global.get_value('input').rstrip('\\/'))[-1]
    dst_path_new = dst_path + util_global.get_value('timestap')
    conver_path = os.walk(util_global.get_value('input'))
    report_dir = util_global.get_value('report')
    mkdir(report_dir)
    report_xlsx = os.path.join(report_dir, 'api_analysis_report.xlsx')
    util_global.set_value('generate_dir_report', pd.DataFrame())

    for path, dir_list, file_list in conver_path:
        for file_name in file_list:
            out_path_dst = abs_join(dst_path_new, path.split(dst_path)[1])
            if file_name.endswith(".py"):
                util_global.set_value('path', os.path.join(path, file_name))
                mkdir(os.path.join(out_path, out_path_dst))
                conver_ast(path, out_path_dst, file_name)
                if util_global.get_value('need_conver', False):
                    content = "Finish conver file: " + os.path.join(path, file_name)
                    print(content)
                    write_report_terminator(content)
                else:
                    mkdir_and_copyfile(path, abs_join(out_path, out_path_dst), file_name)
            else:
                mkdir_and_copyfile(path, abs_join(out_path, out_path_dst), file_name)

    adjust_index()
    analysis_report = util_global.get_value('generate_dir_report')
    if analysis_report.empty:
        print('No api data in the report')
    else:
        analysis_report.to_excel(report_xlsx, index=True)
        get_api_statistic(analysis_report)
    print("Finish conver, output file: " + out_path + "; report file: " + util_global.get_value('report'))
