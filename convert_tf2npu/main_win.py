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

"""User access to initiate script migration on Windows platform"""

import os
import sys
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter import ttk
import pandas as pd
from tkintertable import TableCanvas
import util_global
from conver import conver


class Analyse(object):
    """Use Tkinter to display ayalysis result"""
    def __init__(self, parent):
        self.root = parent
        self.root.title("Tensorflow1.15 API Analysis")

        self.script_path = tk.StringVar()
        tk.Label(self.root, text="原始脚本路径：").grid(row=0, stick=tk.W)
        tk.Entry(self.root, textvariable=self.script_path, width=30).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.root, text="路径选择", command=self.select_script_path).grid(row=0, column=2)

        self.output_path = tk.StringVar()
        tk.Label(self.root, text="输出迁移脚本路径：").grid(row=1, stick=tk.W)
        tk.Entry(self.root, textvariable=self.output_path, width=30).grid(row=1, column=1, padx=10, pady=10)
        tk.Button(self.root, text="路径选择", command=self.select_output_path).grid(row=1, column=2)

        self.report_path = tk.StringVar()
        tk.Label(self.root, text="输出分析报告路径：").grid(row=2, stick=tk.W)
        tk.Entry(self.root, textvariable=self.report_path, width=30).grid(row=2, column=1, padx=10, pady=10)
        tk.Button(self.root, text="路径选择", command=self.select_report_path).grid(row=2, column=2)

        self.main_file = tk.StringVar()
        tk.Label(self.root, text="执行入口脚本：").grid(row=3, stick=tk.W)
        tk.Entry(self.root, textvariable=self.main_file, width=30).grid(row=3, column=1, padx=10, pady=10)
        tk.Button(self.root, text="文件选择", command=self.select_main_file).grid(row=3, column=2)

        tk.Label(self.root, text="分布式模式：").grid(row=4, stick=tk.W)
        self.distributed_mode = ttk.Combobox(self.root, values=["horovod", "tf_strategy"], width=28)
        self.distributed_mode.grid(row=4, column=1, padx=10, pady=10)

        tk.Button(self.root, text="开始分析", command=self.analyse).grid(row=5, column=2, padx=10, pady=10)
        tk.Button(self.root, text="退出", command=exit).grid(row=5, column=1, padx=10, pady=10, stick=tk.E)

    def hide(self):
        """Do not display root window"""
        self.root.withdraw()

    def show(self):
        """Show updated information to window"""
        self.root.update()
        self.root.deiconify()

    def back_to_main(self, new_frame):
        """Back to root window"""
        new_frame.destroy()
        self.show()

    def select_script_path(self):
        """Select input script directory"""
        path_ = askdirectory()
        self.script_path.set(path_)

    def select_report_path(self):
        """Select report directory"""
        path_ = askdirectory()
        self.report_path.set(path_)

    def select_output_path(self):
        """Select output directory"""
        path_ = askdirectory()
        self.output_path.set(path_)

    def select_main_file(self):
        """Select main file for keras script"""
        main_file_ = askopenfilename()
        self.main_file.set(main_file_)

    def get_output_dir(self):
        """Get selected output directory"""
        output = "output" + util_global.get_value('timestap')
        if self.output_path.get():
            output = self.output_path.get()
            if str(output).endswith('/'):
                output = output[:-1]
            output = output.replace('\\', '/')
        return output

    def get_main_file(self):
        """Get selected main file"""
        main_file = ""
        if self.main_file.get():
            main_file = self.main_file.get()
            if os.path.isfile(main_file):
                main_path = os.path.dirname(main_file)
                select_file = os.path.basename(main_file)
                main_path = main_path.replace('\\', '/')
                main_file = os.path.join(main_path, select_file)
            else:
                raise ValueError("--main args must be existed files")
        return main_file

    def get_report_dir(self):
        """Get selected report directory"""
        report = "report" + util_global.get_value('timestap')
        report_suffix = report
        if self.report_path.get():
            report = self.report_path.get()
            if str(report).endswith('/'):
                report = report[:-1]
            report = os.path.join(report, report_suffix)
            report = report.replace('\\', '/')
        return report

    def get_distributed_mode(self):
        """Get selected distributed mode"""
        distributed_mode = ""
        if self.distributed_mode.get():
            distributed_mode = self.distributed_mode.get()
        return distributed_mode

    def analyse(self):
        """Initiate API analysis"""
        util_global._init()

        # verify input arguments
        if not self.script_path.get():
            raise ValueError("Parameter error, please select the folder of source script to be converted.")
        input_dir = self.script_path.get()
        if str(input_dir).endswith('/'):
            input_dir = input_dir[:-1]
        input_dir = input_dir.replace('\\', '/')

        support_list = os.path.dirname(os.path.abspath(__file__)) + "/tf1.15_api_support_list.xlsx"

        output = self.get_output_dir()
        report = self.get_report_dir()
        main_file = self.get_main_file()
        distributed_mode = self.get_distributed_mode()

        if input_dir + '/' in output + '/' or input_dir + '/' in report + '/':
            print("<output> or <report> could not be the subdirectory of <input>, please try another option.")
            sys.exit(2)

        util_global.set_value('input', input_dir)
        util_global.set_value('list', support_list)
        util_global.set_value('output', output)
        util_global.set_value('report', report)
        util_global.set_value('main', main_file)
        util_global.set_value('distributed_mode', distributed_mode)
        conver()
        self.hide()

        new_frame = tk.Toplevel()
        new_frame.title("Report")
        handler = lambda: self.back_to_main(new_frame)
        tk.Button(new_frame, text='重新开始分析', command=handler).grid(row=5, column=2, padx=10, pady=10, stick=tk.W)
        tk.Button(new_frame, text='退出', command=exit).grid(row=5, column=1, padx=10, pady=10, stick=tk.E)

        # load analysis report
        if self.report_path.get() == '':
            self.report_path.set(os.getcwd())

        report_dir = self.report_path.get()
        lateset = []
        for item in os.listdir(report_dir):
            if 'report_npu' in item:
                lateset.append(item)
        lateset.sort()

        report_path = os.path.join(report_dir, lateset[-1], 'api_analysis_report.xlsx')
        if not os.path.exists(report_path):
            print("No api analysis report generated.")
            return
        report = pd.read_excel(report_path)
        file_index = report['序号'].values.tolist()
        file_name = report['脚本文件名'].values.tolist()
        code_line = report['代码行'].values.tolist()
        code_module = report['模块名'].values.tolist()
        code_api = report['API名'].values.tolist()
        support_type = report['工具迁移API支持度'].values.tolist()
        migrate_advice = report['说明'].values.tolist()

        table = TableCanvas(new_frame)
        table.show()
        table.addColumn('6')
        table.addColumn('7')
        for i in range(len(file_name) - 10):
            table.addRow()

        for i in range(len(file_name)):
            table.model.setValueAt(file_index[i], i, 0)
            table.model.setValueAt(file_name[i], i, 1)
            table.model.setValueAt(code_line[i], i, 2)
            table.model.setValueAt(code_module[i], i, 3)
            table.model.setValueAt(code_api[i], i, 4)
            table.model.setValueAt(support_type[i], i, 5)
            table.model.setValueAt(migrate_advice[i], i, 6)

        table.model.columnlabels['1'] = '序号'
        table.model.columnlabels['2'] = '脚本文件名'
        table.model.columnlabels['3'] = '代码行'
        table.model.columnlabels['4'] = '模块名'
        table.model.columnlabels['5'] = 'API名'
        table.model.columnlabels['6'] = '工具迁移API支持度'
        table.model.columnlabels['7'] = '说明'

        table.show()


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('430x260')
    app = Analyse(root)
    root.mainloop()
