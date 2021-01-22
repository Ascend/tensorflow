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
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename
from tkintertable import TableCanvas

class Analyse(object):
    def __init__(self, parent):
        self.root = parent
        self.root.title("Tensorflow1.15 API Analysis")

        self.script_path = tk.StringVar()
        tk.Label(self.root, text="原始脚本路径：").grid(row=0, stick=tk.E)
        tk.Entry(self.root, textvariable=self.script_path, width=30).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.root, text="路径选择", command=self.select_script_path).grid(row=0, column=2)

        self.api_list = tk.StringVar()
        tk.Label(self.root, text="API支持度清单：").grid(row=1, stick=tk.E)
        tk.Entry(self.root, textvariable=self.api_list, width=30).grid(row=1, column=1, padx=10, pady=10)
        tk.Button(self.root, text="文件选择", command=self.select_api_file).grid(row=1, column=2)

        self.output_path = tk.StringVar()
        tk.Label(self.root, text="输出迁移脚本路径：").grid(row=2, stick=tk.W)
        tk.Entry(self.root, textvariable=self.output_path, width=30).grid(row=2, column=1, padx=10, pady=10)
        tk.Button(self.root, text="路径选择", command=self.select_output_path).grid(row=2, column=2)

        self.report_path = tk.StringVar()
        tk.Label(self.root, text="输出分析报告路径：").grid(row=3, stick=tk.W)
        tk.Entry(self.root, textvariable=self.report_path, width=30).grid(row=3, column=1, padx=10, pady=10)
        tk.Button(self.root, text="路径选择", command=self.select_report_path).grid(row=3, column=2)

        tk.Button(self.root, text="开始分析", command=self.analyse).grid(row=5, column=2, padx=10, pady=10)
        tk.Button(self.root, text="退出", command=exit).grid(row=5, column=1, padx=10, pady=10, stick=tk.E)

    def hide(self):
        self.root.withdraw()

    def show(self):
        self.root.update()
        self.root.deiconify()

    def back_to_main(self, new_frame):
        new_frame.destroy()
        self.show()

    def select_script_path(self):
        path_ = askdirectory()
        self.script_path.set(path_)

    def select_report_path(self):
        path_ = askdirectory()
        self.report_path.set(path_)

    def select_output_path(self):
        path_ = askdirectory()
        self.output_path.set(path_)

    def select_api_file(self):
        file_ = askopenfilename()
        self.api_list.set(file_)

    def analyse(self):
        call_main_py = 'python main.py -i ' +  self.script_path.get() + \
                       ' -l ' + self.api_list.get() + \
                       ' -o ' + self.output_path.get() + \
                       ' -r ' + self.report_path.get()
        os.system(call_main_py)
        self.hide()

        new_frame = tk.Toplevel()
        new_frame.title("Report")
        handler = lambda: self.back_to_main(new_frame)
        tk.Button(new_frame, text='重新开始分析', command=handler).grid(row=5, column=2, padx=10, pady=10, stick=tk.W)
        tk.Button(new_frame, text='退出', command=exit).grid(row=5, column=1, padx=10, pady=10, stick=tk.E)

        # load analysis report
        report_dir = self.report_path.get()
        report_path = os.path.join(report_dir, os.listdir(report_dir)[-1], 'api_analysis_report.xlsx')
        if not os.path.exists(report_path):
            print("No api analysis report generated.")
            return
        report = pd.read_excel(report_path)
        file_index = report['序号'].values.tolist()
        file_name = report['脚本文件名'].values.tolist()
        code_line = report['代码行'].values.tolist()
        code_module = report['模块名'].values.tolist()
        code_api = report['API名'].values.tolist()
        support_type = report['支持度'].values.tolist()
        migrate_advice = report['迁移建议'].values.tolist()

        table = TableCanvas(new_frame)
        table.show()
        table.addColumn('6')
        table.addColumn('7')
        for i in range(len(file_name)-10):
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
        table.model.columnlabels['6'] = '支持度'
        table.model.columnlabels['7'] = '迁移建议'

        table.show()

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('425x210')
    app = Analyse(root)
    root.mainloop()






