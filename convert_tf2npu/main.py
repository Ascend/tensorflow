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
import sys
import getopt
import util_global
from file_op import before_clear
from conver import conver

def para_check_and_set(argv):
    input  = "input"
    list = "tf1.15_api_support_list.xlsx"
    output = "output" + util_global.get_value('timestap')
    report = "report" + util_global.get_value('timestap')
    report_suffix = report

    try:
        opts, args = getopt.getopt(argv, "hi:l:o:r:", ["help", "input=", "list=", "output=", "report="])
    except getopt.GetoptError:
        print('Parameter error, please check.')
        print('    main.py -i <input> -l <list> -o <output> -r <report>')
        print('or: main.py --input=<input> --list=<list> --output=<output> --report=<report>')
        print('-i or --input:  The source script to be converted, Default value: input/')
        print('-l or --list:  The list of supported api, Default value: tf1.15_api_support_list.xlsx')
        print('-o or --output: The destination script after converted, Default value: output/')
        print('-r or --report: Conversion report, Default value: report/')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('    main.py -i <input> -l <list> -o <output> -r <report>')
            print('or: main.py --input=<input> --list=<list> --output=<output> --report=<report>')
            print('-i or --input:  The source script to be converted, Default value: input/')
            print('-l or --list:  The list of supported api, Default value: tf1.15_api_support_list.xlsx')
            print('-o or --output: The destination script after converted, Default value: output/')
            print('-r or --report: Conversion report, Default value: report/')
            sys.exit()
        elif opt in ("-i", "--input"):
            input = os.path.abspath(arg)
            if str(input).endswith('/'):
                input = input[0:len(input)-1]
        elif opt in ("-l", "--list"):
            list = arg
        elif opt in ("-o", "--output"):
            output = os.path.abspath(arg)
            if str(output).endswith('/'):
                output = output[0:len(output)-1]
        elif opt in ("-r", "--report"):
            report = os.path.abspath(arg)
            if str(report).endswith('/'):
                report = report[0:len(report)-1]
            report = os.path.join(report, report_suffix)

    if input+'/' in output+'/' or input+'/' in report+'/':
        print("<output> or <report> could not be the subdirectory of <input>, please try another option.")
        sys.exit(2)

    util_global.set_value('input', input)
    util_global.set_value('list', list)
    util_global.set_value('output', output)
    util_global.set_value('report', report)

if __name__ == "__main__":
    util_global._init()
    para_check_and_set(sys.argv[1:])
    conver()