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
import ast
import re

class VisitByAst(ast.NodeVisitor):
    def __init__(self):
        self.calls = []
        self.linenos = []
        self._current = []
        self._in_call = False

    def visit_Call(self, node):
        self._current = []
        self._in_call = True
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if self._in_call:
            self._current.append(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node):
        if self._in_call:
            self._current.append(node.id)
            self.calls.append('.'.join(self._current[::-1]))
            self.linenos.append(getattr(node, "lineno", "None"))
            # Reset the state
            self._current = []
            self._in_call = False
        self.generic_visit(node)

def get_tf_api(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        source = file.read()
    tree = ast.parse(source)
    visitor = VisitByAst()
    visitor.visit(tree)

    # get tensorflow related api
    api = []
    lineno = []
    import_list = ['tf', 'hvd']
    for module in import_list:
        for i in range(len(visitor.calls)):
            if module + '.' in visitor.calls[i] and visitor.calls[i].split('.')[0] == module:
                api.append(visitor.calls[i])
                lineno.append(visitor.linenos[i])
    return api, lineno




