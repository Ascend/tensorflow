#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

class VisitCall(ast.NodeVisitor):
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

class VisitAttr(ast.NodeVisitor):
    def __init__(self):
        self.attrs = []
        self.linenos = []
        self._current = []
        self._in_attr = False

    def visit_Attribute(self, node):
        self._in_attr = True
        self._current.append(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node):
        if self._in_attr:
            self._current.append(node.id)
            self.attrs.append('.'.join(self._current[::-1]))
            self.linenos.append(getattr(node, "lineno", "None"))
            # Reset the state
            self._current = []
            self._in_attr = False
        self.generic_visit(node)

class VisitUnsupportImport(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.import_modules = []
        self.modules = []
        self.unsupport = ['cupy', 'cupyx', 'pynvml']

    def visit_ImportFrom(self, node):
        if node.module != None:
            self.modules = node.module.split('.')
        for value in node.names:
            if isinstance(value, ast.alias):
                classes = value.name.split('.')
                if len(self.modules) == 0:
                    break
                if self.modules[0] in self.unsupport:
                    self.imports.append(classes[0])
                    self.import_modules.append(self.modules[0])
        self.generic_visit(node)

    def visit_Import(self, node):
        for value in node.names:
            if isinstance(value, ast.alias):
                self.modules = value.name.split('.')
                if len(self.modules) == 0:
                    break
                if self.modules[0] in self.unsupport:
                    self.import_modules.append(self.modules[0])
                    if value.asname != None:
                        self.imports.append(value.asname)
                    else:
                        self.imports.append(self.modules[0])
        self.generic_visit(node)

def get_tf_api(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        source = file.read()
    tree = ast.parse(source)
    visitor = VisitCall()
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

def get_tf_enume(file_name, enume_list):
    with open(file_name, 'r', encoding='utf-8') as file:
        source = file.read()
    tree = ast.parse(source)
    visitor = VisitAttr()
    visitor.visit(tree)

    # get tensorflow enume api
    api = []
    lineno = []
    for i in range(len(visitor.attrs)):
        if visitor.attrs[i] in enume_list:
            api.append(visitor.attrs[i])
            lineno.append(visitor.linenos[i])
    return api, lineno

def get_unsupport_api(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        source = file.read()
    tree = ast.parse(source)
    visitor = VisitCall()
    visitor.visit(tree)
    unsupportor = VisitUnsupportImport()
    unsupportor.visit(tree)

    #get unsupport api
    api = []
    lineno = []
    module = []
    for i in range(len(visitor.calls)):
        imports = visitor.calls[i].split('.')[0]
        if imports in unsupportor.imports or visitor.calls[i].startswith('nvml'):
            if visitor.calls[i].startswith('nvml'):
                module.append('pynvml')
            else:
                index = unsupportor.imports.index(imports)
                module.append(unsupportor.import_modules[index])
            api.append(visitor.calls[i])
            lineno.append(visitor.linenos[i])
    return api, module, lineno