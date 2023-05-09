#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -------------------------------------------------------------------
# Purpose:
# Copyright 2023 Huawei Technologies Co., Ltd. All rights reserved.
# -------------------------------------------------------------------

import hashlib
import argparse
import re
import os

PATTERN_0 = r'^ge.*\.txt$'
PATTERN_1 = r'^ge.*\.pbtxt$'


def get_file_hash(filename):
    with open(filename, 'rb') as f:
        obj = hashlib.sha256()
        obj.update(f.read())
        file_hash = obj.hexdigest()
        return file_hash


def remove_duplicate_files(files_to_remove):
    hashes = {}
    for file_to_remove in files_to_remove:
        file_to_remove_hash = get_file_hash(file_to_remove)
        if file_to_remove_hash in hashes:
            print(f"Removing duplicate file: {file_to_remove}")
            os.remove(file_to_remove)
        else:
            hashes[file_to_remove_hash] = file_to_remove


def is_dump_graph(filename):
    if re.match(PATTERN_0, filename) or re.match(PATTERN_1, filename):
        return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dumped file deduplication tool')
    parser.add_argument('-d', '--dir', type=str, required=True, help='the file directory to deduplicate')
    args = parser.parse_args()
    dir_path = args.dir
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if is_dump_graph(file):
                file_list.append(os.path.join(root, file))
    remove_duplicate_files(file_list)
