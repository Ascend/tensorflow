#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from functools import reduce


class BaseTableMapPolicy():
    def __init__(self, assign_groups=None):
        self.table_create_infos = []
        if assign_groups is None:
            self.assign_groups = []
        else:
            self.assign_groups = assign_groups
        self.in_slot_size_group = []
        self.slot_to_table = []
        self.table_to_output_slots = []
        self.table_to_input_groups = []
        self.table_to_slot = []

    @staticmethod
    def _is_equal_table_info(info1, info2):
        if info1['embedding_dim'] != info2['embedding_dim']:  # dim of table is the same or not
            print('embedding dim different!, value is %d and %d' % (info1['embedding_dim'], info2['embedding_dim']))
            return False
        return True

    def map_table_infos(self, user_defined_table_infos):
        raise NotImplementedError()

    def _register_new_table_info(self, new_table_info):
        self.table_create_infos.append(new_table_info)
        self.table_to_output_slots.append([])
        self.table_to_input_groups.append([])
        self.table_to_slot.append([])

    def _merge_new_table_info(self, new_table_info, assign_tabld_id):
        main_table_info = self.table_create_infos[assign_tabld_id]
        main_table_info['multihot_lens'] += new_table_info['multihot_lens']
        main_table_info['max_vocabulary_size'] += new_table_info['max_vocabulary_size']

    def _register_table_info(self, new_table_info, assign_tid=-1):
        multihot_lens = new_table_info['multihot_lens']
        in_slot_size = sum(multihot_lens)
        out_slot_size = len(multihot_lens)

        tid = assign_tid
        if tid == -1:
            tid = len(self.table_create_infos)
            self._register_new_table_info(new_table_info)
        else:
            self._merge_new_table_info(new_table_info, tid)

        self.table_to_slot[tid].append(len(self.in_slot_size_group))
        self.table_to_output_slots[tid].append(in_slot_size)
        self.in_slot_size_group.append(in_slot_size)
        self.slot_to_table.append(tid)

    def _map_table_infos(self, user_defined_table_infos, assign_groups):
        self.table_create_infos = []
        assign_groups_flat = reduce(lambda a, b: a+b, assign_groups, [])
        sid_to_gid = reduce(lambda a, b: {**a, **b},
                            [{sid: gid for sid in group}
                             for gid, group in enumerate(assign_groups)], {})
        gid_to_tid = dict()
        for sid, table_info in enumerate(user_defined_table_infos):
            if sid in assign_groups_flat:
                gid = sid_to_gid.get(sid)
                if gid in gid_to_tid:
                    self._register_table_info(table_info, assign_tid=gid_to_tid.get(gid))
                else:
                    tid = len(self.table_create_infos)
                    self._register_table_info(table_info, assign_tid=-1)
                    gid_to_tid[gid] = tid
            else:
                self._register_table_info(table_info, assign_tid=-1)
        return self.table_create_infos


# no slot merge
class NoneTableMapPolicy(BaseTableMapPolicy):
    def map_table_infos(self, user_defined_table_infos):
        return self._map_table_infos(user_defined_table_infos, self.assign_groups)


# merge slot by user's assign_groups
class AutoMergeTableMapPolicy(BaseTableMapPolicy):
    def map_table_infos(self, user_defined_table_infos):
        assign_groups_flat = reduce(lambda a, b: a+b, self.assign_groups, [])
        new_assign_groups = []
        for sid, table_info in enumerate(user_defined_table_infos):
            if sid in assign_groups_flat:
                continue
            gid = -1
            if user_defined_table_infos[sid]['allow_merge']:
                for ngid, group in enumerate(new_assign_groups):
                    if self._is_equal_table_info(user_defined_table_infos[group[0]], table_info) \
                            and user_defined_table_infos[group[0]]['allow_merge']:
                        gid = ngid
                        break
            if gid == -1:
                gid = len(new_assign_groups)
                new_assign_groups.append([])
            new_assign_groups[gid].append(sid)
        new_assign_groups = self.assign_groups + new_assign_groups
        return self._map_table_infos(user_defined_table_infos, new_assign_groups)
