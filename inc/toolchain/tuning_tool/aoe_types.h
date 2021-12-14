/**
 * @file aoe_types.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 * 描述：mstune调优接口头文件
 */
/** @defgroup mstune mstune调优接口 */
#ifndef AOE_TYPES_H
#define AOE_TYPES_H

#include <vector>
#include <memory>
#include "graph/graph.h"

using AoeStatus = int32_t;
const AoeStatus AOE_SUCCESS = 0;
const AoeStatus AOE_FALLURE = -1;
const AoeStatus AOE_ERROR_NO_AICORE_GRAPH = 12;

/**
 * @ingroup mstune
 *
 * mstune status
 */
enum MsTuneStatus {
    MSTUNE_SUCCESS,  /** tune success */
    MSTUNE_FAILED,   /** tune failed */
};

// Option key: for train options sets
const std::string MSTUNE_SELF_KEY = "mstune";
const std::string MSTUNE_GEINIT_KEY = "initialize";
const std::string MSTUNE_GESESS_KEY = "session";

#endif