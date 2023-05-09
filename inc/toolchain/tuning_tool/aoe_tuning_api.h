/**
 * @file aoe_tuning_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 * 描述：mstune调优接口头文件
 */

#ifndef AOE_TUNING_API_H
#define AOE_TUNING_API_H
#include <map>
#include <string>
#include "ge/ge_api.h"
#include "aoe_types.h"
#include "graph/ascend_string.h"

namespace Aoe {
using SessionId = uint64_t;
using AoeStatus = int32_t;
/**
 * @brief      : initialize aoe tuning api
 * @param  [in] : map<AscendString, AscendString> &globalOptions
 * @return      : == AOE_SUCESS : sucess,!= AOE_SUCESS : failed
 */
extern "C" AoeStatus AoeInitialize(const std::map<ge::AscendString, ge::AscendString> &globalOptions);

/**
 * @brief      : fialize aoe tuning api
 * @return      : == AOE_SUCESS : sucess,!= AOE_SUCESS : failed
 */
extern "C" AoeStatus AoeFinalize();

/**
 * @brief      : destroy aoe session
 * @param  [out] : SessionId SessionId                     session id
 * @return      : == AOE_SUCESS : sucess,!= AOE_SUCESS : failed
 */
extern "C" AoeStatus AoeDestroySession(SessionId SessionId);

/**
 * @brief      : create aoe session
 * @param  [in] : map<AscendString, AscendString> &sessionOptions      session options
 * @param  [out] : SessionId SessionId                     session id
 * @return      : == AOE_SUCESS : sucess,!= AOE_SUCESS : failed
 */
extern "C" AoeStatus AoeCreateSession(SessionId &SessionId);

/**
 * @brief      : set ge session for session id
 * @param  [in] : SessionId SessionId      session id
 * @param  [in] : ge::Session* geSession   ge session handle
 * @return      : == AOE_SUCESS : sucess,!= AOE_SUCESS : failed
 */
extern "C" AoeStatus AoeSetGeSession(SessionId SessionId, ge::Session* geSession);

/**
 * @brief      : set depend graphs for session id
 * @param  [in] : SessionId SessionId      session id
* @param  [in] : std::vector<ge::Graph> &dependGraph     depend graphs
 * @return      : == AOE_SUCESS : sucess,!= AOE_SUCESS : failed
 */
extern "C" AoeStatus AoeSetDependGraphs(SessionId SessionId, std::vector<ge::Graph> &dependGraph);

/**
 * @brief      : set tuning graphs for session id
 * @param  [in] : SessionId SessionId      session id
* @param  [in] : ge::Graph &tuningGraph    tuning graph
 * @return      : == AOE_SUCESS : sucess,!= AOE_SUCESS : failed
 */
extern "C" AoeStatus AoeSetTuningGraph(SessionId SessionId, ge::Graph &tuningGraph);

/**
 * @brief      : tuning graph
 * @param  [in] : SessionId SessionId      session id
* @param  [in] : map<AscendString, AscendString> &tuningOptions   tuning options
 * @return      : == AOE_SUCESS : sucess,!= AOE_SUCESS : failed
 */
extern "C" AoeStatus AoeTuningGraph(SessionId SessionId,
                                    const std::map<ge::AscendString, ge::AscendString> &tuningOptions);
} // namespace Aoe
#endif
