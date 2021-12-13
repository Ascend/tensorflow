/**
 * @file tune_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 * 描述：mstune调优接口头文件
 */
/** @defgroup mstune mstune调优接口 */
#ifndef TUNE_API_H
#define TUNE_API_H
#include <map>
#include <string>
#include "ge/ge_api.h"
#include "aoe_types.h"

/**
 * @ingroup aoe
 * @par 描述: 命令行调优
 *
 * @attention 无
 * @param  option [IN] 调优参数
 * @param  msg [OUT] 调优异常下返回信息
 * @retval #MSTUNE_SUCCESS 执行成功
 * @retval #MSTUNE_FAILED 执行失败
 * @par 依赖:
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
*/
AoeStatus AoeOfflineTuning(const std::map<std::string, std::string> &option, std::string &msg);

/**
 * @ingroup aoe
 * @par 描述: 梯度调优
 *
 * @attention 无
 * @param  tuningGraph [IN] 调优图
 * @param  dependGraph [IN] 调优依赖图
 * @param  session [IN] ge连接会话
 * @param  option [IN] 参数集. 包含调优参数及ge参数
 * @retval #MSTUNE_SUCCESS 执行成功
 * @retval #MSTUNE_FAILED 执行失败
 * @par 依赖:
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" MsTuneStatus MsTrainTuning(ge::Graph &tuningGraph, std::vector<ge::Graph> &dependGraph,
    ge::Session *session, const std::map<std::string, std::map<std::string, std::string>> &option);

/**
 * @ingroup aoe
 * @par 描述: 调优初始化
 * @attention 无
 * @param  option [IN] 参数集. 包含调优参数及ge参数
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILED 执行失败
 * @par 依赖:无
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeInitialize(const std::map<ge::AscendString, ge::AscendString> &globalOptions);

/**
 * @ingroup aoe
 * @par 描述: 调优去初始化
 *
 * @attention 无
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILED 执行失败
 * @par 依赖:无
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeFinalize();

/**
 * @ingroup aoe
 * @par 描述: 调优初始化
 * @attention 无
 * @param  sessionOptions [IN] 参数集. 包含调优参数及ge参数
 * @param  SessionId [IN] ge链接会话
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILED 执行失败
 * @par 依赖:无
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeCreateSession(const std::map<ge::AscendString, ge::AscendString> &sessionOptions,
                                      SessionId &SessionId);

/**
 * @ingroup aoe
 * @par 描述: 调优初始化
 * @attention 无
 * @param  SessionId [IN] ge链接会话
 * @param  geSession [IN] ge链接会话
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILED 执行失败
 * @par 依赖:无
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeSetGeSession(SessionId SessionId, ge::Session* geSession);

/**
 * @ingroup aoe
 * @par 描述: 调优初始化
 * @attention 无
 * @param  SessionId [IN] ge链接会话
 * @param  dependGraph [IN] 调优依赖图
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILED 执行失败
 * @par 依赖:无
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeSetDependGraphs(SessionId SessionId, std::vector<ge::Graph> &dependGraph);

/**
 * @ingroup aoe
 * @par 描述: 调优初始化
 * @attention 无
 * @param  SessionId [IN] ge链接会话
 * @param  tuningGraph [IN] 调优图
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILED 执行失败
 * @par 依赖:无
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeSetTuningGraph(SessionId SessionId, ge::Graph &tuningGraph);

/**
 * @ingroup aoe
 * @par 描述: 调优初始化
 * @attention 无
 * @param  SessionId [IN] ge链接会话
 * @param  tuningOptions [IN] ge参数
 * @retval #AOE_SUCCESS 执行成功
 * @retval #AOE_FAILED 执行失败
 * @par 依赖:无
 * @li tune_api.cpp：该接口所属的开发包。
 * @li tune_api.h：该接口声明所在的头文件。
 * @see 无
 * @since
 */
extern "C" AoeStatus AoeTuningGraph(SessionId SessionId,
                                    const std::map<ge::AscendString, ge::AscendString> &tuningOptions);
#endif
