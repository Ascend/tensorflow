/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tf_adapter/util/session_manager.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/common/adapter_logger.h"

using namespace tensorflow;
/**
 * @brief: get instance
 */
SessionManager &SessionManager::GetInstance() {
  static SessionManager instance;
  return instance;
}

// Returns True if get ge session success.
bool SessionManager::GetOrCreateGeSession(const std::string &tf_session, ge::Session *&ge_session,
                                          std::map<std::string, std::string> &sess_options) {
  // find valid tf session handle
  if (tf_session.empty()) {
    ADP_LOG(ERROR) << "tf session is empty, get ge session failed.";
    LOG(ERROR) << "tf session is empty, get ge session failed.";
    return false;
  }

  // find valid ge session
  auto it = ge_sessions_.find(tf_session);
  if (it != ge_sessions_.end()) {
    ge_session = it->second;
    ADP_LOG(INFO) << "tf session " << tf_session << " get ge session success.";
    return true;
  }

  PrintGeSessionOptions(sess_options);
  bool ret = SessionManager::CreateGeSession(tf_session, ge_session, sess_options);
  if (!ret) {
    ADP_LOG(ERROR) << "tf session " << tf_session << " create ge session failed.";
    LOG(ERROR) << "tf session " << tf_session << " create ge session failed.";
    return false;
  }
  return true;
}

/**
 * @brief: destroy ge session.
 * @param tf_session: tf session
 */
void SessionManager::DestroyGeSession(const std::string &tf_session) {
  if (tf_session.empty()) {
    ADP_LOG(ERROR) << "tf session is empty, can not destroy ge session.";
    LOG(ERROR) << "tf session is empty, can not destroy ge session.";
  }
  auto it = ge_sessions_.find(tf_session);
  if (it != ge_sessions_.end()) {
    if (it->second != nullptr) {
      ADP_LOG(INFO) << "find ge session connect with tf session " << tf_session;
      delete it->second;
      it->second = nullptr;
    }
    (void)ge_sessions_.erase(it);
    ADP_LOG(INFO) << "destroy ge session connect with tf session " << tf_session << " success.";
  }
}

// Returns True if create ge session success.
bool SessionManager::CreateGeSession(const std::string &tf_session, ge::Session *&ge_session,
                                     std::map<std::string, std::string> &sess_options) {
  // hcom parallel
  ADP_LOG(INFO) << "[GEOP] hcom_parallel :" << sess_options[ge::HCOM_PARALLEL];

  // stream max parallel num
  ADP_LOG(INFO) << "[GEOP] stream_max_parallel_num :" << sess_options[ge::STREAM_MAX_PARALLEL_NUM];
  const auto sess_options_ascend_string = ChangeStringToAscendString(sess_options);
  ge_session = new (std::nothrow) ge::Session(sess_options_ascend_string);
  if (ge_session == nullptr) {
    ADP_LOG(ERROR) << "tf session " << tf_session << " create ge session failed.";
    LOG(ERROR) << "tf session " << tf_session << " create ge session failed.";
    return false;
  }
  (void)ge_sessions_.insert(std::make_pair(tf_session, ge_session));
  return true;
}

// Returns True if any ge session exist.
bool SessionManager::IsGeSessionExist() const {
  return !ge_sessions_.empty();
}

void SessionManager::PrintGeSessionOptions(std::map<std::string, std::string> &sess_options) const {
  // variable acceleration configuration
  ADP_LOG(INFO) << "[GEOP] variable_acceleration :" << sess_options["ge.exec.variable_acc"];
  // hcom parallel
  ADP_LOG(INFO) << "[GEOP] hcom_parallel :" << sess_options[ge::HCOM_PARALLEL];

  // stream max parallel num
  ADP_LOG(INFO) << "[GEOP] stream_max_parallel_num :" << sess_options[ge::STREAM_MAX_PARALLEL_NUM];
  // ac parallel enable
  ADP_LOG(INFO) << "[GEOP] ac_parallel_enable :" << sess_options[ge::AC_PARALLEL_ENABLE];

  // graph memory configuration
  if (!sess_options[ge::GRAPH_MEMORY_MAX_SIZE].empty()) {
    ADP_LOG(INFO) << "[GEOP] set graph_memory_max_size: " << sess_options[ge::GRAPH_MEMORY_MAX_SIZE];
  } else {
    (void)sess_options.erase(ge::GRAPH_MEMORY_MAX_SIZE);
  }

  // variable memory configuration
  if (!sess_options[ge::VARIABLE_MEMORY_MAX_SIZE].empty()) {
    ADP_LOG(INFO) << "[GEOP] set variable_memory_max_size: " << sess_options[ge::VARIABLE_MEMORY_MAX_SIZE];
  } else {
    (void)sess_options.erase(ge::VARIABLE_MEMORY_MAX_SIZE);
  }

  // tailing optimization
  ADP_LOG(INFO) << "[GEOP] is_tailing_optimization : " << sess_options["ge.exec.isTailingOptimization"];

  ADP_LOG(INFO) << "[GEOP] op_select_implmode : " << sess_options[ge::OP_SELECT_IMPL_MODE];

  ADP_LOG(INFO) << "[GEOP] optypelist_for_implmode : " << sess_options[ge::OPTYPELIST_FOR_IMPLMODE];

  // dump configuration
  string dump_step = sess_options[ge::OPTION_EXEC_DUMP_STEP];
  ADP_LOG(INFO) << "[GEOP] enable_dump :" << sess_options[ge::OPTION_EXEC_ENABLE_DUMP]
                << ", dump_path :" << sess_options[ge::OPTION_EXEC_DUMP_PATH]
                << ", dump_step :" << (dump_step.empty() ? "NA" : dump_step)
                << ", dump_mode :" << sess_options[ge::OPTION_EXEC_DUMP_MODE]
                << ", enable_dump_debug :" << sess_options[ge::OPTION_EXEC_ENABLE_DUMP_DEBUG]
                << ", dump_debug_mode :" << sess_options[ge::OPTION_EXEC_DUMP_DEBUG_MODE];

  // dynamic input config
  ADP_LOG(INFO) << "[GEOP] input_shape :" << sess_options["ge.inputShape"]
                << ", dynamic_dims :" << sess_options["ge.dynamicDims"]
                << ", dynamic_node_type :" << sess_options["ge.dynamicNodeType"];

  ADP_LOG(INFO) << "[GEOP] buffer_optimize :" << sess_options["ge.bufferOptimize"];

  ADP_LOG(INFO) << "[GEOP] enable_small_channel :" << sess_options["ge.enableSmallChannel"];

  ADP_LOG(INFO) << "[GEOP] fusion_switch_file :" << sess_options["ge.fusionSwitchFile"];

  ADP_LOG(INFO) << "[GEOP] enable_compress_weight :" << sess_options["ge.enableCompressWeight"];

  ADP_LOG(INFO) << "[GEOP] compress_weight_conf :" << sess_options["compress_weight_conf"];
}