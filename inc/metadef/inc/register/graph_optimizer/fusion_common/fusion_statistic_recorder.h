/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_FUSION_STATISTIC_RECORDER_H
#define INC_REGISTER_GRAPH_OPTIMIZER_FUSION_STATISTIC_RECORDER_H

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fe {

class FusionInfo {
 public:
#ifdef ONLY_COMPILE_OPEN_SRC
  explicit FusionInfo(uint64_t session_id = 0, std::string graph_id = "", std::string pass_name = "",
                      int32_t match_times = 0, int32_t effect_times = 0);

  virtual ~FusionInfo();

  void AddMatchTimes(int32_t match_times);

  void AddEffectTimes(int32_t effect_times);

  int32_t GetMatchTimes();

  int32_t GetEffectTimes();

  std::string GetGraphId();

  std::string GetPassName();

  uint64_t GetSessionId();

  void SetMatchTimes(int32_t match_times);

  void SetEffectTimes(int32_t effect_times);
#else
  explicit FusionInfo(const uint64_t session_id = 0, const std::string graph_id = "",
                      const std::string pass_name = "", const int32_t match_times = 0, const int32_t effect_times = 0);

  virtual ~FusionInfo();

  void AddMatchTimes(const int32_t match_times);

  void AddEffectTimes(const int32_t effect_times);

  int32_t GetMatchTimes() const;

  int32_t GetEffectTimes() const;

  std::string GetGraphId() const;

  std::string GetPassName() const;

  uint64_t GetSessionId() const;

  void SetMatchTimes(const int32_t match_times);

  void SetEffectTimes(const int32_t effect_times);
#endif

 private:
  uint64_t session_id_;
  std::string graph_id_;
  std::string pass_name_;
  int32_t match_times_;
  int32_t effect_times_;
};

using FusionStatisticMap = std::map<std::string, std::map<std::string, FusionInfo>>;

class FusionStatisticRecorder {
 public:
  FusionStatisticRecorder(const FusionStatisticRecorder &) = delete;

  FusionStatisticRecorder &operator=(const FusionStatisticRecorder &) = delete;

  static FusionStatisticRecorder &Instance();

#ifdef ONLY_COMPILE_OPEN_SRC
  void UpdateGraphFusionMatchTimes(FusionInfo &fusion_info);

  void UpdateGraphFusionEffectTimes(FusionInfo &fusion_info);

  void UpdateBufferFusionMatchTimes(FusionInfo &fusion_info);

  void UpdateBufferFusionEffectTimes(FusionInfo &fusion_info);
#else
  void UpdateGraphFusionMatchTimes(const FusionInfo &fusion_info);

  void UpdateGraphFusionEffectTimes(const FusionInfo &fusion_info);

  void UpdateBufferFusionMatchTimes(const FusionInfo &fusion_info);

  void UpdateBufferFusionEffectTimes(const FusionInfo &fusion_info);
#endif
  void GetAndClearFusionInfo(const std::string &session_graph_id,
                             std::map<std::string, FusionInfo> &graph_fusion_info_map,
                             std::map<std::string, FusionInfo> &buffer_fusion_info_map);

  void GetAllSessionAndGraphIdList(std::vector<std::string> &session_graph_id_vec);

 private:
  FusionStatisticRecorder();
  virtual ~FusionStatisticRecorder();
  FusionStatisticMap graph_fusion_info_map_;
  FusionStatisticMap buffer_fusion_info_map_;
  void GetFusionInfo(const std::string &session_graph_id, std::map<std::string, FusionInfo> &graph_fusion_info_map,
                     std::map<std::string, FusionInfo> &buffer_fusion_info_map);

#ifdef ONLY_COMPILE_OPEN_SRC
  void ClearFusionInfo(std::string session_graph_id);
#else
  void ClearFusionInfo(const std::string& session_graph_id);
#endif
  std::recursive_mutex mutex_;
};
}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_FUSION_STATISTIC_RECORDER_H
