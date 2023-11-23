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

#include "tf_adapter/common/adapter_logger.h"
#include "toolchain/slog.h"
#include "toolchain/plog.h"

namespace npu {
AdapterLogger::~AdapterLogger() {
  int32_t modeule = FMK_MODULE_NAME;
  if (severity_ == ADP_RUN_INFO) {
    modeule = static_cast<int32_t>(static_cast<uint32_t>(RUN_LOG_MASK) | static_cast<uint32_t>(FMK_MODULE_NAME));
  }
  if (severity_ == ADP_FATAL) {
    DlogSubForC(modeule, ADP_MODULE_NAME, ADP_ERROR, "%s", str().c_str());
    (void) DlogReportFinalize();
  } else {
    DlogSubForC(modeule, ADP_MODULE_NAME, severity_, "%s", str().c_str());
  }
}
}  // namespace npu
