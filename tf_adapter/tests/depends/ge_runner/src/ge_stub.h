/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef COMMON_GRAPH_DEBUG_GE_UTIL_H_
#define COMMON_GRAPH_DEBUG_GE_UTIL_H_

#include <iostream>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/memory/memory_api.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_api.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "ge/ge_ir_build.h"
#include "graph/tensor.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_adapter.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "graph/buffer.h"
#include "graph/model.h"

namespace ge {

using RunGraphWithStreamAsyncStub = std::function<Status(uint32_t, void *, const std::vector<Tensor>&, std::vector<Tensor>&)>;
void RegRunGraphWithStreamAsyncStub(RunGraphWithStreamAsyncStub stub);

using RunGraphStub = std::function<Status(uint32_t, const std::vector<Tensor>&, std::vector<Tensor>&)>;
void RegRunGraphStub(RunGraphStub stub);

using RunGraphAsyncStub = std::function<Status(uint32_t, const std::vector<Tensor>&, RunAsyncCallback)>;
void RegRunGraphAsyncStub(RunGraphAsyncStub stub);
}  // namespace ge
#endif  // COMMON_GRAPH_DEBUG_GE_UTIL_H_
