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

#ifndef INC_GRAPH_COMMON_ERROR_CODES_H_
#define INC_GRAPH_COMMON_ERROR_CODES_H_

#include "external/graph/ge_error_codes.h"

namespace ge {
const graphStatus NO_DEPENDENCE_FUNC = 50331647;
const graphStatus NO_OVERLAP_DIM = 50331646;
const graphStatus NOT_SUPPORT_SLICE = 50331645;
}  // namespace ge

#endif  // INC_GRAPH_COMMON_ERROR_CODES_H_
