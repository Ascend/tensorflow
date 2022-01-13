/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
REGISTER_OP("SendH2D")
  .Input("inputs: Tin")
  .Attr("channel_name: string")
  .Attr("device_ids: list(int)")
  .Attr(
    "Tin: list(type) = [DT_FLOAT, DT_HALF, DT_INT8, DT_INT32, DT_UINT8, DT_INT16, DT_UINT16, DT_UINT32, "
    "DT_INT64, DT_UINT64, DT_DOUBLE, DT_BOOL, DT_STRING]")
  .SetIsStateful();

REGISTER_OP("IteratorH2D")
  .Input("input: resource")
  .Input("nums: int64")
  .Attr("channel_name: string")
  .Attr("device_ids: list(int)")
  .SetIsStateful();
}  // namespace tensorflow