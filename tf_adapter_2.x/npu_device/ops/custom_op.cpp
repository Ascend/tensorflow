/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: Common depends and micro defines for and only for data preprocess module
 */

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_experimental.h"
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