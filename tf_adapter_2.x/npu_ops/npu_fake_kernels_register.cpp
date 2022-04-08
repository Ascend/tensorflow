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

namespace tensorflow {
namespace {
class FakeOp : public AsyncOpKernel {
 public:
  explicit FakeOp(OpKernelConstruction *context) : AsyncOpKernel(context) {}
  ~FakeOp() override = default;

  void ComputeAsync(OpKernelContext *context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(
      context, errors::Internal(context->op_kernel().name(), " registered as fake op and should never run on cpu"),
      done);
  }
};
}  // namespace
}  // namespace tensorflow