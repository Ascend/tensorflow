/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef TENSORFLOW_KERNELS_OM_EXECUTOR_H_
#define TENSORFLOW_KERNELS_OM_EXECUTOR_H_
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
class OmExecutor {
 public:
  /// Create a executor for om located at 'om_path'
  /// @param om_path Absolute file path of the om file
  /// @param executor Created om executor
  /// @return Status::OK() or error status if any error occurs
  static Status Create(const std::string &om_path, std::unique_ptr<OmExecutor> &executor);

  /// Calc om result with feed inputs
  /// @param inputs Tensorflow host input tensors
  /// @param outputs Empty output tensors to be filling
  /// @return Status::OK() or error status if any error occurs
  Status Execute(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs);

 private:
  explicit OmExecutor(const std::string &om_path);
};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_OM_EXECUTOR_H_
