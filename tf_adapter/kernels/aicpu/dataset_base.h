/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#ifndef TENSORFLOW_CORE_KERNELS_DATASET_BASE_H_
#define TENSORFLOW_CORE_KERNELS_DATASET_BASE_H_
#include <utility>
#include <algorithm>
#include "tensorflow/core/framework/cancellation.h"

namespace tensorflow {
namespace data {
#ifndef TF_VERSION_TF2
Status RegisterCancellationCallback(CancellationManager* cancellation_manager,
                                    std::function<void()> callback,
                                    std::function<void()>* deregister_fn) {
  if (cancellation_manager) {
    CancellationToken token = cancellation_manager->get_cancellation_token();
    if (!cancellation_manager->RegisterCallback(token, std::move(callback))) {
      return errors::Cancelled("Operation was cancelled");
    }
    *deregister_fn = [cancellation_manager, token]() {
      cancellation_manager->DeregisterCallback(token);
    };
  } else {
    VLOG(1) << "Cancellation manager is not set. Cancellation callback will "
               "not be registered.";
    *deregister_fn = []() {};
  }
  return Status::OK();
}
#endif
}  // namespace data
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_DATASET_FUNCTION_H_