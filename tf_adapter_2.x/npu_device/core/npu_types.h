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

#ifndef NPU_DEVICE_CORE_NPU_TYPES_H
#define NPU_DEVICE_CORE_NPU_TYPES_H

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"

namespace npu {
using TensorPartialShapes = tensorflow::gtl::InlinedVector<tensorflow::PartialTensorShape, 4>;
using TensorShapes = tensorflow::gtl::InlinedVector<tensorflow::TensorShape, 4>;
using TensorDataTypes = tensorflow::gtl::InlinedVector<tensorflow::DataType, 4>;

using VecTensorPartialShapes = tensorflow::gtl::InlinedVector<TensorPartialShapes, 4>;
using VecTensorShapes = tensorflow::gtl::InlinedVector<TensorShapes, 4>;
using VecTensorDataTypes = tensorflow::gtl::InlinedVector<TensorDataTypes, 4>;

const static tensorflow::TensorShape kScalarShape;

const static uint64_t kInvalidGeGraphId = -1;
const static uint64_t kEmptyGeGraphId = -2;

class ResourceGenerator {
 public:
  ResourceGenerator(std::shared_ptr<tensorflow::NodeDef> def, int index) : def_(def), index_(index) {}
  std::shared_ptr<tensorflow::NodeDef> NodeDef() const { return def_; }
  int Index() const { return index_; }

 private:
  std::shared_ptr<tensorflow::NodeDef> def_;
  int index_;
};

enum class CacheStrategy { DEFAULT, BY_OP_NAME };
}  // namespace npu
#endif  // NPU_DEVICE_CORE_NPU_TYPES_H
