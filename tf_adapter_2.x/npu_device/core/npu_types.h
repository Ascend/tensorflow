/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: Common depends and micro defines for and only for data preprocess module
 */

#ifndef TENSORFLOW_NPU_TYPES_H
#define TENSORFLOW_NPU_TYPES_H

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"

using TensorPartialShapes = tensorflow::gtl::InlinedVector<tensorflow::PartialTensorShape, 4>;
using TensorShapes = tensorflow::gtl::InlinedVector<tensorflow::TensorShape, 4>;
using TensorDataTypes = tensorflow::gtl::InlinedVector<tensorflow::DataType, 4>;

using VecTensorPartialShapes = tensorflow::gtl::InlinedVector<TensorPartialShapes, 4>;
using VecTensorShapes = tensorflow::gtl::InlinedVector<TensorShapes, 4>;
using VecTensorDataTypes = tensorflow::gtl::InlinedVector<TensorDataTypes, 4>;

const static tensorflow::TensorShape kScalarShape;

#endif  // TENSORFLOW_NPU_TYPES_H
