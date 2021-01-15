/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/util/cache_interface.h"

namespace tensorflow {
class EmbeddingRankIdOpKernel : public OpKernel {
 public:
  explicit EmbeddingRankIdOpKernel(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingRankIdOpKernel() {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingRankIdOp Compute."; }
};

class LruCacheOp : public ResourceOpKernel<CacheInterface> {
 public:
  explicit LruCacheOp(OpKernelConstruction* context) : ResourceOpKernel(context) {}
  void Compute(OpKernelContext* context) override { ADP_LOG(INFO) << "LruCacheOp Compute"; }
 private:
  Status CreateResource(CacheInterface** resource) override
                        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return Status::OK();
  }
};

class CacheAddOp : public OpKernel {
 public:
  explicit CacheAddOp(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "CacheAddOp Compute"; }
};

class CacheRemoteIndexToLocalOp : public OpKernel {
 public:
  explicit CacheRemoteIndexToLocalOp(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "CacheRemoteIndexToLocalOp Compute"; }
};

template <typename T>
class DeformableOffsetsOp : public OpKernel {
 public:
  explicit DeformableOffsetsOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DeformableOffsetsOp() override {}
  void Compute(OpKernelContext *context) override {
    ADP_LOG(INFO) << "DeformableOffsetsOp Compute, num_inputs: "
              << context->num_inputs();
  }
  bool IsExpensive() override { return false; }
};

template <typename T>
class DeformableOffsetsGradOp : public OpKernel {
 public:
  explicit DeformableOffsetsGradOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DeformableOffsetsGradOp() override {}
  void Compute(OpKernelContext *context) override {
    ADP_LOG(INFO) << "DeformableOffsetsGradOp Compute, num_inputs: "
              << context->num_inputs();
  }
  bool IsExpensive() override { return false; }
};

class RandomChoiceWithMaskOp : public OpKernel {
 public:
  explicit RandomChoiceWithMaskOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~RandomChoiceWithMaskOp() override {}
  void Compute(OpKernelContext *context) override {
    ADP_LOG(INFO) << "RandomChoiceWithMaskOp Compute ";
  }
};

template <typename T>
class DenseImageWarpOp : public OpKernel {
 public:
  explicit DenseImageWarpOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DenseImageWarpOp() override {}
  void Compute(OpKernelContext *context) override {}
  bool IsExpensive() override { return false; }
};

template <typename T>
class DenseImageWarpGradOp : public OpKernel {
 public:
  explicit DenseImageWarpGradOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DenseImageWarpGradOp() override {}
  void Compute(OpKernelContext *context) override {}
  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("EmbeddingRankId").Device(DEVICE_CPU), EmbeddingRankIdOpKernel);
REGISTER_KERNEL_BUILDER(Name("LruCache").Device(DEVICE_CPU), LruCacheOp);
REGISTER_KERNEL_BUILDER(Name("CacheAdd").Device(DEVICE_CPU), CacheAddOp);
REGISTER_KERNEL_BUILDER(Name("CacheRemoteIndexToLocal").Device(DEVICE_CPU), CacheRemoteIndexToLocalOp);
REGISTER_KERNEL_BUILDER(Name("RandomChoiceWithMask").Device(DEVICE_CPU), RandomChoiceWithMaskOp);

#define REGISTER_KERNEL(type)                                \
REGISTER_KERNEL_BUILDER(Name("DeformableOffsets")            \
                            .Device(DEVICE_CPU)              \
                            .TypeConstraint<type>("T"),      \
                        DeformableOffsetsOp<type>)
REGISTER_KERNEL(float);
REGISTER_KERNEL(Eigen::half);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(type)                                \
REGISTER_KERNEL_BUILDER(Name("DeformableOffsetsGrad")        \
                            .Device(DEVICE_CPU)              \
                            .TypeConstraint<type>("T"),      \
                        DeformableOffsetsGradOp<type>)
REGISTER_KERNEL(float);
REGISTER_KERNEL(Eigen::half);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(type)                                \
REGISTER_KERNEL_BUILDER(Name("DenseImageWarp")               \
                            .Device(DEVICE_CPU)              \
                            .TypeConstraint<type>("T"),      \
                        DenseImageWarpOp<type>)
REGISTER_KERNEL(float);
REGISTER_KERNEL(Eigen::half);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(type)                                \
REGISTER_KERNEL_BUILDER(Name("DenseImageWarpGrad")           \
                            .Device(DEVICE_CPU)              \
                            .TypeConstraint<type>("T"),      \
                        DenseImageWarpGradOp<type>)
REGISTER_KERNEL(float);
REGISTER_KERNEL(Eigen::half);
#undef REGISTER_KERNEL
}  // namespace tensorflow