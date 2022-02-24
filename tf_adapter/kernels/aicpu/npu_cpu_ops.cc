/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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
  ~EmbeddingRankIdOpKernel() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingRankIdOp Compute."; }
};

class EmbeddingLocalIndexOpKernel : public OpKernel {
 public:
  explicit EmbeddingLocalIndexOpKernel(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingLocalIndexOpKernel() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingLocalIndexOp Compute."; }
};

class LruCacheOp : public ResourceOpKernel<CacheInterface> {
 public:
  explicit LruCacheOp(OpKernelConstruction* context) : ResourceOpKernel(context) {}
  ~LruCacheOp() override {}
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
  ~CacheAddOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "CacheAddOp Compute"; }
};

class CacheRemoteIndexToLocalOp : public OpKernel {
 public:
  explicit CacheRemoteIndexToLocalOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~CacheRemoteIndexToLocalOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "CacheRemoteIndexToLocalOp Compute"; }
};

class CacheAllIndexToLocalOp : public OpKernel {
 public:
  explicit CacheAllIndexToLocalOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~CacheAllIndexToLocalOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "CacheAllIndexToLocalOp Compute"; }
};

template <typename T>
class DeformableOffsetsOp : public OpKernel {
 public:
  explicit DeformableOffsetsOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DeformableOffsetsOp() override {}
  void Compute(OpKernelContext *context) override {
    ADP_LOG(INFO) << "DeformableOffsetsOp Compute, num_inputs: " << context->num_inputs();
  }
  bool IsExpensive() override { return false; }
};

template <typename T>
class DeformableOffsetsGradOp : public OpKernel {
 public:
  explicit DeformableOffsetsGradOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DeformableOffsetsGradOp() override {}
  void Compute(OpKernelContext *context) override {
    ADP_LOG(INFO) << "DeformableOffsetsGradOp Compute, num_inputs: " << context->num_inputs();
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

class BatchEnqueueOp : public OpKernel {
 public:
  explicit BatchEnqueueOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~BatchEnqueueOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "BatchEnqueueOp Compute"; }
};

class OCRRecognitionPreHandleOp : public OpKernel {
 public:
  explicit OCRRecognitionPreHandleOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~OCRRecognitionPreHandleOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "OCRRecognitionPreHandleOp Compute"; }
};

class OCRDetectionPreHandleOp : public OpKernel {
 public:
  explicit OCRDetectionPreHandleOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~OCRDetectionPreHandleOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "OCRDetectionPreHandleOp Compute"; }
};

class OCRIdentifyPreHandleOp : public OpKernel {
 public:
  explicit OCRIdentifyPreHandleOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~OCRIdentifyPreHandleOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "OCRIdentifyPreHandleOp Compute"; }
};

class BatchDilatePolysOp : public OpKernel {
  public:
  explicit BatchDilatePolysOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~BatchDilatePolysOp() override {}
  void Compute(OpKernelContext *context) override{ADP_LOG(INFO)<<"BatchDilatePolysOp Compute";}
};

class OCRFindContoursOp : public OpKernel {
  public:
  explicit OCRFindContoursOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~OCRFindContoursOp() override {}
  void Compute(OpKernelContext *context) override{ADP_LOG(INFO)<<"OCRFindContoursOp Compute";}
};

class OCRDetectionPostHandleOp : public OpKernel {
 public:
  explicit OCRDetectionPostHandleOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~OCRDetectionPostHandleOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "OCRDetectionPostHandleOp Compute"; }
};

class ResizeAndClipPolysOp : public OpKernel {
 public:
  explicit ResizeAndClipPolysOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~ResizeAndClipPolysOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "ResizeAndClipPolysOp Compute"; }
};

class DequeueOp : public OpKernel {
 public:
  explicit DequeueOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~DequeueOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "DequeueOp Compute"; }
};

class NonZeroWithValueShapeOp : public OpKernel {
 public:
  explicit NonZeroWithValueShapeOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~NonZeroWithValueShapeOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "NonZeroWithValueShapeOp Compute"; }
};

REGISTER_KERNEL_BUILDER(Name("EmbeddingRankId").Device(DEVICE_CPU), EmbeddingRankIdOpKernel);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLocalIndex").Device(DEVICE_CPU), EmbeddingLocalIndexOpKernel);
REGISTER_KERNEL_BUILDER(Name("LruCache").Device(DEVICE_CPU), LruCacheOp);
REGISTER_KERNEL_BUILDER(Name("CacheAdd").Device(DEVICE_CPU), CacheAddOp);
REGISTER_KERNEL_BUILDER(Name("CacheRemoteIndexToLocal").Device(DEVICE_CPU), CacheRemoteIndexToLocalOp);
REGISTER_KERNEL_BUILDER(Name("CacheAllIndexToLocal").Device(DEVICE_CPU), CacheAllIndexToLocalOp);
REGISTER_KERNEL_BUILDER(Name("RandomChoiceWithMask").Device(DEVICE_CPU), RandomChoiceWithMaskOp);
REGISTER_KERNEL_BUILDER(Name("BatchEnqueue").Device(DEVICE_CPU), BatchEnqueueOp);
REGISTER_KERNEL_BUILDER(Name("OCRRecognitionPreHandle").Device(DEVICE_CPU), OCRRecognitionPreHandleOp);
REGISTER_KERNEL_BUILDER(Name("OCRDetectionPreHandle").Device(DEVICE_CPU), OCRDetectionPreHandleOp);
REGISTER_KERNEL_BUILDER(Name("OCRIdentifyPreHandle").Device(DEVICE_CPU), OCRIdentifyPreHandleOp);
REGISTER_KERNEL_BUILDER(Name("BatchDilatePolys").Device(DEVICE_CPU), BatchDilatePolysOp);
REGISTER_KERNEL_BUILDER(Name("OCRFindContours").Device(DEVICE_CPU), OCRFindContoursOp);
REGISTER_KERNEL_BUILDER(Name("OCRDetectionPostHandle").Device(DEVICE_CPU), OCRDetectionPostHandleOp);
REGISTER_KERNEL_BUILDER(Name("ResizeAndClipPolys").Device(DEVICE_CPU), ResizeAndClipPolysOp);
REGISTER_KERNEL_BUILDER(Name("Dequeue").Device(DEVICE_CPU), DequeueOp);
REGISTER_KERNEL_BUILDER(Name("NonZeroWithValueShape").Device(DEVICE_CPU), NonZeroWithValueShapeOp);

class DecodeImageV3Op : public OpKernel {
public:
  explicit DecodeImageV3Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~DecodeImageV3Op() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "DecodeImageV3Op Compute"; }
};
REGISTER_KERNEL_BUILDER(Name("DecodeImageV3").Device(DEVICE_CPU), DecodeImageV3Op);

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