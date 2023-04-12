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
#include "tf_adapter/common/adapter_logger.h"
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
  Status CreateResource(CacheInterface** resource) override {
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
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "BatchDilatePolysOp Compute"; }
};

class OCRFindContoursOp : public OpKernel {
  public:
  explicit OCRFindContoursOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~OCRFindContoursOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "OCRFindContoursOp Compute"; }
};

class OCRDetectionPostHandleOp : public OpKernel {
 public:
  explicit OCRDetectionPostHandleOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~OCRDetectionPostHandleOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "OCRDetectionPostHandleOp Compute"; }
};

class WarpAffineV2Op : public OpKernel {
 public:
  explicit WarpAffineV2Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~WarpAffineV2Op() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "WarpAffineV2Op Compute"; }
};

class ResizeV2Op : public OpKernel {
 public:
  explicit ResizeV2Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~ResizeV2Op() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "ResizeV2Op Compute"; }
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

class ScatterElementsV2Op : public OpKernel {
 public:
  explicit ScatterElementsV2Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~ScatterElementsV2Op() override = default;
  void Compute(OpKernelContext *context) override {
    (void) (context);
    ADP_LOG(INFO) << "in ScatterElementsV2";
  }
  bool IsExpensive() override {
    ADP_LOG(INFO) << "in ScatterElementsV2 IsExpensive";
    return false;
  }
};

class InitPartitionMapOp : public OpKernel {
public:
  explicit InitPartitionMapOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~InitPartitionMapOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "InitPartitionMapOp Compute"; }
};

class InitEmbeddingHashmapOp : public OpKernel {
public:
  explicit InitEmbeddingHashmapOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~InitEmbeddingHashmapOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "InitEmbeddingHashmapOp Compute"; }
};

class EmbeddingTableFindOp : public OpKernel {
public:
  explicit EmbeddingTableFindOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingTableFindOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingTableFindOp Compute"; }
};

class EmbeddingTableImportOp : public OpKernel {
public:
  explicit EmbeddingTableImportOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingTableImportOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingTableImportOp Compute"; }
};

class UninitPartitionMapOp : public OpKernel {
public:
  explicit UninitPartitionMapOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~UninitPartitionMapOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "UninitPartitionMapOp Compute"; }
};

class UninitEmbeddingHashmapOp : public OpKernel {
public:
  explicit UninitEmbeddingHashmapOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~UninitEmbeddingHashmapOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "UninitEmbeddingHashmapOp Compute"; }
};

class TableToResourceOp : public OpKernel {
public:
  explicit TableToResourceOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~TableToResourceOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "TableToResourceOp Compute"; }
};

class EmbeddingTableFindAndInitOp : public OpKernel {
public:
  explicit EmbeddingTableFindAndInitOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingTableFindAndInitOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingTableFindAndInitOp Compute"; }
};

class EmbeddingApplyAdamOp : public OpKernel {
public:
  explicit EmbeddingApplyAdamOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingApplyAdamOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingApplyAdamOp Compute"; }
};

class EmbeddingApplyAdamWOp : public OpKernel {
public:
  explicit EmbeddingApplyAdamWOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingApplyAdamWOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingApplyAdamWOp Compute"; }
};

class EmbeddingApplyAdaGradOp : public OpKernel {
public:
  explicit EmbeddingApplyAdaGradOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingApplyAdaGradOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingApplyAdaGradOp Compute"; }
};

class EmbeddingTableExportOp : public OpKernel {
public:
  explicit EmbeddingTableExportOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingTableExportOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingTableExportOp Compute"; }
};

class EmbeddingFeatureMappingOp : public OpKernel {
public:
  explicit EmbeddingFeatureMappingOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~EmbeddingFeatureMappingOp() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "EmbeddingFeatureMappingOp Compute"; }
};

REGISTER_KERNEL_BUILDER(Name("ScatterElementsV2").Device(DEVICE_CPU), ScatterElementsV2Op);
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
REGISTER_KERNEL_BUILDER(Name("InitPartitionMap").Device(DEVICE_CPU), InitPartitionMapOp);
REGISTER_KERNEL_BUILDER(Name("InitEmbeddingHashmap").Device(DEVICE_CPU), InitEmbeddingHashmapOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingTableFind").Device(DEVICE_CPU), EmbeddingTableFindOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingTableImport").Device(DEVICE_CPU), EmbeddingTableImportOp);
REGISTER_KERNEL_BUILDER(Name("UninitPartitionMap").Device(DEVICE_CPU), UninitPartitionMapOp);
REGISTER_KERNEL_BUILDER(Name("UninitEmbeddingHashmap").Device(DEVICE_CPU), UninitEmbeddingHashmapOp);
REGISTER_KERNEL_BUILDER(Name("TableToResource").Device(DEVICE_CPU), TableToResourceOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingTableFindAndInit").Device(DEVICE_CPU), EmbeddingTableFindAndInitOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingApplyAdam").Device(DEVICE_CPU), EmbeddingApplyAdamOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingApplyAdamW").Device(DEVICE_CPU), EmbeddingApplyAdamWOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingApplyAdaGrad").Device(DEVICE_CPU), EmbeddingApplyAdaGradOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingTableExport").Device(DEVICE_CPU), EmbeddingTableExportOp);
REGISTER_KERNEL_BUILDER(Name("EmbeddingFeatureMapping").Device(DEVICE_CPU), EmbeddingFeatureMappingOp);
REGISTER_KERNEL_BUILDER(Name("WarpAffineV2").Device(DEVICE_CPU), WarpAffineV2Op);
REGISTER_KERNEL_BUILDER(Name("ResizeV2").Device(DEVICE_CPU), ResizeV2Op);

class DecodeImageV3Op : public OpKernel {
public:
  explicit DecodeImageV3Op(OpKernelConstruction *context) : OpKernel(context) {}
  ~DecodeImageV3Op() override {}
  void Compute(OpKernelContext *context) override { ADP_LOG(INFO) << "DecodeImageV3Op Compute"; }
};
// Since the DecodeImage is registed on 2.x,
// in order to ensure that there si no conflict between operators on 1.x and 2.x,
// it is registered as the DecodeImageV3
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