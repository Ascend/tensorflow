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

#include "tf_adapter/kernels/dp_iterator_ops.h"

#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"

namespace tensorflow {
namespace data {
namespace {
// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following ops.
const char kAnonymousIterator[] = "AnonymousIterator";
const char kAnonymousIteratorV2[] = "AnonymousIteratorV2";
const char kIteratorVariantTypeName[] = "tensorflow::Iterator";
const char kOutputShapes[] = "output_shapes";
const char kOutputTypes[] = "output_types";
}  // namespace

void DpMakeIteratorOp::Compute(OpKernelContext *ctx) {
  ADP_LOG(INFO) << "===Begin Computer MakeIterator===";
  CHECK_NOT_NULL(ctx);
  DatasetBase *dataset = nullptr;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  IteratorResource *iterator_resource = nullptr;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &iterator_resource));
  Status s = iterator_resource->SetIteratorFromDataset(ctx, dataset);
  iterator_resource->Unref();
  if (!s.ok()) { ctx->SetStatus(s); }
  ADP_LOG(INFO) << "===End Computer MakeIterator===";
}

namespace {

REGISTER_KERNEL_BUILDER(Name("MakeIterator").Device(DEVICE_CPU).Priority(2).Label("dp"), DpMakeIteratorOp);
REGISTER_KERNEL_BUILDER(Name("MakeIterator").Device(DEVICE_GPU).Priority(1).HostMemory("dataset").Label("dp"),
                        DpMakeIteratorOp);

}  // namespace

}  // namespace data
}  // namespace tensorflow
