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

#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "acl/acl.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/acl_channel.h"
#include "tf_adapter/util/npu_attrs.h"

namespace tensorflow {
namespace {
class OutfeedEnqueueOp : public OpKernel {
 public:
  explicit OutfeedEnqueueOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("channel_name", &channel_name_));
    ADP_LOG(INFO) << "OutfeedEnqueueOp built";
  }
  ~OutfeedEnqueueOp() override {
    ADP_LOG(INFO) << "OutfeedEnqueueOp has been destructed";
  }
  void Compute(OpKernelContext *ctx) override {
    (void) ctx;
    ADP_LOG(INFO) << "OutfeedEnqueueOp running";
  }
  bool IsExpensive() override {
    return false;
  }

 private:
  std::string channel_name_;
};

class OutfeedDequeueOp : public OpKernel {
 public:
  explicit OutfeedDequeueOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("channel_name", &channel_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    // Create log summary acl channel
    ADP_LOG(INFO) << "Start create acl channel for out-feed dequeue op " << channel_name_;
    uint32_t device_id = 0;
    OP_REQUIRES_OK(ctx, GetEnvDeviceID(device_id));
    device_id_ = device_id;
    const size_t kDefaultCapacity = 3;
    OP_REQUIRES(ctx, aclrtSetDevice(static_cast<int32_t>(device_id_)) == ACL_SUCCESS, errors::Internal("Acl rtSetDevice failed."));
    acl_handle_ = CreateAclTdtRecvChannel(device_id_, channel_name_, kDefaultCapacity);
    OP_REQUIRES(ctx, acl_handle_ != nullptr, errors::Internal("Acl create receive channel failed."));
    ADP_LOG(INFO) << "Succeed create acl channel for out-feed dequeue op " << channel_name_;
  }
  ~OutfeedDequeueOp() override {
    ADP_LOG(INFO) << "Start destroy acl channel for out-feed dequeue op " << channel_name_;
    if (acl_handle_ != nullptr) {
      if (acltdtDestroyChannel(acl_handle_) != ACL_ERROR_NONE) {
        ADP_LOG(ERROR) << "Failed destroy acl channel for out-feed dequeue op " << channel_name_;
      } else {
        acl_handle_ = nullptr;
        ADP_LOG(INFO) << "Succeed destroy acl channel for out-feed dequeue op " << channel_name_;
      }
    }
    if (aclrtResetDevice(static_cast<int32_t>(device_id_)) != ACL_SUCCESS) {
      ADP_LOG(ERROR) << "Acl rtResetDevice failed.";
    }
  }
  void Compute(OpKernelContext *ctx) override {
    ADP_LOG(INFO) << "Start compute out-feed dequeue op " << channel_name_;
    CancellationManager *cm = ctx->cancellation_manager();
    CancellationToken token = cm->get_cancellation_token();
    bool already_cancelled = !cm->RegisterCallback(token, [this]() {
      ADP_LOG(INFO) << "Start run cancellation callback of out-feed dequeue op " << channel_name_;
      Status ret = StopRecvTensorByAcl(&acl_handle_, channel_name_);
      if (!ret.ok()) {
        ADP_LOG(ERROR) << ret.error_message();
      }
    });
    if (TF_PREDICT_FALSE(already_cancelled)) {
      ctx->SetStatus(errors::Internal("out-feed op ", channel_name_, " called after cancelled."));
      return;
    }
    std::vector<Tensor> tensors;
    ADP_LOG(INFO) << "Start recv tensors by acl out-feed dequeue op " << channel_name_;
    auto status = RecvTensorByAcl(acl_handle_, tensors);
    ADP_LOG(INFO) << "Start de-register callback out-feed dequeue op " << channel_name_;
    (void) cm->DeregisterCallback(token);
    OP_REQUIRES_OK(ctx, status);
    OP_REQUIRES(ctx, !tensors.empty(), errors::OutOfRange("out-feed op ", channel_name_, " received end-of-sequence"));
    OP_REQUIRES(ctx, tensors.size() == output_shapes_.size(),
                errors::Internal("out-feed op ", channel_name_, " received ", tensors.size(), " tensors but expect ",
                                 output_shapes_.size(), " tensors"));
    for (int i = 0; i < ctx->num_outputs(); ++i) {
      ctx->set_output(i, tensors[static_cast<size_t>(i)]);
    }
  }
  bool IsExpensive() override {
    return false;
  }

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::string channel_name_;
  acltdtChannelHandle *acl_handle_ = nullptr;
  uint32_t device_id_ = 0;
};

REGISTER_KERNEL_BUILDER(Name("OutfeedDequeueOp").Device(DEVICE_CPU), OutfeedDequeueOp);

REGISTER_KERNEL_BUILDER(Name("OutfeedEnqueueOp").Device(DEVICE_CPU), OutfeedEnqueueOp);
}  // namespace
}  // namespace tensorflow
