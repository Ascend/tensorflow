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
#include <thread>
#include <dlfcn.h>
#include <vector>
#include "unistd.h"
#include "acl/acl_tdt.h"
#include "acl/acl.h"
#include "tdt/tdt_host_interface.h"
#include "runtime/config.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/env_var.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/kernels/aicpu/data_item_deliver.h"
#include "tf_adapter/util/acl_channel.h"
#include "tf_adapter/util/host_queue.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter_2.x/npu_device/core/npu_micros.h"

namespace tensorflow {
namespace data {
namespace {
using namespace std;
using namespace tdt;
const uint32_t kMaxValue = 128U;
const size_t kMaxDepth = 128UL;
const int64_t kUnknownShapeDepth = 3LL;
// total memory usage controlled below 2G
const uint64_t kTotalBytes = 2147483648ULL;
const int64_t kMaxBytes = 2 * 1024 * 1024 * 1024LL;
const int32_t kSleepTime = 1;
std::atomic<bool> tdt_release(false);

enum class ChannelType {
  TDT = 0,
  ACL_QUEUE = 1,
  HOST_QUEUE = 2
};

class HostQueueDatasetOp : public DatasetOpKernel {
 public:
  explicit HostQueueDatasetOp(OpKernelConstruction *ctx) : DatasetOpKernel(ctx),
      local_rank_id_(0U), device_id_(0U) {
    // ctx is not nullptr
    std::string local_rank_id;
    std::string local_device_list;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("channel_name", &channel_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("_local_rank_id", &local_rank_id));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("_local_device_list", &local_device_list));
    ADP_LOG(INFO) << "Get local rank id:" << local_rank_id << ", local device list:" << local_device_list;
    // local rank id range 0-7
    local_rank_id_ = std::atoi(local_rank_id.c_str());
    for (size_t i = 0UL; i < local_device_list.size(); i += 2UL) {
      int device_id = std::atoi(&local_device_list[i]);
      OP_REQUIRES(ctx, device_id >= 0, errors::InvalidArgument("device id should be >= 0."));
      local_device_list_.push_back(device_id);
    }
    GetChannelType(channel_type_);
    ADP_LOG(INFO) << "Start to init channel.";
    OP_REQUIRES_OK(ctx, GetEnvDeviceID(device_id_));
    if (channel_type_ == ChannelType::TDT) {
      int32_t tdt_status = TdtInFeedInit(device_id_);
      OP_REQUIRES(ctx, tdt_status == 0, errors::InvalidArgument("Tdt client init failed."));
      ADP_LOG(INFO) << "Init tdt host success.";
    }
    tdt_release = false;
  }

  ~HostQueueDatasetOp() override {
    if (tdt_release || (channel_type_ != ChannelType::TDT)) {
      return;
    }

    ADP_LOG(INFO) << "Start to destroy tdt.";
    int32_t tdt_status = TdtInFeedDestroy(device_id_);
    if (tdt_status != 0) {
      ADP_LOG(ERROR) << "Tdt client close failed, and response code is " << tdt_status;
      LOG(ERROR) << "Tdt client close failed, and response code is " << tdt_status;
    } else {
      ADP_LOG(INFO) << "Tdt client close success.";
      tdt_release = true;
      NpuAttrs::SetUseTdtStatus(device_id_, false);
    }
  }

  void GetChannelType(ChannelType &type) {
    ADP_LOG(INFO) << "kIsNewDataTransfer is " << kIsNewDataTransfer;
    if (kIsHeterogeneous) {
      type = ChannelType::HOST_QUEUE;
    } else if (kIsNewDataTransfer) {
      type = ChannelType::ACL_QUEUE;
    } else {
      type = ChannelType::TDT;
    }
    ADP_LOG(INFO) << "host queue channel type is " << static_cast<int>(type);
  }

  void MakeDataset(OpKernelContext *ctx, DatasetBase **output) override {
    std::vector<DatasetBase *> inputs;
    tf_session_ = ctx->session_handle();
    CHECK_NOT_NULL(output);
    for (int32_t i = 0; i < ctx->num_inputs(); ++i) {
      DatasetBase *input = nullptr;
      OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(i), &input));
      inputs.push_back(input);
    }
    *output = new (nothrow) Dataset(ctx, inputs, channel_name_, output_types_, output_shapes_, local_rank_id_,
                                    local_device_list_, device_id_, tf_session_, channel_type_);
    OP_REQUIRES(ctx, *output != nullptr,
                errors::InvalidArgument("Data process host queue dataset op: new dataset failed."));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext *ctx, const std::vector<DatasetBase *> &inputs, const string &channelName,
            const DataTypeVector &outputTypes, const vector<PartialTensorShape> &outputShapes, const int &local_rank_id,
            const std::vector<uint32_t> &local_device_list, const uint32_t &device_id, const string &tf_session,
            const ChannelType &channel_type)
        : DatasetBase(DatasetContext(ctx)), inputs_(inputs), channel_name_(channelName), output_types_(outputTypes),
          output_shapes_(outputShapes), tf_session_(tf_session), local_rank_id_(local_rank_id),
          local_device_list_(local_device_list), device_id_(device_id), channel_type_(channel_type) {
      for (const auto &input : inputs_) { input->Ref(); }
    }

    ~Dataset() override {
      for (const auto &input : inputs_) {
        input->Unref();
      }
    }

    unique_ptr<IteratorBase> MakeIteratorInternal(const string &prefix) const override {
      return unique_ptr<IteratorBase>(new (nothrow) Iterator({this, strings::StrCat(prefix, "::HostQueue")}));
    }

    const DataTypeVector &output_dtypes() const override { return output_types_; }
    const vector<PartialTensorShape> &output_shapes() const override { return output_shapes_; }

    string DebugString() const override { return "HostQueueDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext *ctx, DatasetGraphDefBuilder *b, Node **output) const override {
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params &params) : DatasetIterator<Dataset>(params) {
        data_deliver_ = new (nothrow) DataItemDeliver(dataset()->local_rank_id_, dataset()->device_id_,
                                                      dataset()->local_device_list_, dataset()->channel_name_);
      }

      ~Iterator() override {
        std::vector<DataItem> stop_message;
        data_deliver_->ParallelSendDataVec(stop_message);
        {
          mutex_lock lck(mu_);
          finish_send_ = true;
          if (dataset()->channel_type_ != ChannelType::TDT) {
            while (!cancelled_) {
              destory_var_.wait(lck);
            }
          }
        }
        // wait for tdt destory for sleeping one second
        sleep(kSleepTime);
        if (dataset()->channel_type_ == ChannelType::TDT) {
          mutex_lock lck(mu_);
          cancelled_ = true;
        }
        cond_var_.notify_all();
        delete data_deliver_;

        ADP_LOG(INFO) << "HostQueueDatasetOp's iterator is released.";
      }

      void DestroyQueue() {
        ADP_LOG(INFO) << "Start to destroy queue";
        if (dataset()->channel_type_ == ChannelType::ACL_QUEUE) {
          aclError acl_status = acltdtDestroyChannel(acl_handle_);
          if (acl_status != ACL_ERROR_NONE) {
            ADP_LOG(ERROR) << "call acltdtDestroyChannel failed, ret=" << acl_status;
          }
        } else {
          HostQueueDestroy(queue_id_);
        }
        ADP_LOG(INFO) << "End to destroy queue";
      }

      void GetDataThread(const std::shared_ptr<IteratorContext> &ctx) {
        RecordStart(ctx.get());
        auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
        while (true) {
          {
            mutex_lock lck(mu_);
            while (!cancelled_ && (buffer_.size() >= kMaxValue || total_bytes_ > kTotalBytes)) {
              RecordStop(ctx.get());
              cond_var_.wait(lck);
              RecordStart(ctx.get());
            }

            if (cancelled_) { return; }
          }

          mutex_lock parent_l(parent_mu_);
          vector<Tensor> args;
          bool end_of_sequence = false;
          BufferElement buffer_element;
          if (dataset()->local_rank_id_ > 0) {
            ADP_LOG(INFO) << "Do not need to GetNext.";
            return;
          }

          buffer_element.status = input_impls_[1]->GetNext(ctx.get(), &args, &end_of_sequence);
          if (!buffer_element.status.ok() || (buffer_element.status.ok() && end_of_sequence)) {
            if (!buffer_element.status.ok()) {
              ADP_LOG(ERROR) << "Failed to get tensor data, Status:" << buffer_element.status.ToString();
              LOG(ERROR) << "Failed to get tensor data, Status:" << buffer_element.status.ToString();
            } else {
              ADP_LOG(INFO) << "Finish to get tensor data, Status:" << buffer_element.status.ToString()
                            << "; end_of_sequence:" << end_of_sequence;
            }
            mutex_lock lck(mu_);
            buffer_element.host_thread_finished = true;
            buffer_.push_back(std::move(buffer_element));
            cond_var_.notify_all();
            return;
          }
          {
            mutex_lock lck(mu_);
            for (auto &tensor : args) {
              if (tensor.TotalBytes() > UINT64_MAX - total_bytes_) {
                ADP_LOG(ERROR) << "The size of tensor is too big";
                LOG(ERROR) << "The size of tensor is too big";
                buffer_element.host_thread_finished = true;
                buffer_.push_back(std::move(buffer_element));
                cond_var_.notify_all();
                return;
              }
              total_bytes_ += tensor.TotalBytes();
            }
            buffer_element.value = args;
            buffer_.push_back(std::move(buffer_element));
            cond_var_.notify_all();
          }
        }
      }

      void SendDataThread() {
        std::vector<DataItem> items;
        while (data_deliver_->RecvDataVec(items).ok()) {
          int32_t tdt_status = TdtHostPushData(dataset()->channel_name_, items,
                                               dataset()->device_id_);
          if (tdt_status != 0) {
            ADP_LOG(ERROR) << "End training as tdt host push data finished:"
                           << tdt_status;
            break;
          }
          items.clear();
        }
        {
          mutex_lock lck(mu_);
          cancelled_ = true;
        }
        ADP_LOG(INFO) << "Slave SendDataThread exit.";
      }

      Status SendDataByAclQueue(const vector<Tensor> &args, const acltdtTensorType &data_type) {
        Status status;
        bool is_need_resend = false;
        do {
          {
            mutex_lock lck(mu_);
            if (finish_send_) {
              break;
            }
          }
          if (is_need_resend) {
            sleep(kSleepTime);
          }
          status = SendTensorsByAcl(acl_handle_, data_type, args, is_need_resend);
        } while (status.ok() && is_need_resend);
        return status;
      }

      Status SendDataByHostQueue(const vector<Tensor> &args, const acltdtTensorType &data_type) {
        bool is_need_resend = false;
        void *buff = nullptr;
        Status status = Status::OK();
        TF_RETURN_IF_ERROR(MappingTensor2Buff(data_type, args, buff));
        do {
          {
            mutex_lock lck(mu_);
            if (finish_send_) {
              HostQueueFreeBuff(buff);
              break;
            }
          }
          if (is_need_resend) {
            sleep(kSleepTime);
          }
          status = HostQueueSendData(queue_id_, buff, is_need_resend);
        } while (status.ok() && is_need_resend);
        return status;
      }

      void SendDataByQueueThread(const std::shared_ptr<IteratorContext> &ctx) {
        ADP_LOG(INFO) << "Begin to send data.";
        while (true) {
          std::vector<Tensor> args;
          acltdtTensorType data_type = ACL_TENSOR_DATA_TENSOR;
          {
            mutex_lock lck(mu_);
            while (!finish_send_ && buffer_.empty()) {
              RecordStop(ctx.get());
              cond_var_.wait(lck);
              RecordStart(ctx.get());
            }
            if (finish_send_) {
              ADP_LOG(INFO) << "Host queue " << dataset()->channel_name_ << " send data thread exit with cancelled: "
                            << cancelled_ << ", finished:" << finish_send_ << " when wait data.";
              cancelled_ = true;
              destory_var_.notify_all();
              return;
            }
          }
          if (buffer_.front().host_thread_finished) {
            data_type = buffer_.front().status.ok() ? ACL_TENSOR_DATA_END_OF_SEQUENCE : ACL_TENSOR_DATA_ABNORMAL;
          } else {
            args = buffer_.front().value;
            buffer_.pop_front();
            for (auto &tensor : args) {
              total_bytes_ -= tensor.TotalBytes();
            }
          }
          Status status = Status::OK();
          if (dataset()->channel_type_ == ChannelType::ACL_QUEUE) {
            status = SendDataByAclQueue(args, data_type);
          } else {
            status = SendDataByHostQueue(args, data_type);
          }
          if (!status.ok()) {
            {
              mutex_lock lck(mu_);
              cancelled_ = true;
            }
            destory_var_.notify_all();
            cond_var_.notify_all();
            ADP_LOG(ERROR) << "Send data failed." << status.ToString();
            return;
          }
          if (data_type == ACL_TENSOR_DATA_END_OF_SEQUENCE || data_type == ACL_TENSOR_DATA_ABNORMAL) {
            std::string data_type_str = data_type == ACL_TENSOR_DATA_END_OF_SEQUENCE ? "end of sequence" : "abnormal";
            {
              mutex_lock lck(mu_);
              cancelled_ = true;
            }
            destory_var_.notify_all();
            cond_var_.notify_all();
            ADP_LOG(INFO) << "Send " << data_type_str << " data success.";
            return;
          }
          cond_var_.notify_all();
        }
      }

      void SendDataThread(const std::shared_ptr<IteratorContext> &ctx) {
        vector<Tensor> args;
        while (true) {
          {
            mutex_lock lck(mu_);
            while (!cancelled_ && !finish_send_ && buffer_.empty()) {
              RecordStop(ctx.get());
              cond_var_.wait(lck);
              RecordStart(ctx.get());
            }
            if (cancelled_ || finish_send_) {
              ADP_LOG(INFO) << "Host queue " << dataset()->channel_name_
                            << " push data thread exit with cancelled: " << cancelled_ << ", finished:" << finish_send_
                            << " when wait data.";
              return;
            }
            if (buffer_.front().host_thread_finished) {
              std::vector<DataItem> items;
              DataItem end_item;
              if (buffer_.front().status.ok()) {
                end_item.dataType_ = TDT_END_OF_SEQUENCE;
                ADP_LOG(INFO) << "Push data finish, end_of_sequence_ is true.";
              } else {
                end_item.dataType_ = TDT_ABNORMAL;
                ADP_LOG(ERROR) << "Get data failed " << buffer_.front().status.ToString();
                LOG(ERROR) << "Get data failed " << buffer_.front().status.ToString();
              }
              end_item.tensorName_ = "";
              end_item.tensorShape_ = "";
              end_item.tensorType_ = "";
              std::shared_ptr<void> data(nullptr);
              end_item.dataLen_ = 0;
              end_item.dataPtr_ = data;
              items.emplace_back(end_item);
              data_deliver_->ParallelSendDataVec(items);
              int32_t tdt_status = TdtHostPushData(dataset()->channel_name_, items, dataset()->device_id_);
              if (tdt_status != 0) { ADP_LOG(INFO) << "End training as tdt host push end data failed " << tdt_status; }
              cancelled_ = true;
              cond_var_.notify_all();
              return;
            } else {
              args = buffer_.front().value;
              buffer_.pop_front();
            }
          }

          string value;
          uint64_t total_bytes = 0ULL;
          std::vector<DataItem> items;
          for (auto &tensor : args) {
            DataItem data_item;
            data_item.dataType_ = TDT_TENSOR;
            data_item.tensorShape_ = tensor.shape().DebugString();
            data_item.tensorType_ = DataTypeString(tensor.dtype());

            if (DataTypeCanUseMemcpy(tensor.dtype())) {
              data_item.dataLen_ = tensor.tensor_data().size();
              data_item.dataPtr_ =
                std::shared_ptr<void>(const_cast<char *>(tensor.tensor_data().data()), [](void *elem) {});
            } else if (tensor.dtype() == DT_STRING) {
              if (tensor.dims() != 0) {
                ADP_LOG(ERROR) << "input of DT_STRING type should be scalar,"
                                  " current dims:"
                               << tensor.dims();
                LOG(ERROR) << "input of DT_STRING type should be scalar,"
                              " current dims:"
                           << tensor.dims();
                mutex_lock lck(mu_);
                cancelled_ = true;
                cond_var_.notify_all();
                return;
              }
              value = tensor.scalar<string>()();
              data_item.dataLen_ = value.size();
              data_item.dataPtr_ = std::shared_ptr<void>(const_cast<char *>(value.data()), [](void *elem) {});
            } else {
              ADP_LOG(ERROR) << "Unexpected data type " << DataTypeString(tensor.dtype());
              LOG(ERROR) << "Unexpected data type " << DataTypeString(tensor.dtype());
              mutex_lock lck(mu_);
              cancelled_ = true;
              cond_var_.notify_all();
              return;
            }
            items.push_back(data_item);
            total_bytes += tensor.TotalBytes();
          }
          // call tdt interface
          data_deliver_->ParallelSendDataVec(items);
          int32_t tdt_status = TdtHostPushData(dataset()->channel_name_, items, dataset()->device_id_);
          if (tdt_status != 0) {
            ADP_LOG(INFO) << "End training as tdt host push data finished: " << tdt_status;
            mutex_lock lck(mu_);
            cancelled_ = true;
            cond_var_.notify_all();
            return;
          }
          {
            mutex_lock lck(mu_);
            total_bytes_ -= total_bytes;
            cond_var_.notify_all();
          }
        }
      }

      Status EnsureReceiveThreadStarted(IteratorContext *ctx) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        // ctx is not nullptr
        if (!receive_thread_) {
          std::shared_ptr<IteratorContext> new_ctx(new (std::nothrow) IteratorContext(*ctx));
          REQUIRES_NOT_NULL(new_ctx);
          REQUIRES_NOT_NULL(ctx->env());
          receive_thread_.reset(
            ctx->env()->StartThread({}, "receive_thread", [this, new_ctx]() { GetDataThread(new_ctx); }));
        }
        return Status::OK();
      }

      Status EnsureSendThreadStarted(IteratorContext *ctx) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (!send_thread_) {
          std::shared_ptr<IteratorContext> new_ctx(new (std::nothrow) IteratorContext(*ctx));
          REQUIRES_NOT_NULL(new_ctx);
          REQUIRES_NOT_NULL(ctx->env());
          if (dataset()->channel_type_ == ChannelType::TDT) {
            if (dataset()->local_rank_id_ <= 0) {
              send_thread_.reset(
                ctx->env()->StartThread({}, "send_thread", [this, new_ctx]() { SendDataThread(new_ctx); }));
            } else {
              send_thread_.reset(ctx->env()->StartThread({}, "send_thread", [this]() { SendDataThread(); }));
            }
          } else {
            if (dataset()->local_rank_id_ <= 0) {
              send_thread_.reset(
                ctx->env()->StartThread({}, "send_thread", [this, new_ctx]() { SendDataByQueueThread(new_ctx); }));
            }
          }
        }
        return Status::OK();
      }

      int64_t GetTensorElementNum(size_t index) {
        PartialTensorShape tensor_shape = dataset()->output_shapes_[index];
        int64_t element_num = 1LL;
        for (int32_t i = 0; i < tensor_shape.dims(); i++) {
          element_num *= tensor_shape.dim_size(i);
        }
        return element_num;
      }

      int64_t GetChannelDepth() {
        size_t output_shape_size = dataset()->output_shapes_.size();
        size_t output_type_size = dataset()->output_types_.size();
        if (output_shape_size != output_type_size) {
          ADP_LOG(ERROR) << "Output_shape_size " << output_shape_size << "is not equal to output_type_size"
                         << output_type_size;
          return -1LL;
        }
        int64_t total_size = 0LL;
        for (size_t i = 0UL; i < output_shape_size; i++) {
          DataType tensor_data = dataset()->output_types_.at(i);
          if (tensor_data == DT_STRING || (dataset()->output_shapes_[i].unknown_rank())) {
            ADP_LOG(INFO) << "Current tensor type is DT_STRING or output shape is unknown shape";
            return kUnknownShapeDepth;
          }
          int64_t element_num = GetTensorElementNum(i);
          total_size += (element_num * static_cast<int64_t>(DataTypeSize(dataset()->output_types_.at(i))));
        }
        if (total_size <= 0LL) {
          ADP_LOG(ERROR) << "Data size is <= 0, and current size is " << total_size;
          return -1LL;
        }
        return (kMaxBytes / static_cast<int64_t>(total_size));
      }

      Status CreateChannel() {
        std::hash<std::string> channel_name_dict;
        auto acl_status = aclrtSetDevice(static_cast<int32_t>(dataset()->device_id_));
        if (acl_status != ACL_SUCCESS) {
          return errors::InvalidArgument("Set device failed.", acl_status);
        }
        std::string channel_name = std::to_string(channel_name_dict(dataset()->tf_session_ +
                                                  dataset()->channel_name_ +
                                                  "_device_" +
                                                  std::to_string(dataset()->device_id_)));
        ADP_LOG(INFO) << "Channel name is " << channel_name;
        size_t channel_depth = 0UL;
        int64_t current_channel_depth = GetChannelDepth();
        if (current_channel_depth <= 0LL) {
          ADP_LOG(ERROR) << "Current data size is unsupported";
          return errors::InvalidArgument("Current data size is unsupported");
        }
        channel_depth = std::min(static_cast<size_t>(current_channel_depth), kMaxDepth);
        ADP_LOG(INFO) << "Channel depth is " << channel_depth;

        if (dataset()->channel_type_ == ChannelType::HOST_QUEUE) {
          return HostQueueInit(channel_name, channel_depth, queue_id_);
        }

        acl_handle_ = acltdtCreateChannelWithCapacity(dataset()->device_id_, channel_name.c_str(), channel_depth);
        if (acl_handle_ == nullptr) {
          ADP_LOG(ERROR) << "Call acltdtCreateChannelWithCapacity failed.";
          return errors::InvalidArgument("Call acltdtCreateChannelWithCapacity failed.");
        }
        ADP_LOG(INFO) << "Create Channel success.";
        return Status::OK();
      }

      Status Initialize(IteratorContext *ctx) override {
        ADP_LOG(INFO) << "Start to check channel name. channelName: " << dataset()->channel_name_;
        if (dataset()->channel_name_.empty()) {
          return errors::InvalidArgument("HostQueueDataset channel_name is null.");
        }
        if (data_deliver_ == nullptr) {
          ADP_LOG(ERROR) << "Data deliver is nullptr";
          return errors::InvalidArgument("Data deliver is nullptr");
        }

        ADP_LOG(INFO) << "Start to check receive and send thread.";
        try {
          input_impls_.resize(dataset()->inputs_.size());
        } catch (...) { return errors::InvalidArgument("HostQueueDataset resize failed."); }
        if (dataset()->channel_type_ != ChannelType::TDT && (!CreateChannel().ok())) {
          return errors::InvalidArgument("Call CreatChannel queue failed");
        }
        for (size_t i = 0; i < input_impls_.size(); ++i) {
          TF_RETURN_IF_ERROR(
              dataset()->inputs_[i]->MakeIterator(ctx, strings::StrCat(prefix(), "[", i, "]"), &input_impls_[i]));
        }
        if (dataset()->channel_type_ == ChannelType::TDT) {
          if (dataset()->local_rank_id_ == 0) {
            TF_RETURN_IF_ERROR(data_deliver_->ParallelInitSocketClient());
          } else if (dataset()->local_rank_id_ > 0) {
            TF_RETURN_IF_ERROR(data_deliver_->InitSocketServer());
          }
        }
        {
          mutex_lock lck(mu_);
          cancelled_ = false;
          TF_RETURN_IF_ERROR(EnsureReceiveThreadStarted(ctx));
          TF_RETURN_IF_ERROR(EnsureSendThreadStarted(ctx));
        }
        ADP_LOG(INFO) << "HostQueue success to Initialize. channelName: " << dataset()->channel_name_;
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext *ctx, vector<Tensor> *outTensors, bool *endOfSequence) override {
        *endOfSequence = false;
        ADP_LOG(INFO) << "HostQueue Get Next data.";
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter *writer) override { return Status::OK(); }

      Status RestoreInternal(IteratorContext *ctx, IteratorStateReader *reader) override { return Status::OK(); }

     private:
      struct BufferElement {
        bool host_thread_finished = false;
        Status status;
        vector<Tensor> value;
      };
      // This mutex is used to ensure exclusivity between multiple threads
      // reading/writing this iterator's local state.
      mutex mu_;
      // This mutex is used to ensure exclusivity between multiple threads
      // accessing the parent iterator. We keep this separate from `mu_` to
      // allow prefetching to run in parallel with GetNext calls.
      mutex parent_mu_ ACQUIRED_BEFORE(mu_);
      std::vector<std::unique_ptr<IteratorBase>> input_impls_ GUARDED_BY(mu_);
      condition_variable cond_var_;
      condition_variable destory_var_;
      std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
      bool cancelled_ GUARDED_BY(mu_) = true;
      bool finish_send_ GUARDED_BY(mu_) = false;
      uint64_t total_bytes_ GUARDED_BY(mu_) = 0;
      // The following two thread must be the first member to be destructed,
      // because tensorflow::Thread does not provide an explicit join function.
      // If the thread is destructed after other members, such as buffer_, when
      // the thread joins, it will access the already destructed buffer_ ,
      // Resulting in an unknown error.
      std::unique_ptr<Thread> receive_thread_ GUARDED_BY(mu_);
      std::unique_ptr<Thread> send_thread_ GUARDED_BY(mu_);
      DataItemDeliver *data_deliver_;
      acltdtChannelHandle *acl_handle_;
      uint32_t queue_id_;
    };
    const std::vector<DatasetBase *> inputs_;
    std::string channel_name_;
    const DataTypeVector output_types_;
    const vector<PartialTensorShape> output_shapes_;
    std::string tf_session_;
    int local_rank_id_;
    std::vector<uint32_t> local_device_list_;
    uint32_t device_id_;
    ChannelType channel_type_;
  };
  std::string channel_name_;
  DataTypeVector output_types_;
  vector<PartialTensorShape> output_shapes_;
  std::string tf_session_;
  int local_rank_id_;
  std::vector<uint32_t> local_device_list_;
  uint32_t device_id_;
  ChannelType channel_type_;
};

REGISTER_KERNEL_BUILDER(Name("HostQueueDataset").Device(DEVICE_CPU), HostQueueDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
