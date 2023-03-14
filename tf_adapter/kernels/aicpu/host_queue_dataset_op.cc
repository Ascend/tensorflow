/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
#include <algorithm>
#include "unistd.h"
#include "acl/acl_tdt.h"
#include "acl/acl.h"
#include "tdt/tdt_host_interface.h"
#include "runtime/config.h"
#include "runtime/dev.h"
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
#include "tf_adapter/common/common.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/compat_tf1_tf2.h"
#include "tf_adapter/util/util.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/acl_channel.h"
#include "tf_adapter/util/host_queue.h"
#include "tf_adapter/util/memory_pool.h"
#include "tf_adapter/util/host_thread_pool.h"
#include "tf_adapter/util/host_allocator.h"
#include "tf_adapter/kernels/aicpu/data_item_deliver.h"
#include "tf_adapter/kernels/aicpu/npu_tensor.h"

namespace tensorflow {
namespace data {
namespace {
using namespace std;
using namespace tdt;

constexpr int64 kSleepUs = 10;
const uint32_t kMaxValue = 128U;
const size_t kMaxDepth = 128UL;
const int32_t kSleepTime = 1;
const static int64_t kStringTypeDepth = 64LL;
const int64_t kUnknownShapeDepth = 3LL;
const uint64_t PARALLEL_MEMORY_TRHESHOLD = 10 * 1024 * 1024ULL;
const uint32_t MAX_THREAD_NUM = 4U;
std::atomic<bool> tdt_release(false);
// total memory usage controlled below 2G
const uint64_t kTotalBytes = 8 * 1024 * 1024 * 1024LL;
const int64_t kMaxBytes = 2 * 1024 * 1024 * 1024LL;
enum class ChannelType { TDT = 0, ACL_QUEUE = 1, HOST_QUEUE = 2 }; /* ACL_QUEUE indicates mbuf */
enum class ThreadType : size_t { RECV = 0, SEND = 1, BUTT };

class HostQueueDatasetOp : public DatasetOpKernel {
 public:
  explicit HostQueueDatasetOp(OpKernelConstruction *ctx)
      : DatasetOpKernel(ctx), local_rank_id_(0U), device_id_(0U), queue_id_(0U) {
    // ctx is not nullptr
    std::string local_rank_id;
    std::string local_device_list;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("channel_name", &channel_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("_local_rank_id", &local_rank_id));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("_local_device_list", &local_device_list));

    ADP_LOG(INFO) << "Get local rank id: " << local_rank_id << ", local device list: " << local_device_list;
    // local rank id range 0-7
    local_rank_id_ = std::atoi(local_rank_id.c_str());
    for (size_t i = 0UL; i < local_device_list.size(); i += 2UL) {
      int device_id = std::atoi(&local_device_list[i]);
      OP_REQUIRES(ctx, device_id >= 0, errors::InvalidArgument("device id should be >= 0."));
      local_device_list_.push_back(device_id);
    }
    SetChannelType();
    ADP_LOG(INFO) << "Start to init channel.";
    OP_REQUIRES_OK(ctx, GetEnvDeviceID(device_id_));
    if (channel_type_ == ChannelType::TDT) {
      int32_t tdt_status = TdtInFeedInit(device_id_);
      OP_REQUIRES(ctx, tdt_status == 0, errors::InvalidArgument("Tdt client init failed."));
      ADP_LOG(INFO) << "End init tdt host success.";
    }
    tdt_release = false;
  }

  ~HostQueueDatasetOp() override {
    if ((kIsHeterogeneous) && (!queue_name_.empty())) {
      HostQueueDestroy(queue_id_);
    }

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

  void SetChannelType() {
    if (kIsHeterogeneous) {
      channel_type_ = ChannelType::HOST_QUEUE;
    } else if (NpuAttrs::GetNewDataTransferFlag()) {
      channel_type_ = ChannelType::ACL_QUEUE;
    } else {
      channel_type_ = ChannelType::TDT;
    }
    ADP_LOG(INFO) << "Host queue channel type [0:TDT 1:MBuf 2:HostQueue] is: "
                  << static_cast<int>(channel_type_);
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
    size_t channel_depth = 0U;
    if (channel_type_ != ChannelType::TDT) {
      int64_t queue_depth = GetChannelDepth();
      OP_REQUIRES(ctx, queue_depth > 0LL, errors::InvalidArgument("Current data size is unsupported."));
      channel_depth = std::min(static_cast<size_t>(queue_depth), kMaxDepth);
      ADP_LOG(INFO) << "Channel depth is : " << channel_depth;
      if (kIsHeterogeneous) {
        CreateHostQueue(ctx, channel_depth);
      }
    }
    *output = new (nothrow) Dataset(ctx, inputs, channel_name_,
                                    output_types_, output_shapes_, local_rank_id_,
                                    local_device_list_, device_id_, tf_session_,
                                    channel_type_, channel_depth, queue_id_);
    OP_REQUIRES(ctx, *output != nullptr,
                errors::InvalidArgument("Data process host queue dataset op: new dataset failed."));
  }

  int64_t GetTensorElementNum(size_t index) {
    PartialTensorShape tensor_shape = output_shapes_[index];
    int64_t element_number = 1LL;
    for (int32_t i = 0; i < tensor_shape.dims(); i++) {
      element_number *= tensor_shape.dim_size(i);
    }
    return element_number;
  }

  bool IsUnknownShape(const PartialTensorShape &output_shapes) const {
    if (output_shapes.unknown_rank()) {
      return true;
    }
    for (int32_t i = 0; i < output_shapes.dims(); i++) {
      if (output_shapes.dim_size(i) == -1) {
        return true;
      }
    }
    return false;
  }

  int64_t GetChannelDepth() {
    size_t output_shape_size = output_shapes_.size();
    size_t output_type_size = output_types_.size();
    if (output_shape_size != output_type_size) {
      ADP_LOG(ERROR) << "Output_shape_size : " << output_shape_size << "is not equal to output_type_size : "
                     << output_type_size;
      return -1LL;
    }
    int64_t total_sizes = 0LL;
    for (size_t i = 0UL; i < output_shape_size; i++) {
      DataType tensor_data_type = output_types_.at(i);
      if (tensor_data_type == DT_STRING) {
        ADP_LOG(INFO) << "Current tensor type is DT_STRING.";
        return kStringTypeDepth;
      }
      if (IsUnknownShape(output_shapes_[i])) {
        ADP_LOG(INFO) << " Output_shape is unknow shape";
        return kUnknownShapeDepth;
      }
      int64_t element_number = GetTensorElementNum(i);
      total_sizes += (element_number * static_cast<int64_t>(DataTypeSize(output_types_.at(i))));
    }
    if (total_sizes <= 0LL) {
      ADP_LOG(ERROR) << "Data size is <= 0, and current size is " << total_sizes;
      return -1LL;
    }
    return std::max(2L, (kMaxBytes / total_sizes));
  }

  void CreateHostQueue(OpKernelContext *ctx, size_t channel_depth) {
    std::hash<std::string> channel_name_dict;
    std::string queue_name = std::to_string(
      channel_name_dict(tf_session_ + channel_name_ + "_device_" + std::to_string(device_id_)));
    if (queue_name_ == queue_name) {
      ADP_LOG(INFO) << "The channel is already create.";
      return;
    }
    ADP_LOG(INFO) << "Channel name is: " << queue_name << ", Channel depth is: " << channel_depth;
    Status ret = HostQueueInit(queue_name, channel_depth, queue_id_);
    OP_REQUIRES(ctx, ret == Status::OK(), errors::InvalidArgument("Failed to create host queue."));
    queue_name_ = queue_name;
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext *ctx, const std::vector<DatasetBase *> &inputs, const string &channelName,
            const DataTypeVector &outputTypes, const vector<PartialTensorShape> &outputShapes,
            const int &local_rank_id, const std::vector<uint32_t> &local_device_list, const uint32_t &device_id,
            const string &tf_session, const ChannelType &channel_type, size_t channel_depth, uint32_t queue_id)
        : DatasetBase(DatasetContext(ctx)), inputs_(inputs), channel_name_(channelName),
          output_types_(outputTypes), output_shapes_(outputShapes), tf_session_(tf_session),
          local_rank_id_(local_rank_id), local_device_list_(local_device_list), device_id_(device_id),
          channel_type_(channel_type), channel_depth_(channel_depth), queue_id_(queue_id) {
      for (const auto &input : inputs_) { input->Ref(); }
    }

    ~Dataset() override {
      for (const auto &input : inputs_) {
        input->Unref();
      }
    }

    unique_ptr<IteratorBase> MakeIteratorInternal(const string &prefix) const override {
      return absl::make_unique<Iterator>(Iterator::Params({this, prefix + "::HostQueue"}));
    }

    const DataTypeVector &output_dtypes() const override { return output_types_; }
    const vector<PartialTensorShape> &output_shapes() const override { return output_shapes_; }

    string DebugString() const override { return "HostQueueDatasetOp::Dataset"; }

#ifdef TF_VERSION_TF2
    Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
      for (const auto &input : inputs_) { inputs->push_back(input); }
      return Status::OK();
    }
#endif

    STATUS_FUNCTION_ONLY_TF2(CheckExternalState() const override);

   protected:
    Status AsGraphDefInternal(SerializationContext *ctx, DatasetGraphDefBuilder *b, Node **output) const override {
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params &params)
          : DatasetIterator<Dataset>(params),
            data_deliver_(new (nothrow) DataItemDeliver(dataset()->local_rank_id_,
                                                        dataset()->device_id_,
                                                        dataset()->local_device_list_,
                                                        dataset()->channel_name_)) {
      }

      ~Iterator() override {
        ADP_LOG(EVENT) << "DataThreadPerf[" << dataset()->device_id_ <<
          "]::channel_name:" << dataset()->channel_name_ <<
          "[" << buffer_.size() << "], recv [" <<
          data_thread_perf_stat_[static_cast<size_t>(ThreadType::RECV)].elapsed_time << "us, " <<
          data_thread_perf_stat_[static_cast<size_t>(ThreadType::RECV)].total_bytes << "], send [" <<
          data_thread_perf_stat_[static_cast<size_t>(ThreadType::SEND)].elapsed_time << "us, " <<
          data_thread_perf_stat_[static_cast<size_t>(ThreadType::SEND)].total_bytes << "].";
        std::vector<DataItem> stop_message;
        data_deliver_->ParallelSendDataVec(stop_message);
        {
          mutex_lock lck(mu_);
          finish_send_ = true;
          cond_var_.notify_all();
          if (dataset()->channel_type_ != ChannelType::TDT) {
            while (!cancelled_) {
              destory_var_.wait(lck);
            }
          }
        }
        // wait for tdt destory for sleeping one second
        if (dataset()->channel_type_ == ChannelType::TDT) {
          sleep(kSleepTime);
          mutex_lock lck(mu_);
          cancelled_ = true;
        }
        cond_var_.notify_all();

        while (active_thread_num) {
          cond_var_.notify_all();
          (void)usleep(kSleepUs);
        }

        delete data_deliver_;
        FinishMemory();
        DestroyQueue();
        ADP_LOG(INFO) << "HostQueueDatasetOp's iterator is released.";
      }

      void DestroyQueue() {
        ADP_LOG(INFO) << "Start to destroy queue";
        if (dataset()->channel_type_ == ChannelType::ACL_QUEUE) {
          aclError acl_status = acltdtDestroyChannel(acl_handle_);
          if (acl_status != ACL_ERROR_NONE) {
            ADP_LOG(ERROR) << "call acltdtDestroyChannel failed, ret=" << acl_status;
          } else {
            acl_handle_ = nullptr;
          }
        }
        ADP_LOG(INFO) << "End to destroy queue.";
      }

      // Indicates that a thread execution task has finished
      void NotifyEventFinish() {
        if (event_num_.fetch_add(1U) >= (MAX_THREAD_NUM - 1)) {
          {
            mutex_lock lck(event_finish_mu_);
            event_finish_flag_ = true;
          }
          event_finish_var_.notify_all();
        }
      }

      bool ParallelCopy(void* const dst_ptr, uint64_t dst_size,
                        const char *src_ptr, uint64_t src_size) {
        event_num_.store(0U);
        {
          mutex_lock lck(event_finish_mu_);
          event_finish_flag_ = false;
        }
        uint64_t block_size = (src_size / MAX_THREAD_NUM);
        uint64_t remain_size = (src_size % MAX_THREAD_NUM);
        uint64_t start_pos = 0ULL;
        uint64_t end_pos = 0ULL;
        std::atomic<bool> closure_ret(true);
        std::function<void()> closure;
        for (uint32_t i = 0U; i < MAX_THREAD_NUM; i++) {
          start_pos = i * block_size;
          end_pos = (i + 1) * block_size;
          if (i == MAX_THREAD_NUM - 1) {
            end_pos += remain_size;
          }
          closure = [start_pos, end_pos, &dst_ptr, &dst_size, &src_ptr, &src_size, &closure_ret, this] () {
            char *dst = reinterpret_cast<char *>(reinterpret_cast<uintptr_t>(dst_ptr) + start_pos);
            const char *src = src_ptr + start_pos;
            uint64_t dst_len = dst_size - start_pos;
            // end_pos must larger than start_pos
            uint64_t len = end_pos - start_pos;
            if (len > src_size || dst_len < len) {
              closure_ret = false;
              NotifyEventFinish();
              ADP_LOG(ERROR) << "Parameters is invalid. "<< "[len:" << len << ", buffer_size:" << dst_len <<"].";
              return;
            }
            uint64_t temp_len = 0ULL;
            do {
              uint64_t temp_copy_size = len - temp_len;
              uint64_t copy_size = (temp_copy_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : temp_copy_size;
              if (memcpy_s(reinterpret_cast<void *>(dst), static_cast<size_t>(dst_len),
                           src, static_cast<size_t>(copy_size)) != EOK) {
                closure_ret = false;
                NotifyEventFinish();
                ADP_LOG(ERROR) << "Failed to execute memcpy_s. [" << start_pos << "," << copy_size << "].";
                return;
              }
              temp_len += copy_size;
              dst_len -= copy_size;
              dst += copy_size;
              src += copy_size;
            } while (temp_len < len);
            NotifyEventFinish();
          };
          thread_pool_.PushTask(closure);
        }
        ADP_LOG(INFO) << "Wait for all threads finish to copy.";
        mutex_lock lck(event_finish_mu_);
        while (!event_finish_flag_) {
          event_finish_var_.wait(lck);
        }
        ADP_LOG(INFO) << "All threads has finished to copy " <<
          (closure_ret ? "successfully" : "unsuccessfully") << ".";
        return closure_ret;
      }

      bool CopyTensor(void* const memory_ptr, uint64_t memory_size,
                      std::vector<Tensor> &args, std::vector<Tensor> &result_args) {
        uint64_t offset = 0ULL;
        for (size_t i = 0UL; i < args.size(); i++) {
          const char *src_ptr = args[i].tensor_data().data();
          uint64_t src_size = args[i].tensor_data().size();
          void *dst_ptr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(memory_ptr) + offset);
          uint64_t dst_size = memory_size - offset;
          if (src_size > PARALLEL_MEMORY_TRHESHOLD) {
            if (!ParallelCopy(dst_ptr, dst_size, src_ptr, src_size)) {
              ADP_LOG(ERROR) << "Memcpy failed, dst_size:" << dst_size << ", src_size: " << src_size;
              return false;
            }
          } else {
            if (memcpy_s(dst_ptr, dst_size, src_ptr, src_size) != EOK) {
              ADP_LOG(ERROR) << "Memcpy failed, dst_size:" << dst_size << ", src_size: " << src_size;
              return false;
            }
          }
          HostAllocator *host_addr = new (std::nothrow) HostAllocator(dst_ptr);
          if (host_addr == nullptr) {
            ADP_LOG(ERROR) << "Allocator memory failed";
            return false;
          }
          Tensor result_tensor(host_addr, args[i].dtype(), args[i].shape());
          result_args.emplace_back(std::move(result_tensor));
          offset += src_size;
        }
        return true;
      }

      void FinishMemory() {
        if (dataset()->channel_type_ == ChannelType::ACL_QUEUE) {
          ADP_LOG(INFO) << "Begin to release host memory pool.";
          thread_pool_.StopThreadPool();
          (void)mem_pool_.FreeAllMemory();
          ADP_LOG(INFO) << "Finish to release host memory pool.";
        }
      }

      void RefreshDataThreadPerf(const ThreadType type, const double elapsed_time,
                                 const std::vector<Tensor> &args) {
        if (elapsed_time > data_thread_perf_stat_[static_cast<size_t>(type)].elapsed_time) {
          uint64_t total_bytes = 0;
          for (auto &tensor : args) {
            total_bytes += tensor.TotalBytes();
          }
          data_thread_perf_stat_[static_cast<size_t>(type)].total_bytes = total_bytes;
          data_thread_perf_stat_[static_cast<size_t>(type)].elapsed_time = elapsed_time;
        }
      }

      void GetDataThread(const std::shared_ptr<IteratorContext> &ctx) {
        {
          mutex_lock lck(mu_);
          active_thread_num++;
        }

        RecordStart(ctx.get());
        auto cleanup = gtl::MakeCleanup([this, ctx] {
          RecordStop(ctx.get());
          {
            mutex_lock lck(mu_);
            active_thread_num--;
          }
        });
        enum NPU_ALLOCATOR_CHECK {
          NPU_ALLOCATOR_UNKNOW,
          NPU_ALLOCATOR_NONPU,
          NPU_ALLOCATOR_NPU,
        } from_npu_dataset = NPU_ALLOCATOR_UNKNOW;

        ADP_LOG(INFO) << "The data receiving thread starts to work.";
        while (true) {
          {
            mutex_lock lck(mu_);
            while ((!cancelled_) && ((buffer_.size() >= kMaxValue) || (total_bytes_ > kTotalBytes))) {
              RecordStop(ctx.get());
              cond_var_.wait(lck);
              RecordStart(ctx.get());
            }

            if (cancelled_) {
              FinishMemory();
              return;
            }
          }

          mutex_lock parent_l(parent_mu_);
          vector<Tensor> args;
          bool end_of_sequence = false;
          BufferElement buffer_element;
          if (dataset()->local_rank_id_ > 0) {
            ADP_LOG(INFO) << "Do not need to GetNext.";
            get_thread_exception_.store(true);
            return;
          }

          auto start = std::chrono::steady_clock::now();
          buffer_element.status = input_impls_[1]->GetNext(ctx.get(), &args, &end_of_sequence);
          auto end = std::chrono::steady_clock::now();
          if ((!buffer_element.status.ok()) || (buffer_element.status.ok() && end_of_sequence)) {
            if ((!buffer_element.status.ok()) &&
                (!errors::IsCancelled(buffer_element.status))) {
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
            get_thread_exception_.store(true);
            return;
          }
          auto elapsed_time = std::chrono::duration<double, std::micro>(end - start).count();
          RefreshDataThreadPerf(ThreadType::RECV, elapsed_time, args);
          uint64_t args_tensor_size = 0ULL;
          if (from_npu_dataset == NPU_ALLOCATOR_UNKNOW) {
            if (args.empty()) {
              ADP_LOG(ERROR) << "args is null";
              continue;
            }
            from_npu_dataset = NpuAllocator::IsNpuAllocator(args[0]) ? NPU_ALLOCATOR_NPU : NPU_ALLOCATOR_NONPU;
            ADP_LOG(INFO) << "from_npu_dataset = " << static_cast<int>(from_npu_dataset);
          }
          bool is_string = false;
          {
            mutex_lock lck(mu_);
            for (auto &tensor : args) {
              if ((!is_string) && (from_npu_dataset != NPU_ALLOCATOR_NPU) && (tensor.dtype() == DT_STRING)) {
                ADP_LOG(INFO) << "Data type is string, do not use memory pool";
                is_string = true;
              }
              if (tensor.TotalBytes() > (UINT64_MAX - total_bytes_)) {
                ADP_LOG(ERROR) << "The size of tensor is too big";
                LOG(ERROR) << "The size of tensor is too big";
                buffer_element.host_thread_finished = true;
                buffer_.push_back(std::move(buffer_element));
                cond_var_.notify_all();
                get_thread_exception_.store(true);
                return;
              }
              args_tensor_size += tensor.TotalBytes();
              total_bytes_ += tensor.TotalBytes();
            }
          }
          if ((!is_string) && (from_npu_dataset != NPU_ALLOCATOR_NPU) &&
              (dataset()->channel_type_ == ChannelType::ACL_QUEUE)) {
            if (!HandleMemory(args, buffer_element.value, args_tensor_size)) {
              get_thread_exception_.store(true);
              return;
            }
          } else {
            buffer_element.value = args;
          }
          {
            mutex_lock lck(mu_);
            buffer_.push_back(std::move(buffer_element));
            cond_var_.notify_all();
          }
        }
      }

      void SendDataThread() {
        {
          mutex_lock lck(mu_);
          active_thread_num++;
        }

        auto cleanup = gtl::MakeCleanup([this] {
          {
            mutex_lock lck(mu_);
            active_thread_num--;
          }
        });
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
            if (finish_send_) { break; }
          }
          auto start = std::chrono::steady_clock::now();
          status = SendTensorsByAcl(acl_handle_, data_type, args, is_need_resend);
          auto end = std::chrono::steady_clock::now();
          if (status.ok() && !is_need_resend) {
            auto elapsed_time = std::chrono::duration<double, std::micro>(end - start).count();
            RefreshDataThreadPerf(ThreadType::SEND, elapsed_time, args);
          }
        } while (status.ok() && is_need_resend);
        return status;
      }

      Status SendDataByHostQueue(const vector<Tensor> &args, const acltdtTensorType &data_type) {
        Status status;
        bool is_need_resend = false;
        void *buff = nullptr;
        TF_RETURN_IF_ERROR(MappingTensor2Buff(data_type, args, buff));
        TF_RETURN_IF_ERROR(HostQueueSetTransId(dataset()->queue_id_, buff));
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
          status = HostQueueSendData(dataset()->queue_id_, buff, is_need_resend);
        } while (status.ok() && is_need_resend);
        return status;
      }

      void StopNotify() {
        {
          mutex_lock lck(mu_);
          cancelled_ = true;
        }
        destory_var_.notify_all();
        cond_var_.notify_all();
      }

      bool HandleMemory(std::vector<Tensor> &src_args, std::vector<Tensor> &dst_args, uint64_t args_size) {
        void *buffer = nullptr;
        if (!mem_pool_.MallocMemory(buffer, args_size).ok()) {
          ADP_LOG(ERROR) << "MallocMemory memory failed";
          return false;
        }
        if (buffer == nullptr) {
          ADP_LOG(ERROR) << "Memory buffer is invalid";
          return false;
        }
        if (!CopyTensor(buffer, args_size, src_args, dst_args)) {
          ADP_LOG(ERROR) << "Copy data to memory failed";
          return false;
        }
        return true;
      }

      void SendDataByQueueThread(const std::shared_ptr<IteratorContext> &ctx) {
        ADP_LOG(INFO) << "Begin to send data to the NPU. ";
        rtError_t rt = rtSetDevice(dataset()->device_id_);
        if (rt != ACL_RT_SUCCESS) {
          ADP_LOG(ERROR) << "Thread rtSetDevice failed: device_id_ = " << dataset()->device_id_ << " rt=" << rt;
        }

        {
          mutex_lock lck(mu_);
          active_thread_num++;
        }

        auto cleanup = gtl::MakeCleanup([this] {
          {
            mutex_lock lck(mu_);
            active_thread_num--;
          }
          rtError_t rt = rtDeviceReset(this->dataset()->device_id_);
          if (rt != RT_ERROR_NONE) {
            ADP_LOG(ERROR) << "Call reset device failed. device id=" << dataset()->device_id_ << " rt=" << rt;
          }
        });

        while (true) {
          std::vector<Tensor> args;
          acltdtTensorType data_type = ACL_TENSOR_DATA_TENSOR;
          {
            mutex_lock lck(mu_);
            while ((!finish_send_) && buffer_.empty()) {
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
            if (buffer_.front().host_thread_finished) {
              data_type = buffer_.front().status.ok() ? ACL_TENSOR_DATA_END_OF_SEQUENCE : ACL_TENSOR_DATA_ABNORMAL;
            } else {
              args = buffer_.front().value;
              buffer_.pop_front();
              for (auto &tensor : args) {
                total_bytes_ -= tensor.TotalBytes();
              }
            }
            ADP_LOG(INFO) << "Host queue " << dataset()->channel_name_
                          << ", buffer_size: " << buffer_.size() << ", data_type:" << data_type;
          }
          Status status;
          if (dataset()->channel_type_ == ChannelType::ACL_QUEUE) {
            status = SendDataByAclQueue(args, data_type);
          } else {
            status = SendDataByHostQueue(args, data_type);
          }
          if (!status.ok()) {
            StopNotify();
            ADP_LOG(ERROR) << "Send data failed." << status.ToString();
            return;
          }
          if ((data_type == ACL_TENSOR_DATA_END_OF_SEQUENCE) || (data_type == ACL_TENSOR_DATA_ABNORMAL)) {
            std::string data_type_str =
              (data_type == ACL_TENSOR_DATA_END_OF_SEQUENCE) ? "end of sequence" : "abnormal";
            StopNotify();
            ADP_LOG(INFO) << "Host queue send " << data_type_str << " data success.";
            return;
          }
          mem_pool_.ReleaseMemory();
          cond_var_.notify_all();
        }
      }

      void SendDataThread(const std::shared_ptr<IteratorContext> &ctx) {
        {
          mutex_lock lck(mu_);
          active_thread_num++;
        }

        auto cleanup = gtl::MakeCleanup([this] {
          {
            mutex_lock lck(mu_);
            active_thread_num--;
          }
        });
        vector<Tensor> args;
        while (true) {
          {
            mutex_lock lck(mu_);
            while ((!cancelled_) && (!finish_send_) && buffer_.empty()) {
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
              std::vector<DataItem> data_items;
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
              data_items.emplace_back(end_item);
              data_deliver_->ParallelSendDataVec(data_items);
              int32_t tdt_status = TdtHostPushData(dataset()->channel_name_, data_items, dataset()->device_id_);
              if (tdt_status != 0) {
                ADP_LOG(INFO) << "End training as tdt host push end data, ret != 0 " << tdt_status;
              }
              cancelled_ = true;
              cond_var_.notify_all();
              return;
            } else {
              args = buffer_.front().value;
              buffer_.pop_front();
              ADP_LOG(INFO) << "Host queue " << dataset()->channel_name_ << ", buffer_size: " << buffer_.size();
            }
          }

          string value;
          uint64_t bytes_sum = 0ULL;
          std::vector<DataItem> items;
          std::vector<std::unique_ptr<uint8_t[]>> buff_list;
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
              Status s = MappingDTStringTensor2DataItem(tensor, data_item, buff_list);
              if (!s.ok()) {
                ADP_LOG(ERROR) << "mapping dt_stirng type tensor failed, current dims:" << tensor.dims();
                LOG(ERROR) << "mapping dt_stirng type tensor failed, current dims:" << tensor.dims();
                mutex_lock lck(mu_);
                cancelled_ = true;
                cond_var_.notify_all();
                return;
              }
            } else {
              ADP_LOG(ERROR) << "Unexpected data type " << DataTypeString(tensor.dtype());
              LOG(ERROR) << "Unexpected data type " << DataTypeString(tensor.dtype());
              mutex_lock lck(mu_);
              cancelled_ = true;
              cond_var_.notify_all();
              return;
            }
            items.push_back(data_item);
            bytes_sum += tensor.TotalBytes();
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
            total_bytes_ -= bytes_sum;
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

      Status CreateChannel() {
        std::hash<std::string> channel_name_dict;
        std::string channel_name = std::to_string(channel_name_dict(dataset()->tf_session_ +
                                                  dataset()->channel_name_ +
                                                  "_device_" +
                                                  std::to_string(dataset()->device_id_)));
        acl_handle_ = acltdtCreateChannelWithCapacity(
          dataset()->device_id_, channel_name.c_str(), static_cast<size_t>(dataset()->channel_depth_));
        if (acl_handle_ == nullptr) {
          ADP_LOG(ERROR) << "Call acltdtCreateChannelWithCapacity failed.";
          return errors::InvalidArgument("Call acltdtCreateChannelWithCapacity failed.");
        }
        ADP_LOG(INFO) << "Create Channel success. channel_name:" << channel_name;
        return Status::OK();
      }

      Status Initialize(IteratorContext *ctx) override {
        if (dataset()->channel_name_.empty()) {
          return errors::InvalidArgument("HostQueueDataset channel_name is null.");
        }
        if (data_deliver_ == nullptr) {
          ADP_LOG(ERROR) << "Data deliver is nullptr";
          return errors::InvalidArgument("Data deliver is nullptr");
        }

        ADP_LOG(INFO) << "Start to enable receive and send threads.";
        try {
          input_impls_.resize(dataset()->inputs_.size());
        } catch (...) { return errors::InvalidArgument("HostQueueDataset resize failed."); }
        if ((dataset()->channel_type_ == ChannelType::ACL_QUEUE) && (!CreateChannel().ok())) {
          return errors::InvalidArgument("Call CreatChannel queue failed");
        }
        for (size_t i = 0; i < input_impls_.size(); ++i) {
          TF_RETURN_IF_ERROR(
#ifdef TF_VERSION_TF2
              dataset()->inputs_[i]->MakeIterator(ctx, this, prefix() + "[" + std::to_string(i) + "]", &input_impls_[i])
#else
              dataset()->inputs_[i]->MakeIterator(ctx, prefix() + "[" + std::to_string(i) + "]", &input_impls_[i])
#endif
          );
        }
        if (dataset()->channel_type_ == ChannelType::TDT) {
          if (dataset()->local_rank_id_ == 0) {
            TF_RETURN_IF_ERROR(data_deliver_->ParallelInitSocketClient());
          } else if (dataset()->local_rank_id_ > 0) {
            TF_RETURN_IF_ERROR(data_deliver_->InitSocketServer());
          }
        }
        if (dataset()->channel_type_ == ChannelType::ACL_QUEUE) {
          TF_RETURN_IF_ERROR(thread_pool_.Init(dataset()->device_id_));
        }
        {
          mutex_lock lck(mu_);
          cancelled_ = false;
          TF_RETURN_IF_ERROR(EnsureReceiveThreadStarted(ctx));
          TF_RETURN_IF_ERROR(EnsureSendThreadStarted(ctx));
        }
        ADP_LOG(INFO) << "HostQueue success to Initialize. channel_name_:" << dataset()->channel_name_;
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext *ctx, vector<Tensor> *outTensors, bool *endOfSequence) override {
        *endOfSequence = false;
        ADP_LOG(INFO) << "HostQueue GetNextInternal End.";
        return Status::OK();
      }

     protected:
      STATUS_FUNCTION_ONLY_TF2(SaveInternal(SerializationContext *ctx, IteratorStateWriter *writer) override);
      STATUS_FUNCTION_ONLY_TF1(SaveInternal(IteratorStateWriter *writer) override);

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
      MemoryPool mem_pool_;
      HostThreadPool thread_pool_;
      std::atomic<uint32_t> event_num_;
      mutex event_finish_mu_;
      bool event_finish_flag_ GUARDED_BY(event_finish_mu_) = false;
      condition_variable event_finish_var_;
      bool cancelled_ GUARDED_BY(mu_) = true;
      bool finish_send_ GUARDED_BY(mu_) = false;
      std::atomic<bool> get_thread_exception_ { false };
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
      int active_thread_num = 0;
      struct DataThreadPerf {
        double elapsed_time = 0;
        uint64_t total_bytes = 0;
      } data_thread_perf_stat_[static_cast<size_t>(ThreadType::BUTT)];
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
    int64_t channel_depth_;
    uint32_t queue_id_;
  };
  std::string channel_name_;
  DataTypeVector output_types_;
  vector<PartialTensorShape> output_shapes_;
  std::string tf_session_;
  int local_rank_id_;
  std::vector<uint32_t> local_device_list_;
  uint32_t device_id_;
  ChannelType channel_type_;
  uint32_t queue_id_;
  std::string queue_name_;
};

REGISTER_KERNEL_BUILDER(Name("HostQueueDataset").Device(DEVICE_CPU), HostQueueDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
