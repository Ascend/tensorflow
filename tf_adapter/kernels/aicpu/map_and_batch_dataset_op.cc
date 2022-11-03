/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

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
#include "map_and_batch_dataset_op.h"

#include <atomic>
#include <utility>
#include <algorithm>
#include <securec.h>
#include <securectype.h>

#include "graph/types.h"
#include "graph/tensor.h"
#include "runtime/dev.h"
#include "runtime/mem.h"

#include "stream_pool.h"
#include "npu_tensor.h"
#include "dataset_function.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/infershape_util.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/metrics.h"
#if defined(TF_VERSION_TF2)
#include "tensorflow/core/data/name_utils.h"
#endif
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/collective.h"

#include "tensorflow/core/graph/node_builder.h"

#include "acl/acl_mdl.h"

namespace tensorflow {
namespace data {

/* static */
constexpr const char* const NpuMapAndBatchDatasetOp::kDatasetType;
constexpr const char* const NpuMapAndBatchDatasetOp::kInputDataset;
constexpr const char* const NpuMapAndBatchDatasetOp::kOtherArguments;
constexpr const char* const NpuMapAndBatchDatasetOp::kBatchSize;
constexpr const char* const NpuMapAndBatchDatasetOp::kNumParallelCalls;
constexpr const char* const NpuMapAndBatchDatasetOp::kDropRemainder;
constexpr const char* const NpuMapAndBatchDatasetOp::kFunc;
constexpr const char* const NpuMapAndBatchDatasetOp::kTarguments;
constexpr const char* const NpuMapAndBatchDatasetOp::kOutputTypes;
constexpr const char* const NpuMapAndBatchDatasetOp::kOutputShapes;
constexpr const char* const NpuMapAndBatchDatasetOp::kPreserveCardinality;
constexpr const char* const NpuMapAndBatchDatasetOp::kOutputDevice;

// Maximum number of batch results to buffer.

namespace {
constexpr int64 kMicrosToMillis = 1000;
constexpr int64_t kSleepUs = 10;
constexpr int64_t kMaxBatchSize = 2;
constexpr char kParallelism[] = "parallelism";
constexpr char kCallCounter[] = "call_counter";
constexpr char kBatchResultsSize[] = "batch_results_size";
constexpr char kNpuDataMapAndBatch[] = "npu_data_map_and_batch";
constexpr char kBatchResults[] = "batch_results";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kNumCalls[] = "num_calls";
constexpr char kNumElements[] = "num_elements";
constexpr char kOutputAllocated[] = "output_allocated";
constexpr char kStatus[] = "status";
constexpr char kNpu[] = "npu";
constexpr char kCpu[] = "cpu";
constexpr uint64_t INVALID_INDEX = UINT64_MAX;

// Computes ceil(x / y).
inline uint64_t CeilDiv(uint64_t x, uint64_t y) {
  return (y == 0) ? 1 : ((x + y - 1) / y);
}

enum class DataStatus {
  WAIT_WRITE,
  WRITING,
  WAIT_RESULT,
  WAIT_READ,
  READING
};

std::ostream& operator << (std::ostream &os, DataStatus status) {
  static std::map<DataStatus, std::string> data_status_map = {
      {DataStatus::WAIT_WRITE, "WAIT_WRITE"},
      {DataStatus::WRITING, "WRITING"},
      {DataStatus::WAIT_RESULT, "WAIT_RESULT"},
      {DataStatus::WAIT_READ, "WAIT_READ"},
      {DataStatus::READING, "READING"}
  };

  return os << data_status_map[status];
}
}  // namespace

class NpuMapAndBatchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, uint64_t batch_size,
          int64 num_parallel_calls, bool drop_remainder,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& batch_output_shapes,
          const std::vector<PartialTensorShape>& output_shapes,
          std::unique_ptr<CapturedFunction> captured_func,
          bool preserve_cardinality,
          const std::string output_device,
          const std::map<std::string, std::string> &sess_options,
          const std::map<std::string, std::string> &init_options,
          const std::vector<std::pair<std::string, AttrValue>> &attrs)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        batch_size_(batch_size),
        num_parallel_calls_(static_cast<uint64_t>(num_parallel_calls)),
        drop_remainder_(drop_remainder),
        output_types_(output_types),
        batch_output_shapes_(batch_output_shapes),
        output_shapes_(output_shapes),
        captured_func_(std::move(captured_func)),
        preserve_cardinality_(preserve_cardinality),
        output_device_(output_device),
        sess_options_(sess_options),
        init_options_(init_options),
        attrs_(attrs)
#if defined(TF_VERSION_TF2)
        , traceme_metadata_(
            {{"autotune",
              num_parallel_calls == model::kAutotune ? "true" : "false"},
             {"batch_size",
              strings::Printf("%lld", static_cast<long long>(batch_size))},
             {"drop_remainder", drop_remainder ? "true" : "false"}})
#endif
  {
    ADP_LOG(EVENT) << "Dataset construct start.";
    input_->Ref();
    ADP_LOG(EVENT) << "Dataset construct finish.";
  }

  ~Dataset() override {
    ADP_LOG(EVENT) << "~Dataset start.";
    (void)input_->Unref();
    ADP_LOG(EVENT) << "~Dataset finish.";
  }

  bool IsStaticShape() const {
    return !DatasetFunction::HaveUnknowShape(output_shapes_);
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
#if defined(TF_VERSION_TF2)
    std::string prefix_para = name_utils::IteratorPrefix(kDatasetType, prefix);
#else
    std::string prefix_para = prefix + "::" + kDatasetType;
#endif
    if (IsStaticShape()) {
      ADP_LOG(INFO) << "NpuMapAndBatchDatasetOp::MakeIteratorInternal, IteratorStatic "
        << output_device_;
      if (output_device_ == kCpu) {
        return absl::make_unique<IteratorStaticCpu>(IteratorStaticCpu::Params{
            this, prefix_para});
      } else {
        return absl::make_unique<IteratorStaticNpu>(IteratorStaticNpu::Params{
            this, prefix_para});
      }
    } else {
      ADP_LOG(INFO) << "NpuMapAndBatchDatasetOp::MakeIteratorInternal, IteratorDyn "
        << output_device_;
      if (output_device_ == kCpu) {
        return absl::make_unique<IteratorDynCpu>(IteratorDynCpu::Params{
            this, prefix_para});
      } else {
        return absl::make_unique<IteratorDynNpu>(IteratorDynNpu::Params{
            this, prefix_para});
      }
    }
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return batch_output_shapes_;
  }

  string DebugString() const override {
#if defined(TF_VERSION_TF2)
    return name_utils::DatasetDebugString(kDatasetType);
#else
    return "NpuMapAndBatchDatasetOp::DataSet";
#endif
  }

  int64 Cardinality() const override {
    if (!preserve_cardinality_) {
      return kUnknownCardinality;
    }
    int64 n = input_->Cardinality();
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return n / static_cast<int64>(batch_size_) +
      ((n % static_cast<int64>(batch_size_) == 0 || drop_remainder_) ? 0 : 1);
  }
#if defined(TF_VERSION_TF2)
  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }
#endif
  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* batch_size_node;
    TF_RETURN_IF_ERROR(b->AddScalar(static_cast<int64>(batch_size_), &batch_size_node));
    Node* num_parallel_calls_node;
    TF_RETURN_IF_ERROR(
        b->AddScalar(static_cast<int64>(num_parallel_calls_), &num_parallel_calls_node));
    Node* drop_remainder_node;
    TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder_node));
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));

    std::vector<std::pair<StringPiece, AttrValue>> attrs;
    for (auto attr : attrs_) {
      attrs.emplace_back(attr.first, attr.second);
    }
    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {std::make_pair(0, input_graph_node),
         std::make_pair(2, batch_size_node),
         std::make_pair(3, num_parallel_calls_node),
         std::make_pair(4, drop_remainder_node)},  // Single tensor inputs.
        {std::make_pair(1, other_arguments)},      // Tensor list inputs.
        attrs,  // Attrs
        output));
    return Status::OK();
  }

 private:
  class IteratorMeBase : public DatasetIterator<Dataset> {
    public:
      explicit IteratorMeBase(const Params& params)
        : DatasetIterator<Dataset>(params),
          mu_(std::make_shared<mutex>()),
          cond_var_(std::make_shared<condition_variable>()),
          num_parallel_calls_(std::make_shared<model::SharedState>(
              params.dataset->num_parallel_calls_, mu_, cond_var_)),
          output_tensor_num_(dataset()->output_dtypes().size()),
          func_(dataset()->init_options_, dataset()->captured_func_->func().name(),
              dataset()->input_->output_dtypes(), dataset()->output_dtypes(),
              dataset()->input_->output_shapes(), dataset()->output_shapes()) {
        ADP_LOG(EVENT) << "IteratorMeBase construct start.";
        Status status = GetEnvDeviceID(device_id_);
        if (!status.ok()) {
          ADP_LOG(ERROR) << "GetEnvDeviceID failed: rt = " << status.ToString()
                         << "device_id_ = " << device_id_;
        }
        timestat = std::make_shared<TimeStatistic>(static_cast<int64_t>(GetParallelCallsNum()));
        ADP_LOG(EVENT) << "IteratorMeBase construct finish.";
      }

      virtual ~IteratorMeBase() {
        ADP_LOG(EVENT) << "~Dataset::IteratorMeBase start.";
        timestat->ShowTimeStatistic();
        ADP_LOG(EVENT) << "~Dataset::IteratorMeBase finish.";
      }

      uint64_t GetParallelCallsNum() const {
        uint64_t threads = static_cast<uint64_t>((num_parallel_calls_->value <= 0) && (ctx_ != nullptr) ?
            ctx_->runner_threadpool_size() :
            num_parallel_calls_->value);
        return StreamPool::CheckStreamNum(threads);
      }

      Status Initialize(IteratorContext* ctx) override {
        int64 startTime = InferShapeUtil::GetCurrentTimestap();
        mutex_lock l(*mu_);
        TF_RETURN_IF_ERROR(DatasetFunction::RegisterNpuCancellation(
            [this]() { CancelThreads(/*wait=*/ true); }, &deregister_fn_));

        IteratorContext::Params params(ctx);
#if defined(TF_VERSION_TF2)
        cancellation_manager_ = absl::make_unique<CancellationManager>();
        params.cancellation_manager = cancellation_manager_.get();
        TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(
            IteratorContext(params), this, prefix(), &input_impl_));
#else
        TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(
            IteratorContext(params), prefix(), &input_impl_));
#endif
        Status status = func_.Initialize(dataset()->sess_options_,
            const_cast<FunctionLibraryDefinition &>(*dataset()->captured_func_->lib_def()));
        int64 endTime = InferShapeUtil::GetCurrentTimestap();
        ADP_LOG(EVENT) << "[MapAndBatchDatasetOp] Initialize finish, cost: "
                       << " [" << ((endTime - startTime) / kMicrosToMillis) << " ms]";
        return status;
      }

      Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      timestat->RecordStartTime(timestat->statis);
      {
        {
          mutex_lock l(*mu_);
          *end_of_sequence = false;
          Status status = EnsureRunnerThreadStarted(*ctx);
          // 1. No data, return ok, end_of_sequence is false;
          // 2. get first data failed, return status;
          if (!status.ok()) {
            *end_of_sequence = end_of_input_;
            return status;
          }
          while (!cancelled_ && (!(batch_results_[read_index_]->data_status == DataStatus::WAIT_READ))) {
            if ((end_of_input_) && (batch_results_[read_index_]->data_status == DataStatus::WAIT_WRITE)) {
              *end_of_sequence = true;
              return Status::OK();
            }
            ++waiting_;
            RecordStop(ctx);
            cond_var_->wait(l);
            RecordStart(ctx);
            --waiting_;
          }
          if (cancelled_) {
            return errors::Cancelled("IteratorStatic was cancelled");
          }
        }
      }

      uint64_t result_index = read_index_;
      read_index_++;
      if (read_index_ >= max_batch_results_) {
        read_index_ = 0;
      }

      Status status = ProcessBatch(batch_results_[result_index], *out_tensors, *end_of_sequence);
      timestat->RecordEndTime(timestat->statis);
      return status;
    }

    protected:
      class BatchResultBase {
       public:
        explicit BatchResultBase(int64_t batch_id_, uint64_t batch_size_)
            : batch_id(batch_id_),
              status(Status::OK()),
              batch_size(batch_size_) {};
        virtual ~BatchResultBase() {
          ADP_LOG(EVENT) << "~BatchResultBase start.";
          if (output != nullptr) {
            rtError_t rt = rtFree(output);
            if (rt != RT_ERROR_NONE) {
              ADP_LOG(ERROR) << "RT FREE failed, rtError_t = " << rt;
            }
            output = nullptr;
          }

          if (output_cpu != nullptr) {
            delete[] output_cpu;
            output_cpu = nullptr;
          }
          ADP_LOG(EVENT) << "~BatchResultBase finish.";
        }

        void InitOutputs(uint8_t *start_addr, std::vector<uint64_t> &tensor_align_size,
            std::vector<uint8_t*> &outputs_) const {
          uint64_t dim_num = tensor_align_size.size();
          uint64_t offset = 0;
          uint8_t *align_addr = start_addr;
          for (uint64_t i = 0; i < dim_num; i++) {
            outputs_.push_back(align_addr + offset);
            offset += tensor_align_size[i] * batch_size;
          }
        }

        uint64_t GetNextOffset() {
          if (data_status == DataStatus::WAIT_WRITE) {
            num_run = 1;
            (num_run < batch_size) ? UpdateState(DataStatus::WRITING) : UpdateState(DataStatus::WAIT_RESULT);
            return 0;
          }

          if (data_status == DataStatus::WRITING) {
            if (num_run >= batch_size) {
              return INVALID_INDEX;
            }
            uint64_t batch_offset = num_run;
            num_run++;
            if (num_run >= batch_size) {
              UpdateState(DataStatus::WAIT_RESULT);
            }
            return batch_offset;
          }

          return INVALID_INDEX;
        }

        void UpdateState(DataStatus new_status) {
          act(batch_id, data_status, new_status);
          data_status = new_status;
        }

        void UpdateStatus(const Status& s, uint64_t offset) LOCKS_EXCLUDED(mu) {
          mutex_lock l(mu);
          num_recv++;
          if (TF_PREDICT_TRUE(s.ok())) {
            num_ok++;
          } else {
            UpdateState(DataStatus::WAIT_RESULT);
            ADP_LOG(INFO) << "UpdateStatus::Batch result status change, batch_id = " << batch_id
                          << ", data status => " << data_status;
            if (status.ok() || offset < status_offset) {
              status = s;
              status_offset = offset;
            }
          }

          if ((data_status == DataStatus::WAIT_RESULT) && (num_recv >= num_run)) {
            UpdateState(DataStatus::WAIT_READ);
            ADP_LOG(INFO) << "UpdateStatus::Batch result status change, batch_id = " << batch_id
                          << ", data status => " << data_status
                          << ", num_recv = " << num_recv
                          << "num_run = " << num_run;
          }
        }

        void EndofInputUpdateStatus() LOCKS_EXCLUDED(mu) {
          mutex_lock l(mu);
          num_recv++;
          if (num_recv >= num_run) {
            UpdateState(DataStatus::WAIT_READ);
          } else {
            UpdateState(DataStatus::WAIT_RESULT);
          }
          ADP_LOG(INFO) << "EndofInput::Batch result status change, batch_id = " << batch_id
                        << ", data status => " << data_status << ", num_recv = " << num_recv
                        << "num_run = " << num_run;
        }

        void Clear() {
          status = Status::OK();
          status_offset = INVALID_INDEX;
          UpdateState(DataStatus::WAIT_WRITE);
          num_run = 0;
          num_recv = 0;
          num_ok = 0;
        }

        uint64_t GetNumElements() const {
          return (status_offset == INVALID_INDEX) ? num_ok : status_offset;
        }

        void SetAction(const std::function<void(uint64_t, DataStatus, DataStatus)> act_) {
          act = act_;
        }

        mutex mu;
        uint64_t batch_id GUARDED_BY(mu) = INVALID_INDEX;
        Status status;
        uint64_t status_offset = INVALID_INDEX;
        DataStatus data_status = DataStatus::WAIT_WRITE;
        uint64_t batch_size;
        uint64_t num_run GUARDED_BY(mu) = 0;
        // Counts the number of outstanding calls for this batch.
        uint64_t num_recv GUARDED_BY(mu) = 0;
        uint64_t num_ok GUARDED_BY(mu) = 0;

        uint8_t *output = nullptr;
        std::vector<uint8_t*> outputs;

        uint8_t *output_cpu = nullptr;
        std::vector<uint8_t*> outputs_cpu;
        std::function<void(uint64_t, DataStatus, DataStatus)> act;
      };

      virtual uint64_t MapFunc(const std::shared_ptr<IteratorContext>& ctx, uint64_t thread_id,
          DatasetFunction::ModelId model_id, uint64_t batch_id, uint64_t batch_offset, std::vector<Tensor> &input) = 0;
      virtual Status InitBatchResult() = 0;
      virtual void DestroyOutputDataset(BatchResultBase &batch_result) = 0;
      virtual uint8_t* GetStartAddr(BatchResultBase &batch_result) = 0;
      virtual NpuAllocator* CreateAllocator(BatchResultBase &batch_result,
          uint64_t step, std::function<void(void *)> del) = 0;

      virtual bool HasRunRes(uint64_t thread_id) const {
        (void)thread_id;
        return true;
      }

      virtual uint64_t WaitRunRes(const std::shared_ptr<IteratorContext>& ctx, uint64_t thread_id) {
        (void)ctx;
        (void)thread_id;
        return 0;
      }

      void InitBatchResultAction(BatchResultBase &batch_result) {
        batch_result.SetAction([this](uint64_t batch_id, DataStatus old_status, DataStatus new_status) {
          if (batch_id == INVALID_INDEX) {
            return;
          }
          if ((old_status <= DataStatus::WRITING) && (new_status > DataStatus::WRITING)) {
            (void)this->write_index_++;
          }
        });
      }

      void Finalize() noexcept {
        ADP_LOG(INFO) << "~Finalize start.";
        CancelThreads(true);
        if (deregister_fn_) {
          deregister_fn_();
        }

        delete[] batch_results_;
        batch_results_ = nullptr;
        ADP_LOG(INFO) << "~Finalize finish.";
      }

      Status InitBatchResultNpuMem(BatchResultBase &batch_result) {
        // use ACL_MEM_MALLOC_HUGE_FIRST policy for higher performance
        aclError rt = aclrtMalloc(reinterpret_cast<void **>(&batch_result.output),
            batch_mem_size_, ACL_MEM_MALLOC_HUGE_FIRST);
        DATASET_REQUIRES(rt == ACL_SUCCESS, errors::InvalidArgument(
            "Alloc mem failed: ", batch_mem_size_, " aclError: ", rt));
        ADP_LOG(INFO) << "Total reused device memory is " << batch_mem_size_ << " Bytes.";
        batch_result.InitOutputs(batch_result.output, func_tensor_align_size_, batch_result.outputs);
        return Status::OK();
      }

      void InitBatchResultCpuMem(BatchResultBase &batch_result) {
        batch_result.output_cpu = new (std::nothrow)uint8_t[batch_mem_size_];
        if (batch_result.output_cpu != nullptr) {
          batch_result.InitOutputs(batch_result.output_cpu, func_tensor_align_size_, batch_result.outputs_cpu);
        } else {
          ADP_LOG(ERROR) << "InitBatchResultCpuMem, new [" << batch_mem_size_ << "], failed";
        }
      }

      Status ProcessBatch(const std::shared_ptr<BatchResultBase> &batch_result,
          std::vector<Tensor> &out_tensors, bool &end_of_sequence) LOCKS_EXCLUDED(mu) {
        mutex_lock l(batch_result->mu);
        batch_result->UpdateState(DataStatus::READING);

        if (!batch_result->status.ok() && !errors::IsOutOfRange(batch_result->status)) {
          return batch_result->status;
        }

        if ((batch_result->num_ok < batch_result->batch_size) && (dataset()->drop_remainder_)) {
          end_of_sequence = true;
          return Status::OK();
        }

        if ((!batch_result->status.ok() && batch_result->status_offset == 0) ||
            (batch_result->status.ok() && batch_result->num_ok == 0)) {
          end_of_sequence = true;
          return Status::OK();
        }

        return TransBatchResultToTensor(batch_result, out_tensors);
      }

      uint64_t WaitingResultNum() const {
        uint64_t count = 0;
        for (uint64_t i = 0; i < max_batch_results_; i++) {
          if (batch_results_[i]->data_status == DataStatus::WAIT_RESULT) {
            count++;
          }
        }
        return count;
      }

      void CancelThreads(bool wait) LOCKS_EXCLUDED(*mu_) {
        ADP_LOG(INFO) << "CancelThreads start. wait = " << wait;
#if defined(TF_VERSION_TF2)
        cancellation_manager_->StartCancel();
#endif
        {
          mutex_lock l(*mu_);
          if (runner_threads_.size() > 0) {
            cancelled_ = true;
            StopBatchResultWritingStatus();
            cond_var_->notify_all();
          }
        }
        while (wait && thread_num_ > 0) {
          cond_var_->notify_all();
          (void)usleep(kSleepUs);
        }
        ADP_LOG(INFO) << "CancelThreads finish.";
      }

      void CallCompleted() LOCKS_EXCLUDED(*mu_) {
        mutex_lock l(*mu_);
        cond_var_->notify_all();
      }

      void CallCompleted(uint64_t thread_id, std::shared_ptr<Items> &it) LOCKS_EXCLUDED(*mu_) {
        mutex_lock l(*mu_);
        timestat->UpdateWithTimeTag(timestat->statis_threads_ge[thread_id], it);
        cond_var_->notify_all();
      }

      void StopBatchResultWritingStatus() {
        for (uint64_t i = 0; i < max_batch_results_; i++) {
          mutex_lock l(batch_results_[i]->mu);
          if (batch_results_[i]->data_status == DataStatus::WRITING) {
            if (batch_results_[i]->num_recv < batch_results_[i]->num_run) {
              batch_results_[i]->UpdateState(DataStatus::WAIT_RESULT);
            } else {
              batch_results_[i]->UpdateState(DataStatus::WAIT_WRITE);
              batch_results_[(write_index_.load() % max_batch_results_)]->num_run = 0;
            }
          }
        }
      }

      bool GetNextBatchIndex(uint64_t &batch_id, uint64_t &batch_offset) EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        uint64_t idx = write_index_.load() % max_batch_results_;
        if (batch_results_[idx]->data_status > DataStatus::WAIT_RESULT) {
          return false;
        }
        batch_offset = batch_results_[idx]->GetNextOffset();
        if (batch_offset == INVALID_INDEX) {
          return false;
        }
        batch_id = idx;
        return true;
      }

      void RunnerThread(const std::shared_ptr<IteratorContext>& ctx, uint64_t thread_id) LOCKS_EXCLUDED(*mu_) {
        rtError_t rt = rtSetDevice(static_cast<int32_t>(device_id_));
        if (rt != ACL_RT_SUCCESS) {
          ADP_LOG(ERROR) << "Thread rtSetDevice failed: thread_id = " << thread_id
            << "thread_num = " << this->thread_num_ << "device_id_ = " << device_id_ << " rt=" << rt;
          return;
        }

        DatasetFunction::ModelId model_id;
        Status status = func_.LoadGeModelFromMem(model_id);
        if (!status.ok()) {
          ADP_LOG(ERROR) << "DatasetFunction Instantialte failed, status = " << status.ToString();
          return;
        }

        {
          mutex_lock l(*mu_);
          thread_num_++;
          RecordStart(ctx.get());
        }

        auto stop_cleanup = gtl::MakeCleanup([this, &ctx, thread_id]() {
          mutex_lock l(*this->mu_);
          RecordStop(ctx.get());
          this->thread_num_--;
          rtError_t rt = rtDeviceReset(static_cast<int32_t>(device_id_));
          if (rt != RT_ERROR_NONE) {
            ADP_LOG(ERROR) << "Call reset device failed. device id=" << device_id_ << " rt=" << rt;
          }
          ADP_LOG(INFO) << "Thread exit: thread_id = " << thread_id
                        << " thread_num = " << this->thread_num_;
        });

        uint64_t run_res = 0;
        while (!cancelled_) {
          // Implementation class to implement
          // if no run res, need to wait run res
          // if end of input, need to wait run end
          while (!HasRunRes(thread_id) || (end_of_input_ && (run_res > 0))) {
            run_res = WaitRunRes(ctx, thread_id);
          }

          uint64_t batch_id;
          uint64_t batch_offset;
          bool map_func = false;
          std::vector<Tensor> in_tensors;
          {
            mutex_lock l(*mu_);
            //  if tf restore the data status, this will continue;
            if (!end_of_input_ && GetNextBatchIndex(batch_id, batch_offset)) {
              status = input_impl_->GetNext(ctx.get(), &in_tensors, &end_of_input_);
              if (status.ok() && !end_of_input_) {
                map_func = true;
              } else {
                if (!status.ok()) {
                  batch_results_[batch_id]->UpdateStatus(status, batch_offset);
                } else {
                  batch_results_[batch_id]->EndofInputUpdateStatus();
                  ADP_LOG(INFO) << "End of sequence in thread " << thread_id;
                }
                cond_var_->notify_all();
              }
            } else {
              if (run_res > 0) {
                // to wait run complete
              } else {
                RecordStop(ctx.get());
                cond_var_->wait(l);
                RecordStart(ctx.get());
                continue;
              }
            }
          }
          if (map_func) {
            timestat->RecordStartTime(timestat->statis_threads[thread_id]);
            run_res = MapFunc(ctx, thread_id, model_id, batch_id, batch_offset, in_tensors);
            timestat->RecordEndTime(timestat->statis_threads[thread_id]);
          } else {
            run_res = WaitRunRes(ctx, thread_id);
          }
        }
      }

      Status EnsureRunnerThreadStarted(IteratorContext &ctx) EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
        if (runner_threads_.size() == 0) {
          rtError_t rt = rtSetDevice(static_cast<int32_t>(device_id_));
          if (rt != ACL_RT_SUCCESS) {
            ADP_LOG(ERROR) << "Main thread rtSetDevice failed: device_id_ = " << device_id_;
          }

          Status status = InitBatchResult();
          if (!status.ok()) {
            return status;
          }

          ctx_ = std::make_shared<IteratorContext>(ctx);
          for (uint64_t i = 0; i < GetParallelCallsNum(); i++) {
            runner_threads_.emplace_back(std::move(ctx.StartThread(
                kNpuDataMapAndBatch + std::to_string(i),
                std::bind(&IteratorMeBase::RunnerThread, this, ctx_, i))));
          }
        }
        return Status::OK();
      }

      Status TransBatchResultToTensor(const std::shared_ptr<BatchResultBase> &batch_result,
          std::vector<Tensor> &out_tensors) {
        out_tensors.clear();
        std::shared_ptr<uint8_t> npu_addr(GetStartAddr(*batch_result), [this, batch_result](uint8_t *) {
            batch_result->Clear();
            this->cond_var_->notify_all();
          });
        DATASET_REQUIRES((npu_addr != nullptr),
            errors::InvalidArgument("Alloc mem failed: mem_size = ", batch_mem_size_));
        for (uint64_t i = 0; i < output_tensor_num_; i++) {
          out_tensors.push_back(BuildTensorByMem(i, npu_addr, *batch_result.get()));
        }
        DestroyOutputDataset(*batch_result);
        return Status::OK();
      }

      Tensor BuildTensorByMem(uint64_t tensor_id, std::shared_ptr<uint8_t> &npu_addr, BatchResultBase &batch_result) {
        NpuAllocator *npu_allocator = CreateAllocator(batch_result, tensor_id,
            [npu_addr](const void *addr) mutable {
              npu_addr.reset();
              (void)addr;
            });
        TensorShape shape = {static_cast<int64_t>(batch_result.GetNumElements())};
        (void)std::for_each(func_output_shape_[tensor_id].cbegin(), func_output_shape_[tensor_id].cend(),
            [&shape](int64_t i) { shape.AddDim(static_cast<int64>(i)); });
        return Tensor(npu_allocator, dataset()->output_types_[tensor_id], shape);
      }

      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeAsyncKnownRatioNode(
            std::move(args), dataset()->batch_size_,
            {model::MakeParameter(kParallelism, num_parallel_calls_, /*min=*/1,
                                  /*max=*/ctx->runner_threadpool_size())});
      }
#if defined(TF_VERSION_TF2)
      TraceMeMetadata GetTraceMeMetadata() const override {
        int64 parallelism = -1;
        int64 max_batch_results = -1;
        // NOTE: We only set the parallelism value if the lock can be acquired
        // right away to avoid introducing tracing overhead.
        if (mu_->try_lock()) {
          parallelism = static_cast<int64>(GetParallelCallsNum());
          max_batch_results = static_cast<int64>(max_batch_results_);
          mu_->unlock();
        }
        auto result = dataset()->traceme_metadata_;
        result.push_back(std::make_pair(
            "max_batch_results",
            strings::Printf("%lld", static_cast<long long>(max_batch_results))));
        result.push_back(std::make_pair(
            "parallelism",
            parallelism == -1
                ? kTraceInfoUnavailable
                : strings::Printf("%lld", static_cast<long long>(parallelism))));
        return result;
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        return Status::OK();
      }
#else
      string BuildTraceMeName() override {
        return prefix() + "#parallelism=" + std::to_string(num_parallel_calls_->value) + "#";
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        return Status::OK();
      }
#endif

      Status RestoreInternal(IteratorContext* ctx,
                            IteratorStateReader* reader) override {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        return Status::OK();
      }

      const std::shared_ptr<mutex> mu_;
      std::shared_ptr<IteratorContext> ctx_;
      std::vector<std::unique_ptr<Thread>> runner_threads_ GUARDED_BY(*mu_);
      const std::shared_ptr<condition_variable> cond_var_;
      std::unique_ptr<IteratorBase> input_impl_;
      bool cancelled_ GUARDED_BY(*mu_) = false;
      int64 waiting_ GUARDED_BY(*mu_) = 0;
#ifdef TF_VERSION_TF2
      std::unique_ptr<CancellationManager> cancellation_manager_;
#endif
      std::function<void()> deregister_fn_;

      const std::shared_ptr<model::SharedState> num_parallel_calls_;
      uint64_t max_batch_results_ GUARDED_BY(*mu_) = 0;
      bool end_of_input_ GUARDED_BY(*mu_) = false;
      std::vector<std::vector<int64_t>> func_output_shape_;
      uint64_t read_index_ = 0;
      std::atomic<uint64_t> write_index_{0};
      std::shared_ptr<BatchResultBase> *batch_results_ GUARDED_BY(*mu_) = nullptr;

      uint64_t output_tensor_num_;
      std::vector<uint64_t> func_tensor_size_;
      std::vector<uint64_t> func_tensor_align_size_;
      uint64_t batch_mem_size_ = 0;
      DatasetFunction func_;
      uint32_t device_id_ = 0;
      int64_t thread_num_ GUARDED_BY(*mu_) = 0;
      std::shared_ptr<TimeStatistic> timestat = nullptr;
  };

  class IteratorStatic : public IteratorMeBase {
   public:
    explicit IteratorStatic(const Params& params)
        : IteratorMeBase(params) {
      ADP_LOG(EVENT) << "IteratorStatic";
      max_batch_results_ = CeilDiv(GetParallelCallsNum() * kMaxTask, params.dataset->batch_size_);
      if (max_batch_results_ <= 1) {
        max_batch_results_ = kMaxBatchSize;
      }
      max_batch_results_ += 1;
    }

    ~IteratorStatic() override {
      ADP_LOG(EVENT) << "~IteratorStatic.";
    }

   protected:
    class BatchStaticResult : public BatchResultBase {
     public:
      explicit BatchStaticResult(uint64_t batch_id_, uint64_t batch_size_)
          : BatchResultBase(batch_id_, batch_size_) {
        ADP_LOG(EVENT) << "BatchStaticResult batch_id = " << batch_id;
      }
      ~BatchStaticResult() override {
        ADP_LOG(EVENT) << "~BatchStaticResult.";
      }
    };

    static constexpr uint64_t kMaxTask = 2;
    Status InitBatchResult() override {
      uint64_t dim_num = dataset()->output_types_.size();
      batch_mem_size_ = 0;
      for (uint64_t i = 0; i < dim_num; i++) {
        std::vector<int64_t> output_shape = DatasetFunction::GetTfShapeDims(dataset()->output_shapes_[i]);
        func_output_shape_.push_back(output_shape);
        int64_t shape_size = DatasetFunction::GetShapeDims(output_shape);
        DATASET_REQUIRES(shape_size > 0,
            errors::Unavailable("tensor is too small"));
        int64_t type_size = DataTypeSize(dataset()->output_types_[i]);
        DATASET_REQUIRES(!DatasetFunction::CheckMultiplyOverflow(shape_size, type_size),
            errors::Unavailable("tensor is too big, shape_size = ", shape_size,
                                ", type = ", dataset()->output_types_[i]));
        uint64_t tensor_size = static_cast<uint64_t>(shape_size * type_size);
        func_tensor_size_.push_back(tensor_size);
        func_tensor_align_size_.push_back(tensor_size);
        DATASET_REQUIRES(!DatasetFunction::CheckAddOverflow(batch_mem_size_, tensor_size),
            errors::Unavailable("batch_mem_size_ is too big, batch_mem_size_ = ",
                batch_mem_size_, ", tensor_size = ", tensor_size));
        batch_mem_size_ += tensor_size;
      }
      DATASET_REQUIRES(!DatasetFunction::CheckMultiplyOverflow(batch_mem_size_, dataset()->batch_size_),
          errors::Unavailable("batch results memory is too big, batch_mem_size_ = ", batch_mem_size_,
              ", batch_size_ = ", dataset()->batch_size_));
      batch_mem_size_ *= dataset()->batch_size_;
      ADP_LOG(INFO) << "BatchMem, batch_mem_size : " << batch_mem_size_;

      stream_pool_.reset(new (std::nothrow)StreamPool(GetParallelCallsNum(), static_cast<int>(kMaxTask)));
      DATASET_REQUIRES(stream_pool_ != nullptr, errors::Unavailable("create stream pool failed."));

      return InitBatchResultMem();
    }

    bool HasRunRes(uint64_t thread_id) const override {
      return stream_pool_->GetIdleEventCount(thread_id) > 0;
    }

    uint64_t MapFunc(const std::shared_ptr<IteratorContext>& ctx, uint64_t thread_id, DatasetFunction::ModelId model_id,
        uint64_t batch_id, uint64_t batch_offset, std::vector<Tensor> &input) override {
      (void)ctx;
      aclmdlDataset *input_dataset = nullptr;
      aclmdlDataset *output_dataset = nullptr;

      input_dataset = DatasetFunction::CreateAclInputDatasetWithTFTensors(input);
      output_dataset = InitOutTensorsMem(*static_cast<BatchStaticResult*>(batch_results_[batch_id].get()),
                                         batch_offset);

      std::shared_ptr<Items> time_tag = std::make_shared<Items>();
      BatchResultBase *batch_result = batch_results_[batch_id].get();
      auto done = [this, batch_result, batch_offset, input_dataset, output_dataset,
          thread_id, time_tag](const Status &status) mutable {
        batch_result->UpdateStatus(status, batch_offset);
        this->CallCompleted(thread_id, time_tag);
        // free input_dataset by current dataset op
        DatasetFunction::DestroyAclOutputDataset(output_dataset);
        DatasetFunction::DestroyAclInputDataset(input_dataset);
      };

      if (input_dataset == nullptr) {
        done(errors::Unavailable("Create input dataset failed."));
        return stream_pool_->GetWaitingEventCount(thread_id);
      }

      if (output_dataset == nullptr) {
        done(errors::Unavailable("Create output dataset failed."));
        return stream_pool_->GetWaitingEventCount(thread_id);
      }

      std::shared_ptr<Stream> stream = stream_pool_->GetStream(thread_id);
      if (stream == nullptr) {
        done(errors::Unavailable("Get Stream failed"));
        return stream_pool_->GetWaitingEventCount(thread_id);
      }
      timestat->RecordStartTime(*time_tag);
      Status status = func_.RunWithStreamAsyn(model_id, input_dataset, output_dataset, stream->GetStream());
      if (status.ok()) {
        status = stream_pool_->RecordEvent(stream, done);
      }
      if (!status.ok()) {
        ADP_LOG(ERROR) << "RecordEvent Failed, status = " << status.ToString();
        done(status);
      }
      return stream_pool_->GetWaitingEventCount(thread_id);
    }

    uint64_t WaitRunRes(const std::shared_ptr<IteratorContext>& ctx, uint64_t thread_id) override {
      (void)ctx;
      (void)stream_pool_->WaitOneEvent(thread_id);
      return stream_pool_->GetWaitingEventCount(thread_id);
    }

    void DestroyOutputDataset(BatchResultBase &batch_result) override {
      (void)batch_result;
    }

    ge::Format GetFormat() const {
      return ge::Format::FORMAT_ND;
    }

    std::unique_ptr<StreamPool> stream_pool_ = nullptr;
   private:
    aclmdlDataset* InitOutTensorsMem(BatchStaticResult &batch_result, uint64_t step) {
      aclmdlDataset* output_dataset = aclmdlCreateDataset();
      if (output_dataset == nullptr) {
        return nullptr;
      }

      uint64_t tensor_num = func_output_shape_.size();
      for (uint64_t i = 0; i < tensor_num; i++) {
        aclDataBuffer *data_buff = aclCreateDataBuffer(batch_result.outputs[i] + step * func_tensor_size_[i],
                                                       func_tensor_size_[i]);
        aclError ret = aclmdlAddDatasetBuffer(output_dataset, data_buff);
        if (ret != ACL_SUCCESS) {
          // tensor的description不需要添加进去吗？
          (void)aclDestroyDataBuffer(data_buff);
          ADP_LOG(ERROR) << "Add data buffer to output dataset failed.";
          return nullptr;
        }
      }
      return output_dataset;
    }

    Status InitBatchResultMem() {
      batch_results_ = new std::shared_ptr<BatchResultBase>[max_batch_results_]();
      DATASET_REQUIRES(batch_results_ != nullptr,
          errors::InvalidArgument("Make batch results faild."));
      for (uint64_t i = 0; i < max_batch_results_; i++) {
        batch_results_[i].reset(new (std::nothrow)BatchStaticResult(i, dataset()->batch_size_));
        DATASET_REQUIRES(batch_results_[i] != nullptr,
            errors::InvalidArgument("Make batch result faild: i = ", i));
        InitBatchResultAction(*batch_results_[i]);
        Status status = InitBatchResultNpuMem(*batch_results_[i]);
        DATASET_REQUIRES(status.ok(), status);
      }
      return Status::OK();
    }
  };

  class IteratorStaticNpu : public IteratorStatic {
   public:
    explicit IteratorStaticNpu(const Params& params)
        : IteratorStatic(params) {
      ADP_LOG(EVENT) << "IteratorStaticNpu";
    }

    ~IteratorStaticNpu() override {
      Finalize();
      ADP_LOG(EVENT) << "~IteratorStaticNpu finish.";
    }

   protected:
    uint8_t* GetStartAddr(BatchResultBase &batch_result) override {
      return batch_result.output;
    }

    NpuAllocator* CreateAllocator(BatchResultBase &batch_result, uint64_t step,
        std::function<void(void *)> del) override {
      return NpuAllocator::CreateNpuAllocator(batch_result.outputs[step], del);
    }
  };

  class IteratorStaticCpu : public IteratorStatic {
   public:
    explicit IteratorStaticCpu(const Params& params)
        : IteratorStatic(params) {
      ADP_LOG(EVENT) << "IteratorStaticCpu";
    }

    ~IteratorStaticCpu() override {
      Finalize();
      ADP_LOG(EVENT) << "~IteratorStaticCpu finish.";
    }

   protected:
    uint8_t* GetStartAddr(BatchResultBase &batch_result) override {
      if (batch_result.output_cpu == nullptr) {
        InitBatchResultCpuMem(batch_result);
      }

      if (batch_result.output_cpu != nullptr) {
        aclError ret = aclrtMemcpy(batch_result.output_cpu, batch_mem_size_,
            batch_result.output, batch_mem_size_, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
          ADP_LOG(ERROR) << "aclrtMemcpy failed, return " << ret;
          return nullptr;
        }
      }

      return batch_result.output_cpu;
    }

    NpuAllocator* CreateAllocator(BatchResultBase &batch_result, uint64_t step,
        std::function<void(void *)> del) override {
      return  NpuAllocator::CreateCpuAllocator(batch_result.outputs_cpu[step], del);
    }
  };

  class IteratorDyn : public IteratorMeBase {
  public:
    explicit IteratorDyn(const Params& params)
        : IteratorMeBase(params) {
      max_batch_results_ = CeilDiv(GetParallelCallsNum(), params.dataset->batch_size_);
      ADP_LOG(EVENT) << "IteratorDyn.";
    };

    ~IteratorDyn() override {
      ADP_LOG(EVENT) << "~IteratorDyn.";
    }

  protected:
    class BatchDynResult : public BatchResultBase {
     public:
      explicit BatchDynResult(uint64_t batch_id_, uint64_t batch_size_)
          : BatchResultBase(batch_id_, batch_size_) {
        output_datasets.resize(static_cast<size_t>(batch_size_));
      }
      std::vector<aclmdlDataset*> output_datasets;
    };

    Status InitTensorSize(BatchDynResult &batch_result) {
      aclmdlDataset *out_dataset = batch_result.output_datasets[0];
      for (size_t idx = 0; idx < aclmdlGetDatasetNumBuffers(out_dataset); idx++) {
        // get output desc and size
        aclTensorDesc *out_desc = aclmdlGetDatasetTensorDesc(out_dataset, idx);
        DATASET_REQUIRES(out_desc != nullptr, errors::Internal("Get aclTensorDesc failed."));

        uint64_t tensor_size = static_cast<uint64_t>(aclGetTensorDescSize(out_desc));
        DATASET_REQUIRES(tensor_size != 0U, errors::Internal("Get tensor_size == 0."));

        func_tensor_size_.push_back(tensor_size);
        func_tensor_align_size_.push_back(tensor_size);
        DATASET_REQUIRES(!DatasetFunction::CheckAddOverflow(batch_mem_size_, tensor_size),
            errors::Unavailable("batch_mem_size_ is too big, batch_mem_size_ = ",
                batch_mem_size_, ", tensor_size = ", tensor_size));
        batch_mem_size_ += tensor_size;

        std::vector<int64_t> dims;
        Status status = DatasetFunction::GetAclTenorDescDims(out_desc, dims);
        DATASET_REQUIRES(status.ok(), status);
        func_output_shape_.emplace_back(std::move(dims));
      }
      DATASET_REQUIRES(!DatasetFunction::CheckMultiplyOverflow(batch_mem_size_, dataset()->batch_size_),
          errors::Unavailable("batch results memory is too big, batch_mem_size_ = ", batch_mem_size_,
              ", batch_size_ = ", dataset()->batch_size_));
      batch_mem_size_ *= dataset()->batch_size_;
      return Status::OK();
    }

    uint64_t MapFunc(const std::shared_ptr<IteratorContext>& ctx, uint64_t thread_id, DatasetFunction::ModelId model_id,
        uint64_t batch_id, uint64_t batch_offset, std::vector<Tensor> &input) override {
      (void)ctx;
      (void)thread_id;
      aclmdlDataset *input_dataset = nullptr;
      input_dataset = DatasetFunction::CreateAclInputDatasetWithTFTensors(input);
      if (input_dataset == nullptr) {
        return 0;
      }

      aclmdlDataset *output_dataset = nullptr;
      output_dataset = DatasetFunction::CreateAclOutputDataset(model_id);
      if (output_dataset == nullptr) {
        return 0;
      }

      BatchDynResult *batch_dyn_result = static_cast<BatchDynResult*>(batch_results_[batch_id].get());

      timestat->RecordStartTime(timestat->statis_threads_ge[thread_id]);
      Status status = func_.Run(model_id, input_dataset, output_dataset);
      timestat->RecordEndTime(timestat->statis_threads_ge[thread_id]);
      if (status.ok()) {
        batch_dyn_result->output_datasets[batch_offset] = std::move(output_dataset);
      }
      batch_dyn_result->UpdateStatus(status, batch_offset);
      CallCompleted();
      DatasetFunction::DestroyAclInputDataset(input_dataset);
      return 0;
    }

    Status InitBatchResult() override {
      batch_results_ = new (std::nothrow)std::shared_ptr<BatchResultBase>[max_batch_results_]();
      DATASET_REQUIRES(batch_results_ != nullptr,
          errors::InvalidArgument("Make batch results faild."));
      for (uint64_t i = 0; i < max_batch_results_; i++) {
        batch_results_[i].reset(new (std::nothrow)BatchDynResult(i, dataset()->batch_size_));
        DATASET_REQUIRES(batch_results_[i] != nullptr,
            errors::InvalidArgument("Make batch result faild: i = ", i));
        InitBatchResultAction(*batch_results_[i]);
      }
      return Status::OK();
    }

    void DestroyOutputDataset(BatchResultBase &batch_result) override {
      BatchDynResult& results = static_cast<BatchDynResult&>(batch_result);
      uint64_t batch_size = results.GetNumElements();
      for (uint64_t j = 0; j < batch_size; j++) {
        aclmdlDataset *out_dataset = results.output_datasets[j];
        DatasetFunction::DestroyAclOutputDataset(out_dataset);
      }
    }

  private:
    ge::Format GetFormat() const {
      return ge::Format::FORMAT_ND;
    }
  };

  class IteratorDynNpu : public IteratorDyn {
   public:
    explicit IteratorDynNpu(const Params& params)
        : IteratorDyn(params) {
    };

    ~IteratorDynNpu() override {
      Finalize();
      ADP_LOG(EVENT) << "~IteratorDynNpu finish.";
    }

   protected:
    uint8_t* GetStartAddr(BatchResultBase &batch_result) override {
      if (batch_mem_size_ == 0ULL) {
        Status status = InitTensorSize(static_cast<BatchDynResult&>(batch_result));
        if (!status.ok()) {
          return nullptr;
        }
      }

      if (batch_result.output == nullptr) {
        (void)InitBatchResultNpuMem(batch_result);
      }

      if (batch_result.output != nullptr) {
        return MemCpyData(static_cast<BatchDynResult&>(batch_result));
      }

      return nullptr;
    }

    NpuAllocator* CreateAllocator(BatchResultBase &batch_result, uint64_t step,
        std::function<void(void *)> del) override {
      return  NpuAllocator::CreateNpuAllocator(batch_result.outputs[step], del);
    }
   private:
    uint8_t *MemCpyData(BatchDynResult &batch_result) const {
      uint64_t tensor_num = batch_result.outputs.size();
      uint64_t batch_size = batch_result.GetNumElements();
      for (uint64_t i = 0; i < tensor_num; i++) {
        uint8_t *npu_addr = batch_result.outputs[i];
        uint64_t tensor_size = func_tensor_size_[i];
        for (uint64_t j = 0; j < batch_size; j++) {
          aclmdlDataset *out_dataset = batch_result.output_datasets[j];
          aclTensorDesc *out_desc = aclmdlGetDatasetTensorDesc(out_dataset, i);
          uint64_t out_tensor_size = static_cast<uint64_t>(aclGetTensorDescSize(out_desc));
          aclDataBuffer *data_buff = aclmdlGetDatasetBuffer(out_dataset, i);
          void* data_addr = aclGetDataBufferAddr(data_buff);
          DATASET_REQUIRES_RETURN_NULL(data_addr != nullptr, errors::Internal("Get data addr is nullptr."));

          aclError ret = aclrtMemcpy(npu_addr, tensor_size, data_addr, out_tensor_size,
              ACL_MEMCPY_DEVICE_TO_DEVICE);
          if (ret != RT_ERROR_NONE) {
            ADP_LOG(ERROR) << "Mem copy from device to device failed, from "
                << out_tensor_size << " to " << tensor_size << " ret: " << ret;
            return nullptr;
          }
          npu_addr += tensor_size;
        }
      }
      return batch_result.output;
    }
  };

  class IteratorDynCpu : public IteratorDyn {
   public:
    explicit IteratorDynCpu(const Params& params)
        : IteratorDyn(params) {
    };

    ~IteratorDynCpu() override {
      Finalize();
      ADP_LOG(EVENT) << "~IteratorDynCpu finish.";
    }
   protected:
    uint8_t* GetStartAddr(BatchResultBase &batch_result) override {
      if (batch_mem_size_ == 0) {
        Status status = InitTensorSize(static_cast<BatchDynResult&>(batch_result));
        if (!status.ok()) {
          return nullptr;
        }
      }

      if (batch_result.output_cpu == nullptr) {
        InitBatchResultCpuMem(batch_result);
      }

      if (batch_result.output_cpu != nullptr) {
        return MemCpyData(static_cast<BatchDynResult&>(batch_result));
      }

      return nullptr;
    }

    NpuAllocator* CreateAllocator(BatchResultBase &batch_result, uint64_t step,
        std::function<void(void *)> del) override {
      return  NpuAllocator::CreateCpuAllocator(batch_result.outputs_cpu[step], del);
    }

    uint8_t *MemCpyData(BatchDynResult &batch_result) const {
      uint64_t tensor_num = batch_result.outputs_cpu.size();
      uint64_t batch_size = batch_result.GetNumElements();
      for (uint64_t i = 0; i < tensor_num; i++) {
        uint8_t *npu_addr = batch_result.outputs_cpu[i];
        uint64_t tensor_size = func_tensor_size_[i];
        for (uint64_t j = 0; j < batch_size; j++) {
          aclmdlDataset *out_dataset = batch_result.output_datasets[j];
          aclTensorDesc *out_desc = aclmdlGetDatasetTensorDesc(out_dataset, i);
          uint64_t out_tensor_size = static_cast<uint64_t>(aclGetTensorDescSize(out_desc));
          aclDataBuffer *data_buff = aclmdlGetDatasetBuffer(out_dataset, i);
          void* data_addr = aclGetDataBufferAddr(data_buff);
          DATASET_REQUIRES_RETURN_NULL(data_addr != nullptr, errors::Internal("Get data addr is nullptr."));

          aclError ret = aclrtMemcpy(npu_addr, tensor_size, data_addr, out_tensor_size,
              ACL_MEMCPY_DEVICE_TO_HOST);
          if (ret != ACL_ERROR_NONE) {
            ADP_LOG(ERROR) << "Mem copy from device to host failed: device size: " << out_tensor_size <<
                " host size: " << tensor_size << "ret: " << ret;
            return nullptr;
          }
          npu_addr += tensor_size;
        }
      }
      return batch_result.output_cpu;
    }
  };

  const DatasetBase* const input_;
  const uint64_t batch_size_;
  const uint64_t num_parallel_calls_;
  const bool drop_remainder_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> batch_output_shapes_;
  const std::vector<PartialTensorShape> output_shapes_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const bool preserve_cardinality_;
  const std::string output_device_;

  const std::map<std::string, std::string> sess_options_;
  const std::map<std::string, std::string> init_options_;
  const std::vector<std::pair<std::string, AttrValue>> attrs_;
#if defined(TF_VERSION_TF2)
  const TraceMeMetadata traceme_metadata_;
#endif
};

Status NpuMapAndBatchDatasetOp::CheckOutputType() {
  for (auto type : output_types_) {
    if (!DataTypeCanUseMemcpy(type)) {
      return errors::InvalidArgument("DT_TYPE is not unspported, DT_TYPE = ", DataTypeString(type));
    }
  }
  return Status::OK();
}

void NpuMapAndBatchDatasetOp::RemoveBatchForOutputShapes(std::vector<PartialTensorShape>& output_shapes) const {
  // remove the first dim for output_shapes
  for (auto& sp : output_shapes) {
    if (sp.dims() == -1) {
      continue;
    }
    sp.RemoveDim(0);
  }
}

NpuMapAndBatchDatasetOp::NpuMapAndBatchDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx),
      sess_options_(NpuAttrs::GetSessOptions(ctx)),
      init_options_(NpuAttrs::GetInitOptions(ctx)) {
  ADP_LOG(EVENT) << "NpuMapAndBatchDatasetOp";
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kFunc, /*params=*/{},
                                               &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));

  // when drop_remainder=False, we still run this graph as static graph.
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  RemoveBatchForOutputShapes(output_shapes_);

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &batch_output_shapes_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kPreserveCardinality, &preserve_cardinality_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kOutputDevice, &output_device_));
  OP_REQUIRES_OK(ctx, CheckOutputType());
  for (auto attr : ctx->def().attr()) {
    attrs_.emplace_back(attr.first, attr.second);
  }
}

void NpuMapAndBatchDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
    DatasetBase** output) {
  ADP_LOG(INFO) << "NpuMapAndBatchDatasetOp:: MakeDataset";
  if (std::any_of(output_types_.cbegin(), output_types_.cend(),
      [](DataType type) { return type == DT_STRING; })) {
    ADP_LOG(ERROR) << "NpuMapAndBatchDatasetOp does not support output type DT_STRING.";
    return;
  }

  int64 batch_size = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kBatchSize, &batch_size));
  OP_REQUIRES(ctx, batch_size > 0,
              errors::InvalidArgument("batch_size must be greater than zero."));

  int64 num_parallel_calls = 0;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kNumParallelCalls, &num_parallel_calls));
  OP_REQUIRES(
      ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutotune,
      errors::InvalidArgument("num_parallel_calls must be greater than zero."));

  bool drop_remainder = false;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument(ctx, kDropRemainder, &drop_remainder));

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  if (num_parallel_calls == model::kAutotune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }

  *output = new(std::nothrow) Dataset(ctx, input, static_cast<uint64_t>(batch_size), num_parallel_calls,
                        drop_remainder, output_types_, batch_output_shapes_, output_shapes_,
                        std::move(captured_func), preserve_cardinality_, output_device_,
                        sess_options_, init_options_, attrs_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("NpuMapAndBatchDataset").Device(DEVICE_CPU),
                        NpuMapAndBatchDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("NpuMapAndBatchDataset");
}  // namespace
}  // namespace data
}  // namespace tensorflow
