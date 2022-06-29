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
#include <atomic>
#include <utility>
#include <algorithm>
#include <securec.h>
#include <securectype.h>
#include <queue>

#include "map_dataset_op.h"
#include "dataset_function.h"
#include "stream_pool.h"
#include "npu_tensor.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "runtime/dev.h"
#include "runtime/mem.h"

#include "tf_adapter/util/npu_attrs.h"

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
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {
namespace data {
/* static */
constexpr const char* const NpuMapDatasetOp::kDatasetType;
constexpr const char* const NpuMapDatasetOp::kInputDataset;
constexpr const char* const NpuMapDatasetOp::kOtherArguments;
constexpr const char* const
    NpuMapDatasetOp::kNumParallelCalls;
constexpr const char* const NpuMapDatasetOp::kFunc;
constexpr const char* const NpuMapDatasetOp::kTarguments;
constexpr const char* const NpuMapDatasetOp::kOutputTypes;
constexpr const char* const NpuMapDatasetOp::kOutputShapes;
constexpr const char* const NpuMapDatasetOp::kOutputDevice;
constexpr const char* const NpuMapDatasetOp::kDeterministic;
constexpr const char* const NpuMapDatasetOp::kSloppy;
constexpr const char* const
    NpuMapDatasetOp::kPreserveCardinality;

namespace {
constexpr int64_t kSleepUs = 10;
constexpr char kParallelism[] = "parallelism";
constexpr char kOutputResultsSize[] = "output_results_size";
constexpr char kOutputResults[] = "output_results";
constexpr char kComponent[] = "component";
constexpr char kNpuMapDataset[] = "npu_map_dataset";
constexpr char kSize[] = "size";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kErrorCode[] = "code";
constexpr char kErrorMessage[] = "error_message";
constexpr char kNpu[] = "npu";
constexpr char kCpu[] = "cpu";
} // namespace

class NpuMapDatasetOp::Dataset : public DatasetBase {
public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          int64_t num_parallel_calls, const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes,
          const std::string output_device,
          bool deterministic,
          std::unique_ptr<CapturedFunction> captured_func,
          bool preserve_cardinality,
          const std::map<std::string, std::string> &sess_options,
          std::map<std::string, std::string> &init_options,
          std::vector<std::pair<StringPiece, AttrValue>> &attrs)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        num_parallel_calls_(num_parallel_calls),
        output_types_(output_types),
        output_shapes_(output_shapes),
        output_device_(output_device),
        deterministic_(deterministic),
        preserve_cardinality_(preserve_cardinality),
        captured_func_(std::move(captured_func)),
        sess_options_(sess_options),
        init_options_(init_options),
        attrs_(attrs)
#if defined(TF_VERSION_TF2)
        , traceme_metadata_(
          {{"autotune",
            num_parallel_calls == model::kAutotune ? "true" : "false"}})
#endif
  {
    input_->Ref();
    ADP_LOG(INFO) << "NpuMapDatasetOp::Dataset";
  }

  ~Dataset() override {
    ADP_LOG(INFO) << "~Dataset enter.";
    input_->Unref();
    ADP_LOG(INFO) << "~Dataset finish.";
  }

  bool IsStaticShape() const {
    // DT_STRING is dyn len type;
    if (std::any_of(output_types_.cbegin(), output_types_.cend(),
        [](DataType type) { return type == DT_STRING; })) {
      return true;
    }
    return (!DatasetFunction::HaveUnknowShape(output_shapes_));
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
#if defined(TF_VERSION_TF2)
    string prefix_para = name_utils::IteratorPrefix(kDatasetType, prefix);
#else
    string prefix_para = strings::StrCat(prefix, "::", kDatasetType);
#endif
    if (IsStaticShape()) {
      if (output_device_ == kCpu) {
        ADP_LOG(INFO) << "NpuMapDatasetOp::MakeIteratorInternal, IteratorStaticCpu";
        return absl::make_unique<IteratorStaticCpu>(IteratorStaticCpu::Params {
            this, prefix_para});
      } else {
        ADP_LOG(INFO) << "NpuMapDatasetOp::MakeIteratorInternal, IteratorStaticNpu";
        return absl::make_unique<IteratorStaticNpu>(IteratorStaticNpu::Params {
            this, prefix_para});
      }
    } else {
      if (output_device_ == kCpu) {
        ADP_LOG(INFO) << "NpuMapDatasetOp::MakeIteratorInternal, IteratorDynCpu";
        return absl::make_unique<IteratorDynCpu>(IteratorDynCpu::Params {
            this, prefix_para});
      } else {
        ADP_LOG(INFO) << "NpuMapDatasetOp::MakeIteratorInternal, IteratorDynNpu";
        return absl::make_unique<IteratorDynNpu>(IteratorDynNpu::Params {
            this, prefix_para});
      }
    }
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
#if defined(TF_VERSION_TF2)
    return name_utils::DatasetDebugString(kDatasetType);
#else
    return "NpuMapDatasetOp::DataSet";
#endif
  }

  int64 Cardinality() const override {
    if (!preserve_cardinality_) {
      return kUnknownCardinality;
    }
    return input_->Cardinality();
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
    // Input: input_dataset
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    // Input: other_arguments
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));

    // Input: num_parallel_calls
    Node* num_parallel_calls_node;
    TF_RETURN_IF_ERROR(
        b->AddScalar(static_cast<int32>(num_parallel_calls_), &num_parallel_calls_node));

    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {std::make_pair(0, input_graph_node),
          std::make_pair(2, num_parallel_calls_node)},  // Single tensor inputs.
        {std::make_pair(1, other_arguments)},      // Tensor list inputs.
        attrs_,  // Attrs
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
        deterministic_(params.dataset->deterministic_),
        preserve_cardinality_(params.dataset->preserve_cardinality_),
        func_(dataset()->init_options_, dataset()->captured_func_->func().name(),
            dataset()->input_->output_dtypes(), dataset()->output_dtypes(),
            dataset()->input_->output_shapes(), dataset()->output_shapes()) {
          ADP_LOG(INFO) << "Dataset::IteratorMeBase";
        }

    virtual ~IteratorMeBase() = default;

    int64_t GetParallelCallsNum() const {
      return (num_parallel_calls_->value <= 0) && (ctx_ != nullptr) ?
          ctx_->runner_threadpool_size() :
          num_parallel_calls_->value;
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(*mu_);

      TF_RETURN_IF_ERROR(DatasetFunction::RegisterNpuCancellation(
          [this]() { CancelThreads(/* wait= */ true); }, &deregister_fn_));

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

      return func_.Initialize(dataset()->sess_options_,
          const_cast<FunctionLibraryDefinition *>(dataset()->captured_func_->lib_def()));
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      int result_id = -1;
      {
        mutex_lock l(*mu_);
        *end_of_sequence = false;
        Status status = EnsureRunnerThreadStarted(ctx);
        // 1. No data, return ok, end_of_sequence is false
        // 2. get first data failed, return status
        if (!status.ok()) {
          *end_of_sequence = end_of_input_;
          return status;
        }
        while (!cancelled_ && ShouldWait(result_id)) {
          if ((end_of_input_) && (recved_num_ == 0)) {
            *end_of_sequence = true;
            return Status::OK();
          }
          ++waiting_;
          RecordStop(ctx);
          ADP_LOG(INFO) << "Wait data ...";
          cond_var_->wait(l);
          RecordStart(ctx);
          --waiting_;
          }
          if (cancelled_) {
            return errors::Cancelled("Iterator was cancelled");
          }
      }

      output_results_[result_id]->result_id= result_id;
      Status status = ProcessOutputResults(output_results_[result_id], *out_tensors);
      ADP_LOG(INFO) << "GetNext return result_id = " << result_id
          << ", status = " << status.ToString()
          << ", end_of_sequence = " << *end_of_sequence;
      for (int i = 0; i < out_tensors->size(); i++) {
        ADP_LOG(INFO) << "Tensor-" << i << ", shape: " << (*out_tensors)[i].shape().DebugString();
      }
      return status;
    }

  protected:
    class OutputResultBase {
    public:
      explicit OutputResultBase()
          : status(Status::OK()) {};
      virtual ~OutputResultBase() {
        ADP_LOG(INFO) << "~OutputResultBase enter.";
        if (output != nullptr) {
          rtError_t rt = rtFree(output);
          if (rt != RT_ERROR_NONE) {
            ADP_LOG(ERROR) << "RT FREE failed, rtError_t = " << rt;
          } else {
            output = nullptr;
          }
        }

        if (output_cpu != nullptr) {
          delete[] output_cpu;
          output_cpu = nullptr;
        }
        ADP_LOG(INFO) << "~OutputResultBase finish.";
      }

      void InitOutputs(uint8_t *start_addr, std::vector<int64> &tensor_align_size, std::vector<uint8_t*> &outputs) {
        int dim_num = tensor_align_size.size();
        int64_t offset = 0;
        uint8_t *align_addr = reinterpret_cast<uint8_t *>(NpuAllocator::AlignSize(static_cast<int64_t>(reinterpret_cast<intptr_t>(start_addr))));
        for (int i = 0; i < dim_num; i++) {
          outputs.push_back(align_addr + offset);
          offset += tensor_align_size[i];
        }
      }

      mutex mu;
      Status status;
      int64 result_id;

      uint8_t *output = nullptr;
      std::vector<uint8_t*> outputs;

      uint8_t *output_cpu = nullptr;
      std::vector<uint8_t*> outputs_cpu;
    }; // class OutputResultBase

    virtual bool HasRunRes(int thread_id) { return true; }
    virtual int MapFunc(const std::shared_ptr<IteratorContext>& ctx, int thread_id,
        DatasetFunction::Instance instance, int result_id, std::vector<Tensor> &input) = 0;
    virtual int WaitRunRes(const std::shared_ptr<IteratorContext>& ctx, int thread_id) { return 0; };
    virtual Status InitOutputResults() = 0;
    virtual uint8_t* GetStartAddr(OutputResultBase *output_result) {};
    virtual NpuAllocator* CreateAllocator(OutputResultBase *output_result, int step, std::function<void(void *)> del) {};

    void Finalize() {
      ADP_LOG(INFO) << "~Finalize enter.";
      CancelThreads(true);
      if (deregister_fn_) deregister_fn_();

      delete[] output_results_;
      output_results_ = nullptr;
      ADP_LOG(INFO) << "~Finalize finish.";
    }

    void InitOutputResultsQueue() {
      for (int id = 0; id < max_output_results_; id++) {
        results_empty_que_.emplace_back(id);
      }
    }

    virtual Status InitOutputResultNpuMem(OutputResultBase *output_result) {
      rtError_t rt = rtMalloc(reinterpret_cast<void **>(&output_result->output), output_mem_size_, RT_MEMORY_HBM);
      DATASET_REQUIRES(rt == RT_ERROR_NONE, errors::InvalidArgument(
          "Alloc rtMalloc mem failed: ", output_mem_size_, "rtError_t: ", rt));

      output_result->InitOutputs(output_result->output, func_tensor_align_size_, output_result->outputs);
      return Status::OK();
    }

    void InitOutputResultCpuMem(OutputResultBase *output_result) {
      output_result->output_cpu = new (std::nothrow)uint8_t[output_mem_size_];
      if (output_result->output_cpu != nullptr) {
        output_result->InitOutputs(output_result->output_cpu, func_tensor_align_size_, output_result->outputs_cpu);
      }
    }

    Status ProcessOutputResults(std::shared_ptr<OutputResultBase> &output_result,
        std::vector<Tensor> &out_tensors) {
      mutex_lock l(output_result->mu);
      recved_num_--;

      if (!output_result->status.ok() && !errors::IsOutOfRange(output_result->status)) {
        return output_result->status;
      }

      return TransOutputResultToTensor(output_result, out_tensors);
    }

    void CancelThreads(bool wait) {
      ADP_LOG(INFO) << "CancelThreads wait="<<wait;
#if defined(TF_VERSION_TF2)
      cancellation_manager_->StartCancel();
#endif
      {
        mutex_lock l(*mu_);
        if (runner_threads_.size() > 0) {
          cancelled_ = true;
          cond_var_->notify_all();
        }
      }
      while (wait && thread_num_ > 0) {
        cond_var_->notify_all();
        usleep(kSleepUs);
      }
      ADP_LOG(INFO) << "CancelThreads finish";
    }

    void CallCompleted() {
      mutex_lock l(*mu_);
      cond_var_->notify_all();
    }

    bool ShouldWait(int &result_id) {
      if (cancelled_) {
        return false;
      }
      if (results_ready_que_.empty()) {
        ADP_LOG(INFO) << "ShouldWait results_ready_que_ is empty.";
        return true;
      }

      int temp_read_idx = results_ready_que_.begin()->first;
      result_id = results_ready_que_.begin()->second;
      if (!deterministic_) {
        read_idx_ = temp_read_idx;
      } else {
        if (read_idx_ != temp_read_idx) {
          return true;
        }
        read_idx_++;
      }
      results_ready_que_.erase(results_ready_que_.begin());
      ADP_LOG(INFO) << "ShouldWait end. find one avaliable result. deterministic_=" << deterministic_
            << " , read_idx_=" << (read_idx_-1) << " , result_id=" << result_id;
      return false;
    }

    bool GetNextResultIndex(int64_t &result_id) {
      if (results_empty_que_.empty()) {
        return false;
      }

      // results_empty_que_ stores all the available result_id
      result_id = results_empty_que_.front();
      results_empty_que_.pop_front();

      return true;
    }

    void RunnerThread(const std::shared_ptr<IteratorContext>& ctx, int64_t thread_id) {
      DatasetFunction::Instance instance;
      Status status = func_.Instantialte(instance);
      if (!status.ok()) {
        ADP_LOG(ERROR) << "DatasetFunction Instantialte failed, status = " << status.ToString()
          << " , thread_id=" << thread_id;
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
        ADP_LOG(INFO) << "Thread exit: thread_id = " << thread_id << "thread_num = " << this->thread_num_;
      });

      rtSetDevice(device_id_);

      int run_res = 1;
      while (!cancelled_) {
        // Implementation class to implement
        // if no run res, need to wait run res
        // if end of input, need to wait run end
        while (!HasRunRes(thread_id) || (end_of_input_ && run_res)) {
          run_res = WaitRunRes(ctx, thread_id);
        }

        int64_t result_id;
        bool map_func = false;
        std::vector<Tensor> in_tensors;
        {
          mutex_lock l(*mu_);
          // if tf restore the data status, this will continue;
          if (!end_of_input_ && GetNextResultIndex(result_id)) {
            Status status = input_impl_->GetNext(ctx.get(), &in_tensors, &end_of_input_);
            ADP_LOG(INFO) << "input_impl_->GetNext return failed or end, status = " << status.ToString()
                << ", thread_id = " << thread_id << ", result_id = " << result_id
                << ", end_of_input_="<<end_of_input_;
            if (status.ok() && !end_of_input_) {
              map_func = true;
              recved_num_++;
            } else {
              if (!status.ok()) {
                ADP_LOG(INFO) << "input_impl_->GetNext return failed, status = " << status.ToString()
                  << " thread_id = " << thread_id << " result_id = " << result_id;
                results_empty_que_.emplace_back(result_id);
              } else {
                results_empty_que_.emplace_back(result_id);
              }
              cond_var_->notify_all();
            }
          } else {
            if (run_res > 0) {
              // to wait run complete
            } else {
              ADP_LOG(INFO) << "RunnerThread: thread_id=" << thread_id <<", end_of_input_="<<end_of_input_;
              RecordStop(ctx.get());
              cond_var_->wait(l);
              RecordStart(ctx.get());
              continue;
            }
          }
        }
        if (map_func) {
          ADP_LOG(INFO) << "MapFunc enter: thread_id = " << thread_id << ", result_id = " << result_id;
          run_res = MapFunc(ctx, thread_id, instance, result_id, in_tensors);
        } else {
          run_res = WaitRunRes(ctx, thread_id);
        }
      }
    }

    Status EnsureRunnerThreadStarted(IteratorContext* ctx) {
      if (runner_threads_.empty()) {
        rtSetDevice(device_id_);
        Status status = InitOutputResults();
        if (!status.ok()) {
          return status;
        }

        ctx_ = std::make_shared<IteratorContext>(*ctx);
        // if this GetParallelCallsNum() can be stores in a variable ?
        for (int64_t i = 0; i < GetParallelCallsNum(); i++) {
          runner_threads_.emplace_back(ctx->StartThread(
              kNpuMapDataset + std::to_string(i),
              std::bind(&IteratorMeBase::RunnerThread, this, ctx_, i)));
        }
      }
      return Status::OK();
    }

    virtual Status TransOutputResultToTensor(std::shared_ptr<OutputResultBase> &output_result, std::vector<Tensor> &out_tensors) {
      out_tensors.clear();
      std::shared_ptr<uint8_t> npu_addr(GetStartAddr(output_result.get()), [this, output_result](uint8_t *) {
        this->results_empty_que_.emplace_back(output_result->result_id);
        this->cond_var_->notify_all();
        ADP_LOG(INFO) << "Output result reset  output_result->status=" << output_result->status
            << " , and then update results_empty_que_ with result_id=" << output_result->result_id;
      });
      DATASET_REQUIRES((npu_addr != nullptr),
          errors::InvalidArgument("Alloc mem failed: ", output_mem_size_));

      int tensor_num = func_tensor_size_.size();
      for (int i = 0; i < tensor_num; i++) {
        out_tensors.push_back(BuildTensorByMem(i, npu_addr, output_result.get()));
      }

      return Status::OK();
    }

    Tensor BuildTensorByMem(int tensor_id, std::shared_ptr<uint8_t> &npu_addr, OutputResultBase *output_result) {
      NpuAllocator *npu_allocator = CreateAllocator(output_result, tensor_id, [npu_addr](void *addr) mutable { npu_addr.reset(); });
      TensorShape shape = {};
      std::for_each(func_output_shape_[tensor_id].cbegin(), func_output_shape_[tensor_id].cend(),
          [&shape](int64_t i) { shape.AddDim(static_cast<int64>(i)); });
      return Tensor(npu_allocator, dataset()->output_types_[tensor_id], shape);
    }

    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncKnownRatioNode(
          std::move(args), /* ratio= */ 1,
          { model::MakeParameter(kParallelism, num_parallel_calls_, /* min= */ 1,
                                 /* max= */ ctx->runner_threadpool_size()) });
    }
#if defined(TF_VERSION_TF2)
    TraceMeMetadata GetTraceMeMetadata() const override {
      int64 parallelism = -1;
      int64 max_output_results = -1;
      // NOTE: We only set the parallelism value if the lock can be acquired
      // right away to avoid introducing tracing overhead.
      if (mu_->try_lock()) {
        parallelism = GetParallelCallsNum();
        max_output_results = max_output_results_;
        mu_->unlock();
      }
      auto result = dataset()->traceme_metadata_;
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
      return strings::StrCat(prefix(), "#parallelism=", num_parallel_calls_->value, "#");
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
    std::vector<std::unique_ptr<Thread>> runner_threads_;
    const std::shared_ptr<condition_variable> cond_var_;
    std::unique_ptr<IteratorBase> input_impl_;
    bool cancelled_ = false;
    int64 waiting_ = 0;
#ifdef TF_VERSION_TF2
    std::unique_ptr<CancellationManager> cancellation_manager_;
#endif
    std::function<void()> deregister_fn_;

    const std::shared_ptr<model::SharedState> num_parallel_calls_;
    int64 max_output_results_ = 0;
    bool end_of_input_ = false;
    std::vector<std::vector<int64_t>> func_output_shape_;
    std::shared_ptr<OutputResultBase> *output_results_ = nullptr;
    std::map<int, int> results_ready_que_;
    std::deque<int> results_empty_que_;
    int64 recved_num_ = 0;
    int64 write_idx_ = 0;
    int64 read_idx_ = 0;

    DatasetFunction func_;
    const bool deterministic_;
    const bool preserve_cardinality_;
    std::vector<int64> func_tensor_size_;
    std::vector<int64> func_tensor_align_size_;
    int64 output_mem_size_ = 0;
    int64 max_output_mem_size_ = 0;
    int64_t thread_num_ = 0;
    uint32_t device_id_ = 0;
  }; // class IteratorMeBase

  class IteratorStatic : public IteratorMeBase {
  public:
    explicit IteratorStatic(const Params& params)
        : IteratorMeBase(params) {
      max_output_results_ = GetParallelCallsNum()<<1;
    }

    virtual ~IteratorStatic() {
      ADP_LOG(INFO) << "~IteratorStatic.";
    }

  protected:
    class OutputStaticResult : public OutputResultBase {
    public:
      explicit OutputStaticResult() {};
      virtual ~OutputStaticResult() {
        ADP_LOG(INFO) << "~OutputStaticResult.";
      }
    };

    static constexpr int kMaxTask = 2;
    Status InitOutputResults() override {
      int dim_num = dataset()->output_types_.size();
      output_mem_size_ = 1;
      for (int i = 0; i < dim_num; i++) {
        std::vector<int64_t> output_shape = DatasetFunction::GetTfShapeDims(dataset()->output_shapes_[i]);
        output_shape.erase(output_shape.begin());
        func_output_shape_.push_back(output_shape);
        int64_t tensor_size = DatasetFunction::GetShapeDims(output_shape) * DataTypeSize(dataset()->output_types_[i]);
        func_tensor_size_.push_back(tensor_size);
        tensor_size = NpuAllocator::AlignSize(tensor_size);
        func_tensor_align_size_.push_back(tensor_size);
        output_mem_size_ += tensor_size;
      }
      output_mem_size_ += NpuAllocator::GetAlignment();

      stream_pool_.reset(new (std::nothrow)StreamPool(GetParallelCallsNum(), kMaxTask));
      DATASET_REQUIRES(stream_pool_ != nullptr, errors::Unavailable("create stream pool failed."));
      return InitOutputResultsMem();
    }

    bool HasRunRes(int thread_id) override {
      return stream_pool_->GetIdleEventCount(thread_id) > 0;
    }

    int MapFunc(const std::shared_ptr<IteratorContext>& ctx, int thread_id, DatasetFunction::Instance instance,
        int result_id, std::vector<Tensor> &input)  override {
      std::shared_ptr<std::vector<ge::Tensor>> out_tensors = std::make_shared<std::vector<ge::Tensor>>();
      auto done = [this, out_tensors, result_id = result_id](Status status) mutable {
        {
          mutex_lock l(*this->mu_);
          if (status.ok()) {
            this->results_ready_que_.insert(std::pair<int, int>(this->write_idx_, result_id));
            ADP_LOG(INFO) << "Call GE run function success. update results_ready_que_ enqueue with write_idx_="
                << this->write_idx_ << " result_id = " << result_id;
            this->write_idx_++;
          } else {
            this->results_empty_que_.emplace_back(result_id);
            ADP_LOG(INFO) << "Call GE run function failed. update results_empty_que_ enqueue with result_id = "
                << result_id;
          }
        }
        this->CallCompleted();
        out_tensors.reset();
      };

      if (out_tensors == nullptr) {
        done(errors::Unavailable("create out_tensors failed."));
        return stream_pool_->GetWaitingEventCount(thread_id);
      }

      out_tensors->resize(func_output_shape_.size());
      InitOutTensorsMem(*out_tensors, *static_cast<OutputStaticResult*>(output_results_[result_id].get()));
      ADP_LOG(INFO) << "InitOutTensorsMem finish";

      std::shared_ptr<Stream> stream = stream_pool_->GetStream(thread_id);
      if (stream == nullptr) {
        done(errors::Unavailable("Get Stream failed"));
      }

      Status status = func_.RunWithStreamAsyn(instance, stream->GetStream(), input, *out_tensors);
      if (status.ok()) {
        status = stream_pool_->RecordEvent(stream, done);
      }
      if (!status.ok()) {
        ADP_LOG(ERROR) << "RecordEvent Failed, status = " << status.ToString();
        done(status);
      }
      return stream_pool_->GetWaitingEventCount(thread_id);
    }

    int WaitRunRes(const std::shared_ptr<IteratorContext>& ctx, int thread_id) override {
      ADP_LOG(INFO) << "stream_pool_->WaitOneEvent(thread_id) enter. thread_id = " << thread_id;
      stream_pool_->WaitOneEvent(thread_id);
      ADP_LOG(INFO) << "stream_pool_->WaitOneEvent(thread_id) out. thread_id = " << thread_id;
      return stream_pool_->GetWaitingEventCount(thread_id);
    }

    ge::Format GetFormat() {
      return ge::Format::FORMAT_ND;
    }

    std::unique_ptr<StreamPool> stream_pool_ = nullptr;

  private:
    void InitOutTensorsMem(std::vector<ge::Tensor> &tensors, OutputStaticResult &output_result) {
      int tensor_num = tensors.size();
      const std::vector<ge::DataType> &ge_output_type = func_.GetGeDataTypes();
      for (int i = 0; i < tensor_num; i++) {
        ge::TensorDesc tensor_desc = tensors[i].GetTensorDesc();
        tensor_desc.Update(ge::Shape(func_output_shape_[i]), GetFormat(), ge_output_type[i]);
        tensors[i].SetTensorDesc(tensor_desc);
        tensors[i].SetData(output_result.outputs[i] + func_tensor_size_[i], func_tensor_size_[i], [](uint8_t *p) {
          ADP_LOG(INFO) << "DeInitOutTensorsMem, p = " << static_cast<const void*>(p);
          return;
        });
      }
    }

    Status InitOutputResultsMem() {
      output_results_ = new std::shared_ptr<OutputResultBase>[max_output_results_]();
      DATASET_REQUIRES(output_results_ != nullptr,
          errors::InvalidArgument("Make output results faild."));
      for (int i = 0; i < max_output_results_; i++) {
        output_results_[i].reset(new (std::nothrow)OutputStaticResult());
        DATASET_REQUIRES(output_results_[i] != nullptr,
            errors::InvalidArgument("Make output result faild: i = ", i));
        Status status = InitOutputResultNpuMem(output_results_[i].get());
        DATASET_REQUIRES(status.ok(), status);
      }

      InitOutputResultsQueue();
      return Status::OK();
    }
  }; // class IteratorStatic

  class IteratorStaticNpu : public IteratorStatic {
  public:
    explicit IteratorStaticNpu(const Params& params)
        : IteratorStatic(params) {
    }

    ~IteratorStaticNpu() override {
      ADP_LOG(INFO) << "~IteratorStaticNpu enter.";
      Finalize();
      ADP_LOG(INFO) << "~IteratorStaticNpu finish.";
    }

  protected:
    uint8_t* GetStartAddr(OutputResultBase *output_result) override {
      return output_result->output;
    }

    NpuAllocator* CreateAllocator(OutputResultBase *output_result, int step,
      std::function<void(void *)> del) override {
        return NpuAllocator::CreateNpuAllocator(output_result->outputs[step], del);
    }
  }; // class IteratorStaticNpu

  class IteratorStaticCpu : public IteratorStatic {
  public:
    explicit IteratorStaticCpu(const Params& params)
        : IteratorStatic(params) {
      ADP_LOG(INFO)<<"Construct IteratorStaticCpu";
    }

    ~IteratorStaticCpu() override {
      ADP_LOG(INFO) << "~IteratorStaticCpu enter.";
      Finalize();
      ADP_LOG(INFO) << "~IteratorStaticCpu finish.";
    }

  protected:
    uint8_t* GetStartAddr(OutputResultBase *output_result) override {
      if (output_result->output_cpu == nullptr) {
        InitOutputResultCpuMem(output_result);
      }

      if (output_result->output_cpu != nullptr) {
        rtError_t rt = rtMemcpy(output_result->output_cpu, output_mem_size_,
            output_result->output, output_mem_size_, RT_MEMCPY_DEVICE_TO_HOST);
        if (rt != RT_ERROR_NONE) {
          return nullptr;
        }
      }

      return output_result->output_cpu;
    }

    NpuAllocator* CreateAllocator(OutputResultBase *output_result, int step,
      std::function<void(void *)> del) override {
        return  NpuAllocator::CreateCpuAllocator(output_result->outputs[step], del);
    }
  }; // class IteratorStaticCpu

  class IteratorDyn : public IteratorMeBase {
  public:
    explicit IteratorDyn(const Params& params)
        : IteratorMeBase(params) {
      max_output_results_ = GetParallelCallsNum()<<1;
    };

    virtual ~IteratorDyn() {
      ADP_LOG(INFO) << "~IteratorDyn.";
    }

  protected:
    class OutputDynResult : public OutputResultBase {
    public:
      explicit OutputDynResult() {};
      void InitTensors(int tensor_num) {
        output_tensor.resize(tensor_num);
      }
      std::vector<ge::Tensor> output_tensor;
    };

    void InitTensorSize(OutputDynResult *output_result) {
      std::vector<ge::Tensor> &tensors = output_result->output_tensor;

      // reset infos for current output
      output_mem_size_ = 0;
      func_tensor_size_.clear();
      func_output_shape_.clear();
      func_tensor_align_size_.clear();
      output_result->outputs.clear();

      // value i for debug
      int i = 0;
      for (auto it : tensors) {
        ge::Shape shape = it.GetTensorDesc().GetShape();
        int64_t shape_size = shape.GetShapeSize();
        int64_t tensor_size = ((shape_size == 0) ? 1 : shape_size) * DataTypeSize(dataset()->output_types_[i]);
        ADP_LOG(INFO) << "MemCkeck OutputMem, tensor_size[" << i << "] : " << tensor_size << ", datatype: " << dataset()->output_types_[i];
        func_tensor_size_.push_back(tensor_size);
        tensor_size = NpuAllocator::AlignSize(tensor_size);
        func_tensor_align_size_.push_back(tensor_size);
        output_mem_size_ += tensor_size;
        func_output_shape_.emplace_back(std::move(shape.GetDims()));
        i++;
      }
      output_mem_size_ += NpuAllocator::GetAlignment();
    }

    int MapFunc(const std::shared_ptr<IteratorContext>& ctx, int thread_id,
        DatasetFunction::Instance instance, int result_id, std::vector<Tensor> &input) override {
      OutputDynResult *output_dyn_result = static_cast<OutputDynResult*>(output_results_[result_id].get());
      std::vector<ge::Tensor> output;
      Status status = func_.Run(instance, input, output);
      {
        mutex_lock l(*mu_);
        if (status.ok()) {
          ADP_LOG(INFO) << "MapFunc, input: " << *(uint64_t *)DMAHelper::base(&input[0])
              << ", output: " << *((uint64_t *)output[0].GetData());
          results_ready_que_.insert(std::pair<int, int>(write_idx_, result_id));
          output_dyn_result->output_tensor = std::move(output);
          ADP_LOG(INFO) << "Call GE run function success. update results_ready_que_ enqueue with write_idx_="
              << write_idx_ << " result_id = " << result_id;
          write_idx_++;
        } else {
          results_empty_que_.emplace_back(result_id);
          ADP_LOG(INFO) << "Call GE run function failed. update results_empty_que_ enqueue with result_id = "
              << result_id;
        }
      }
      CallCompleted();
      return 0;
    }

    Status InitOutputResults() override {
      output_results_ = new (std::nothrow)std::shared_ptr<OutputResultBase>[max_output_results_]();
      DATASET_REQUIRES(output_results_ != nullptr,
          errors::InvalidArgument("Make output results faild."));
      for (int i = 0; i < max_output_results_; i++) {
        output_results_[i].reset(new (std::nothrow)OutputDynResult());
        DATASET_REQUIRES(output_results_[i] != nullptr,
            errors::InvalidArgument("Make output result faild: i = ", i));
        OutputDynResult *dyn_result = static_cast<OutputDynResult*>(output_results_[i].get());
        dyn_result->InitTensors(dataset()->output_types_.size());
      }
      InitOutputResultsQueue();
      return Status::OK();
    }

  private:
    ge::Format GetFormat() {
      return ge::Format::FORMAT_ND;
    }
  }; // class IteratorDyn

  class IteratorDynNpu : public IteratorDyn {
  public:
    explicit IteratorDynNpu(const Params& params)
        : IteratorDyn(params) {
    };

    ~IteratorDynNpu() override {
      ADP_LOG(INFO) << "~IteratorDynNpu enter.";
      Finalize();
      ADP_LOG(INFO) << "~IteratorDynNpu finish.";
    }

  protected:
    uint8_t* GetStartAddr(OutputResultBase *output_result) {
      InitTensorSize(static_cast<OutputDynResult*>(output_result));

      // reuse the device memory if needed
      if (output_mem_size_ < max_output_mem_size_) {
        output_result->InitOutputs(output_result->output, func_tensor_align_size_, output_result->outputs);
      } else {
        InitOutputResultNpuMem(output_result);
        max_output_mem_size_ = output_mem_size_;
      }

      if (output_result->output != nullptr) {
        ADP_LOG(INFO) << "GetStartAddr return npu_addr = " << (void*)output_result->output;
        return MemCpyData(*static_cast<OutputDynResult*>(output_result));
      }

      return nullptr;
    }

    NpuAllocator* CreateAllocator(OutputResultBase *output_result, int step,
      std::function<void(void *)> del) override {
        return  NpuAllocator::CreateNpuAllocator(output_result->outputs[step], del);
    }

  private:
    uint8_t *MemCpyData(OutputDynResult &output_result) {
      int tensor_num = output_result.outputs.size();
      for (int i = 0; i < tensor_num; i++) {
        uint8_t *npu_addr = output_result.outputs[i];
        int64_t tensor_size = func_tensor_size_[i];
        rtError_t rt = rtMemcpy(npu_addr, tensor_size, output_result.output_tensor[i].GetData(),
            output_result.output_tensor[i].GetSize(), RT_MEMCPY_HOST_TO_DEVICE);
        if (rt != RT_ERROR_NONE) {
          ADP_LOG(ERROR) << "Mem copy from host to device failed: hostsize: " << tensor_size <<
              "device size: " << output_result.output_tensor[i].GetSize() << "rt: " << rt;
          return nullptr;
        }
        npu_addr += tensor_size;
      }
      return output_result.output;
    }
  }; // class IteratorDynNpu

  class IteratorDynCpu : public IteratorDyn {
  public:
    explicit IteratorDynCpu(const Params& params)
        : IteratorDyn(params) {
    };

    ~IteratorDynCpu() override {
      ADP_LOG(INFO) << "~IteratorDynCpu enter.";
      Finalize();
      ADP_LOG(INFO) << "~IteratorDynCpu finish.";
    }

  protected:
    Status TransOutputResultToTensor(std::shared_ptr<OutputResultBase> &output_result,
     std::vector<Tensor> &out_tensors) override {
      InitTensorSize(static_cast<OutputDynResult*>(output_result.get()));

      out_tensors.clear();
      int tensor_num = func_tensor_size_.size();
      for (int i = 0; i < tensor_num; i++) {
        Tensor tensor;
        Status status = BuildTensor(i, tensor, output_result.get());
        if (!status.ok()) {
          return status;
        }
        out_tensors.push_back(std::move(tensor));
      }

      results_empty_que_.emplace_back(output_result->result_id);
      cond_var_->notify_all();
      ADP_LOG(INFO) << "TransOutputResultToTensor out. update results_empty_que_ with result_id="
          << output_result->result_id;
      return Status::OK();
    }

    Status BuildTensor(int tensor_id, Tensor &tensor, OutputResultBase *output_result) {
      OutputDynResult *result_tensor = static_cast<OutputDynResult*>(output_result);
      TensorShape shape = {};
      std::for_each(func_output_shape_[tensor_id].cbegin(), func_output_shape_[tensor_id].cend(),
          [&shape](int64_t i) { shape.AddDim(static_cast<int64>(i)); });

      std::unique_ptr<uint8_t[], std::function<void(uint8_t *)>> addr
        = result_tensor->output_tensor[tensor_id].ResetData();
      std::function<void(uint8_t *)> pdel = addr.get_deleter();
      NpuAllocator *npu_allocator = NpuAllocator::CreateCpuAllocator(std::move(addr.release()),
          [pdel](void *p) mutable {
            if (pdel != nullptr) {
              pdel(reinterpret_cast<uint8_t *>(p));
            }else {
              delete p;
            }});

      tensor = Tensor(npu_allocator, dataset()->output_types_[tensor_id], shape);
      return Status::OK();
    }
  }; // class IteratorDynCpu

  const bool deterministic_;
  // This is used for random access provided by Get().
  mutable std::unique_ptr<InstantiatedCapturedFunction>
      instantiated_captured_func_;

  const DatasetBase* const input_;
  const int64 num_parallel_calls_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const bool preserve_cardinality_;
  const std::string output_device_;
#if defined(TF_VERSION_TF2)
  const TraceMeMetadata traceme_metadata_;
#endif

  const std::map<std::string, std::string> sess_options_;
  std::map<std::string, std::string> init_options_;
  std::vector<std::pair<StringPiece, AttrValue>>& attrs_;
}; // class Dataet

Status NpuMapDatasetOp::CheckOutputType() {
  for (auto type : output_types_) {
    if (!DataTypeCanUseMemcpy(type)) {
      return errors::InvalidArgument("DT_TYPE is not unspported, DT_TYPE = ", DataTypeString(type));
    }
  }
  return Status::OK();
}

NpuMapDatasetOp::NpuMapDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx),
      sess_options_(NpuAttrs::GetSessOptions(ctx)),
      init_options_(NpuAttrs::GetSessOptions(ctx))   {
  ADP_LOG(INFO) << "Construct of NpuMapDatasetOp";
  FunctionMetadata::Params params;
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFunc, params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputDevice, &output_device_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kDeterministic, &deterministic_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kPreserveCardinality, &preserve_cardinality_));
  OP_REQUIRES_OK(ctx, CheckOutputType());
  for (auto attr : ctx->def().attr()) {
    attrs_.emplace_back(attr.first, attr.second);
  }
}

void NpuMapDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                  DatasetBase** output) {
  ADP_LOG(INFO) << "NpuMapDatasetOp::MakeDataset";
  int64 num_parallel_calls = 0;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kNumParallelCalls, &num_parallel_calls));
  OP_REQUIRES(
      ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutotune,
      errors::InvalidArgument("  num_parallel_calls must be greater than zero."));

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  if (num_parallel_calls == model::kAutotune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }

  *output =
      new Dataset(ctx, input, num_parallel_calls, output_types_, output_shapes_,
                  output_device_, deterministic_, std::move(captured_func),
                  preserve_cardinality_, sess_options_, init_options_, attrs_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("NpuMapDataset").Device(DEVICE_CPU), NpuMapDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("NpuMapDataset");
}  // namespace
}  // namespace data
}  // namespace tensorflow
