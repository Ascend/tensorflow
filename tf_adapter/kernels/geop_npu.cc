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

#include "tf_adapter/kernels/geop_npu.h"

#include <chrono>
#include <cstdint>
#include <dirent.h>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <map>
#include <memory>
#include <mmpa/mmpa_api.h>
#include <queue>
#include <securec.h>
#include <securectype.h>
#include <thread>
#include <vector>

#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/ge_plugin.h"
#include "tf_adapter/util/infershape_util.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/generate_report.h"
#include "tf_adapter/util/npu_ops_identifier.h"
#include "tf_adapter/util/session_manager.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_api.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "tdt/tdt_host_interface.h"
#include "tdt/tsd_client.h"

using namespace tdt;

namespace tensorflow {
Status FunctionalizeControlFlow(Graph *graph, FunctionLibraryDefinition *library);
namespace {
using geDataUniquePtr = std::unique_ptr<uint8_t[], std::function<void(uint8_t *)>>;

class NpuHostFixedAllocator : public tensorflow::Allocator {
 public:
  static tensorflow::Allocator *Create(geDataUniquePtr ptr) {
    return new (std::nothrow) NpuHostFixedAllocator(std::move(ptr));
  }
 private:
  explicit NpuHostFixedAllocator(geDataUniquePtr ptr) : ptr_(std::move(ptr)) {
    ADP_LOG(INFO) << "[GEOP] Zero copied ge tensor " << reinterpret_cast<uintptr_t>(ptr_.get());
  }
  ~NpuHostFixedAllocator() override {
    ADP_LOG(INFO) << "[GEOP] Release zero copied ge tensor " << reinterpret_cast<uintptr_t>(ptr_.get());
  }
  std::string Name() override { return "NpuHostFixedAllocator"; }
  void *AllocateRaw(size_t alignment, size_t num_bytes) override { return ptr_.get(); }
  void DeallocateRaw(void *ptr) override { delete this; }
  geDataUniquePtr ptr_;
};

class NpuGetNextOutputInfo {
public:
  NpuGetNextOutputInfo(ge::Placement placement, std::vector<int64_t> dims,
     size_t output_size, geDataUniquePtr data)
    : placement_(placement), dims_(dims), output_size_(output_size), data_(std::move(data)) {}
  ~NpuGetNextOutputInfo() { ADP_LOG(INFO) << "[GEOP] Release NpuGetNextOutputInfo."; }
  ge::Placement placement_;
  std::vector<int64_t> dims_;
  size_t output_size_;
  geDataUniquePtr data_;
};

class NpuHostGetNextAllocator : public tensorflow::Allocator {
 public:
  static tensorflow::Allocator *Create(std::unique_ptr<NpuGetNextOutputInfo> output) {
    return new (std::nothrow) NpuHostGetNextAllocator(std::move(output));
  }
 private:
  explicit NpuHostGetNextAllocator(std::unique_ptr<NpuGetNextOutputInfo> output) : output_(std::move(output)) {
    ADP_LOG(INFO) << "[GEOP] getnext data addr:" << reinterpret_cast<uintptr_t>(output_->data_.get());
  }
  ~NpuHostGetNextAllocator() override {
    ADP_LOG(INFO) << "[GEOP] Release getnext data addr:" << reinterpret_cast<uintptr_t>(output_->data_.get());
  }
  std::string Name() override { return "NpuHostGetNextAllocator"; }
  void *AllocateRaw(size_t alignment, size_t num_bytes) override { return output_.get(); }
  void DeallocateRaw(void *ptr) override { delete this; }
  std::unique_ptr<NpuGetNextOutputInfo> output_;
};

inline string ToString(ge::Status status) { return ::ge::StatusFactory::Instance()->GetErrDesc(status); }
Status BuildOutputTensorInfo(OpKernelContext *ctx, std::vector<ge::Tensor> &outputs) {
  // ctx is not nullptr
  int num_outputs = ctx->num_outputs();
  ADP_LOG(INFO) << "BuildOutputTensorInfo, num_outputs:" << num_outputs;
  if (num_outputs != static_cast<int>(outputs.size())) {
    ADP_LOG(ERROR) << "[GEOP] Outputs num mismatched, need:" << num_outputs << ", while GE return:" << outputs.size();
    LOG(ERROR) << "[GEOP] Outputs num mismatched, need:" << num_outputs << ", while GE return:" << outputs.size();
    return errors::InvalidArgument("Outputs num mismatched, need:", num_outputs, ", while GE return:", outputs.size());
  }

  // populate outputs
  for (int i = 0; i < num_outputs; ++i) {
    ge::Tensor &output = outputs[i];
    std::vector<int64_t> ge_output_dims = output.GetTensorDesc().GetShape().GetDims();
    ge::Placement data_placement = output.GetTensorDesc().GetPlacement();
    std::vector<int64> dims;
    for (int64_t dim : ge_output_dims) {
      dims.push_back(dim);
    }
    TensorShape out_shape(dims);
    const DataType out_type = ctx->op_kernel().output_type(i);
    size_t output_size = output.GetSize();
    geDataUniquePtr data_ptr = std::move(output.ResetData());
    ADP_LOG(INFO) << "[GEOP] Get ge output: " << i << " tensor shape is: " << out_shape.DebugString()
                  << ", data placement is: " << data_placement << ", output_size is: "
                  << output_size << ", data addr is: " << reinterpret_cast<uintptr_t>(data_ptr.get());

    if (data_placement != ge::kPlacementDevice) {
      const static int64_t kTensorAlignBytes = 64;
      if (reinterpret_cast<uintptr_t>(data_ptr.get()) % kTensorAlignBytes == 0) {
        ADP_LOG(INFO) << "[GEOP] Zero copy ge tensor " << reinterpret_cast<uintptr_t>(data_ptr.get())
                      << " as aligned with " << kTensorAlignBytes << " bytes";
        Allocator *allocator = NpuHostFixedAllocator::Create(std::move(data_ptr));
        Tensor cpu_tensor(allocator, out_type, out_shape);
        if (output_size != cpu_tensor.TotalBytes()) {
          LOG(ERROR) << "[GEOP] Graph engine process graph success but output " << i << " total bytes "
                     << output_size << " mismatched with expected " << cpu_tensor.TotalBytes();
          return errors::Internal("Graph engine process graph success but output length mismatched with expected.");
        }
        ctx->set_output(i, cpu_tensor);
      } else {
        ADP_LOG(ERROR) << "[GEOP] Skip zero copy as ge tensor, " << reinterpret_cast<uintptr_t>(data_ptr.get())
                       << " not aligned with " << kTensorAlignBytes << " bytes";
        return errors::Internal("[GEOP] Skip zero copy ge tensor, bytes not aligned with expected.");
      }
    } else {
      ADP_LOG(INFO) << "[GEOP] GE output data placement is device, construct output info tensor.";
      auto getnext_output_info = std::unique_ptr<NpuGetNextOutputInfo>(new NpuGetNextOutputInfo(
                                   data_placement, ge_output_dims, output_size, std::move(data_ptr)));
      Allocator *allocator = NpuHostGetNextAllocator::Create(std::move(getnext_output_info));
      Tensor cpu_tensor(allocator, out_type, out_shape);
      ctx->set_output(i, cpu_tensor);
    }
  }
  ADP_LOG(INFO) << "[GEOP] Build output tensor info success.";
  return Status::OK();
}

bool CmpValue(const std::pair<std::vector<string>, uint32_t> &p1, const std::pair<std::vector<string>, uint32_t> &p2) {
  return p1.second < p2.second;
}

bool CmpVecValue(Node *node1, Node *node2) {
  if (node1 == nullptr || node2 == nullptr) {
    ADP_LOG(ERROR) << "node1 or node2 is nullptr.";
    LOG(ERROR) << "node1 or node2 is nullptr.";
    return false;
  }
  return node1->name() < node2->name();
}
}  // namespace

std::string CurrentTimeInStr() {
  std::time_t now = std::time(nullptr);
  std::tm *ptm = std::localtime(&now);
  if (nullptr == ptm) { return ""; }

  const int time_buffer_len = 32;
  char buffer[time_buffer_len] = {0};
  // format: 20171122042550
  std::strftime(buffer, time_buffer_len, "%Y%m%d%H%M%S", ptm);
  return std::string(buffer);
}

static const int64 kMicrosToMillis = 1000;
const int kInvalidGraphId = 0;
const int kMaxCacheNum = 10;
const int kFatalSleepTime = 3000;

GeOp::GeOp(OpKernelConstruction *ctx)
    : AsyncOpKernel(ctx), init_flag_(false), build_flag_(false), add_graph_flag_(false),
      sess_init_flag_(false), compute_graph_empty_(false), data_format_(""), graph_id_(0),
      is_initialized_graph_(false), need_iteration_(false), tf_session_(""), ge_session_(nullptr),
      job_type_(""), is_host_graph_(false), handle_(nullptr), aoe_tuning_(nullptr),
      need_compile_graph_first_(false), aoe_init_(nullptr), aoe_finalize_(nullptr) {
  Initialize(ctx);
}

GeOp::~GeOp() { Finalize(); }

void GeOp::Initialize(OpKernelConstruction *ctx) {
  int64 startTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "[GEOP] Begin GeOp initialize.";
  if (init_flag_) {
    ADP_LOG(WARNING) << "[GEOP] GEOP already Initialize.";
    LOG(WARNING) << "[GEOP] GEOP already Initialize.";
    return;
  }

  CHECK_NOT_NULL(ctx);
  const NameAttrList *func = nullptr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("function", &func));
  function_ = *func;
  string data_format;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
  this->data_format_ = data_format;

  Status s = ctx->GetAttr("_session", &tf_session_);
  if (s.ok()) { ADP_LOG(INFO) << "[GEOP] get session info from attr, tf session: " << tf_session_; }

  ctx->GetAttr("_dynamic_input", &dynamic_input_);
  if (!dynamic_input_.empty() && dynamic_input_ == "1") {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("_dynamic_graph_execute_mode", &dynamic_graph_execute_mode_));
    ctx->GetAttr("_getnext_inputs_shape_range", &getnext_inputs_shape_range_);
    ctx->GetAttr("_data_inputs_shape_range", &data_inputs_shape_range_);
    ctx->GetAttr("_is_dynamic_getnext", &is_dynamic_getnext_);
    ctx->GetAttr("_placeholder_index", &placeholder_index_);
  }
  ctx->GetAttr("_train_graph", &is_train_graph_);
  ADP_LOG(INFO) << "[GEOP] dynamic_input: " << dynamic_input_
                << ", dynamic_graph_execute_mode: " << dynamic_graph_execute_mode_
                << ", getnext_inputs_shape_range: " << getnext_inputs_shape_range_
                << ", data_inputs_shape_range: " << data_inputs_shape_range_
                << ", is_train_graph: " << is_train_graph_
                << ", is_dynamic_getnext: " << is_dynamic_getnext_
                << ", placeholder_index: " << placeholder_index_;

  // global environment Initialize, invoke once for each process
  string sess_config = "";
  OP_REQUIRES_OK(ctx, ctx->GetAttr("_NpuOptimizer", &sess_config));
  std::map<std::string, std::string> pass_options = NpuAttrs::GetPassOptions(ctx);
  iteration_per_loop_ = std::atoi(pass_options["iterations_per_loop"].c_str());
  job_type_ = pass_options["job"];
  if (GePlugin::GetInstance()->IsGlobal()) {
    ADP_LOG(INFO) << "[GEOP] GePlugin global, skip GePlugin init";
    init_options_ = GePlugin::GetInstance()->GetInitOptions();
  } else {
    init_options_ = NpuAttrs::GetInitOptions(ctx);
    GePlugin::GetInstance()->Init(init_options_);
    ADP_LOG(INFO) << "[GEOP] GePlugin init success";
  }
  ADP_LOG(INFO) << "init options:";
  NpuAttrs::LogOptions(init_options_);

  if (!init_options_["ge.jobType"].empty() && !init_options_["ge.tuningPath"].empty()) {
    handle_ = mmDlopen("libaoe_tuning.so", MMPA_RTLD_NOW);
    OP_REQUIRES(ctx, handle_ != nullptr,
      errors::InvalidArgument("libaoe_tuning.so dlopen failed, ", mmDlerror()));
    aoe_tuning_ = (AoeTuningFunc)mmDlsym(handle_, const_cast<char *>("AoeOnlineTuning"));
    aoe_init_ = (AoeInitFunc)mmDlsym(handle_, const_cast<char *>("AoeOnlineInitialize"));
    aoe_finalize_ = (AoeFinalizeFunc)mmDlsym(handle_, const_cast<char *>("AoeOnlineFinalize"));
    OP_REQUIRES(ctx, aoe_tuning_ != nullptr && aoe_init_ != nullptr && aoe_finalize_ != nullptr,
      errors::InvalidArgument("dlsym Aoe API failed, ", mmDlerror()));
  }

  sess_options_ = NpuAttrs::GetSessOptions(ctx);
  ADP_LOG(INFO) << "session options:";
  NpuAttrs::LogOptions(sess_options_);

  init_flag_ = true;
  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(EVENT) << "[GEOP] GeOp Initialize success, cost:"
                 << " [" << ((endTime - startTime) / kMicrosToMillis) << " ms]";
  return;
}

void GeOp::Finalize() {
  {
    ADP_LOG(INFO) << "[GEOP] GeOp start to finalize, tf session: " << tf_session_ << ", graph_id_: " << graph_id_;
    // global environment finalize, invoke once for each process
    {
      mutex_lock lock{mu_};
      uint32_t graph_id = -1;
      if (sess_init_flag_ || !tf_session_.empty()) {
        bool ret = DecrementGraphIdCount(tf_session_, graph_id);
        if (!ret || graph_id < kInvalidGraphId) {
          ADP_LOG(ERROR) << "tf session " << tf_session_ << " sub graph id failed.";
          LOG(ERROR) << "tf session " << tf_session_ << " sub graph id failed.";
          return;
        }
        if (graph_id == kInvalidGraphId) {
          SessionManager::GetInstance().DestroyGeSession(tf_session_);
          ClearGraphIdCount(tf_session_);
        }
      }

      if (!SessionManager::GetInstance().IsGeSessionExist()) {
        if (!GePlugin::GetInstance()->IsGlobal()) {
          if (!init_options_["ge.jobType"].empty() && !init_options_["ge.tuningPath"].empty() &&
              aoe_finalize_ != nullptr) {
            AoeStatus tune_ret = (*aoe_finalize_)();
            if (tune_ret != AOE_SUCCESS) {
              ADP_LOG(ERROR) << "[GEOP] exec aoe finalize func failed.";
              LOG(ERROR) << "[GEOP] exec aoe finalize func failed.";
              return;
            }
          }
          GePlugin::GetInstance()->Finalize();
          ADP_LOG(INFO) << "[GEOP] GePlugin Finalize success";
        } else {
          ADP_LOG(INFO) << "[GEOP] GePlugin global, skip GePlugin Finalize";
        }
        if (!GenerateReport::GetInstance()->SaveUnsupportedInfo().ok()) {
          ADP_LOG(WARNING) << "[GEOP] Save check report failed.";
          LOG(WARNING) << "[GEOP] Save check report failed.";
        }
        if (handle_ != nullptr) {
          (void)mmDlclose(handle_);
        }
      }
    }
  }
  init_flag_ = false;
  ADP_LOG(INFO) << "[GEOP] GeOp Finalize success, tf session: " << tf_session_ << ", graph_id_: " << graph_id_;
  return;
}

int GeOp::InitRebuildFlag(uint32_t cache_graph_id) {
  if (!build_flag_) {
    ADP_LOG(INFO) << "[GEOP] tf session " << tf_session_ << ", graph id: " << cache_graph_id
              << " does not build yet, no need to check rebuild";
    return 0;
  }
  if (ge_session_ == nullptr) {
    ADP_LOG(ERROR) << "[GEOP] GE session is nullptr";
    LOG(ERROR) << "[GEOP] GE session is nullptr";
    return -1;
  }
  if (!ge_session_->IsGraphNeedRebuild(cache_graph_id)) {
    ADP_LOG(INFO) << "[GEOP] tf session " << tf_session_ << ", graph id: " << cache_graph_id << " no need to rebuild";
    return 0;
  }

  ADP_LOG(INFO) << "[GEOP] The graph need rebuild, graph id " << cache_graph_id;

  // The graph need to rebuild, remove it from GE first.
  ADP_LOG(INFO) << "[GEOP] tf session: " << tf_session_ << ", graph id: " << cache_graph_id;
  auto ret = ge_session_->RemoveGraph(cache_graph_id);
  if (ret != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GEOP] Failed to remove graph " << cache_graph_id << " from ge, error code " << ret;
    std::string error_message = ge::GEGetErrorMsg();
    LOG(ERROR) << "[GEOP] Failed to remove graph " << cache_graph_id << " from ge, error code " << ret << std::endl
               << "Error Message is : " << std::endl
               <<error_message;
    return -1;
  }

  build_flag_ = false;
  compute_graph_empty_ = false;
  return 0;
}

bool GeOp::IncrementGraphIdCount(std::string &tf_session, uint32_t &graph_id) {
  if (tf_session_.empty()) {
    ADP_LOG(ERROR) << "[GEOP] Add graph id failed, tf session is empty.";
    LOG(ERROR) << "[GEOP] Add graph id failed, tf session is empty.";
    return false;
  }
  auto it = session_and_graph_id_map_.find(tf_session_);
  if (it != session_and_graph_id_map_.end()) {
    it->second = it->second + kMaxCacheNum;
    graph_id = it->second;
    return true;
  }
  graph_id = 1;
  session_and_graph_id_map_.insert(std::make_pair(tf_session_, graph_id));
  return true;
}

bool GeOp::DecrementGraphIdCount(std::string &tf_session, uint32_t &graph_id) {
  if (tf_session_.empty()) {
    ADP_LOG(ERROR) << "[GEOP] Sub graph id failed, tf session is empty.";
    LOG(ERROR) << "[GEOP] Sub graph id failed, tf session is empty.";
    return false;
  }

  auto it = session_and_graph_id_map_.find(tf_session_);
  if (it != session_and_graph_id_map_.end()) {
    if (it->second == 1) {
      it->second = it->second - 1;
      graph_id = it->second;
      return true;
    }
    it->second = it->second - kMaxCacheNum;
    graph_id = it->second;
    return true;
  }
  ADP_LOG(ERROR) << "[GEOP] Sub graph id failed, can not find tf session " << tf_session;
  LOG(ERROR) << "[GEOP] Sub graph id failed, can not find tf session " << tf_session;
  return false;
}

void GeOp::ClearGraphIdCount(std::string &tf_session) {
  auto it = session_and_graph_id_map_.find(tf_session_);
  if (it != session_and_graph_id_map_.end()) { session_and_graph_id_map_.erase(it); }
}

void GeOp::GetExecGraphId(OpKernelContext *ctx, uint32_t &cache_graph_id,
                          std::vector<std::string> input_shapes) {
  size_t num = cache_graphs_.size();
  if (cache_graphs_.find(input_shapes) != cache_graphs_.end()) {
    for (auto &graph_count : graph_counts_) {
      if (graph_count.first == input_shapes) {
        graph_count.second += 1;
        break;
      }
    }
    cache_graph_id = cache_graphs_[input_shapes];
    build_flag_ = true;
  } else {
    if (num >= kMaxCacheNum) {
      ADP_LOG(INFO) << "[GEOP] the cache vector size is : " << num << " , begin erase the least uesed";
      std::sort(graph_counts_.begin(), graph_counts_.end(), CmpValue);
      uint32_t erased_graph_id = cache_graphs_[graph_counts_[0].first];
      cache_graphs_.erase(graph_counts_[0].first);
      graph_counts_.erase(graph_counts_.begin());
      ge::Status status = ge_session_->RemoveGraph(erased_graph_id);
      if (status != ge::SUCCESS) {
        ADP_LOG(WARNING) << "[GEOP] GE Remove Graph failed, ret : " << ToString(status);
        LOG(WARNING) << "[GEOP] GE Remove Graph failed, ret : " << ToString(status);
      }
      cache_graph_id = erased_graph_id;
    } else {
      cache_graph_id = graph_id_ + num;
    }
    build_flag_ = false;
    compute_graph_empty_ = false;
  }
}

void GeOp::ComputeAsync(OpKernelContext *ctx, DoneCallback done) {
  // ctx is not nullptr
  OP_REQUIRES_ASYNC(ctx, init_flag_, errors::InvalidArgument("GeOp not Initialize success."), done);
  // ge ge session
  {
    mutex_lock lock{mu_};
    if (!sess_init_flag_) {
      if (job_type_ != "localhost") {  // in ps mode : ctx->session_handle() is empty
        tf_session_ = "ps_worker_session";
        ADP_LOG(INFO) << "[GEOP] get tf session " << tf_session_ << " when in ps mode.";
      }

      if (tf_session_.empty()) {
        tf_session_ = ctx->session_handle();
        ADP_LOG(INFO) << "[GEOP] get tf session " << tf_session_ << " from session handle.";
      }

      bool res = IncrementGraphIdCount(tf_session_, graph_id_);
      if (!res || graph_id_ < kInvalidGraphId) {
        OP_REQUIRES_ASYNC(ctx, false, errors::Unavailable("Get ge session failed."), done);
        return;
      }

      ADP_LOG(INFO) << "[GEOP] Node name: " << ctx->op_kernel().name() << " , tf session: " << tf_session_;

      res = SessionManager::GetInstance().GetOrCreateGeSession(tf_session_, ge_session_, sess_options_);
      if (!res || tf_session_.empty() || ge_session_ == nullptr) {
        OP_REQUIRES_ASYNC(ctx, false, errors::Unavailable("Get ge session failed."), done);
        return;
      }
      if (!init_options_["ge.jobType"].empty() && !init_options_["ge.tuningPath"].empty()) {
        uint32_t device_id = 0;
        OP_REQUIRES_OK_ASYNC(ctx, GetEnvDeviceID(device_id), done);
        ADP_LOG(INFO) << "[GEOP] in tuning func, mstune_mode:" << init_options_["ge.jobType"]
                      << ", work_path:" << init_options_["ge.tuningPath"]
                      << ", op_tune_mode:" << init_options_["op_tune_mode"]
                      << ", distribute_config:" << init_options_["distribute_config"];
        tune_options_.insert(init_options_.begin(), init_options_.end());
        tune_options_.insert({"devices", std::to_string(device_id)});
        tune_options_.insert(sess_options_.begin(), sess_options_.end());
        tune_options_.insert({"work_path", init_options_["ge.tuningPath"]});
        tune_options_.insert({"job_type", init_options_["ge.jobType"]});
        AoeStatus tune_ret = (*aoe_init_)(ge_session_, tune_options_);
        OP_REQUIRES_ASYNC(ctx, tune_ret == AOE_SUCCESS, errors::Internal("[GEOP] exec aoe init func failed."), done);
      }
      ADP_LOG(INFO) << "[GEOP] tf session: " << tf_session_ << " get ge session success.";
      sess_init_flag_ = true;
    }
  }
  string geop_name = ctx->op_kernel().name();
  uint32_t num_inputs = static_cast<uint32_t>(ctx->num_inputs());
  ADP_LOG(INFO) << "[GEOP] Begin GeOp::ComputeAsync"
            << ", kernel_name:" << geop_name << ", num_inputs:" << num_inputs << ", num_outputs:" << ctx->num_outputs();
  int64 startTime = InferShapeUtil::GetCurrentTimestap();
  std::vector<Tensor> input_vec;
  std::vector<std::string> input_shapes;
  std::vector<ge::Tensor> inputs;
  OP_REQUIRES_OK_ASYNC(ctx, (BuildInputTensorInfo(ctx, input_vec, input_shapes, inputs)), done);

  // if input shapes changed, cache graphs
  uint32_t cache_graph_id = graph_id_;
  bool is_set_dynamic_config = !sess_options_["ge.inputShape"].empty() && !sess_options_["ge.dynamicDims"].empty();
  bool is_tuning = !init_options_["ge.jobType"].empty() && !init_options_["ge.tuningPath"].empty();
  bool is_lazy_recompile_mode = dynamic_input_ == "1" && dynamic_graph_execute_mode_ == "lazy_recompile";
  if (is_set_dynamic_config && is_tuning) {
    ADP_LOG(FATAL) << "dynamic input config can not use with mstuning.";
    LOG(FATAL) << "dynamic input config can not use with mstuning.";
  } else if (is_set_dynamic_config && !is_tuning) {
    if (InitRebuildFlag(cache_graph_id) != 0) {
      OP_REQUIRES_ASYNC(ctx, false, errors::Internal("Failed to check rebuild flag"), done);
      return;
    }
  } else if (!is_set_dynamic_config && is_tuning) {
    ADP_LOG(INFO) << "[GEOP] in tune func, do not rebuild graph.";
  } else {
    // in dynamic input mode, cache graphs.
    if (is_lazy_recompile_mode) {
      GetExecGraphId(ctx, cache_graph_id, input_shapes);
    }
    if (InitRebuildFlag(cache_graph_id) != 0) {
      OP_REQUIRES_ASYNC(ctx, false, errors::Internal("Failed to check rebuild flag"), done);
      return;
    }
  }

  if (!build_flag_) {
    // Get Graph
    OP_REQUIRES_ASYNC(ctx, ctx->function_library() != nullptr, errors::Internal("function library is nullptr"), done);
    FunctionLibraryDefinition *flib_def = const_cast<FunctionLibraryDefinition *>(ctx->function_library()->GetFunctionLibraryDefinition());
    OP_REQUIRES_ASYNC(ctx, flib_def != nullptr, errors::Internal("flib_def is nullptr"), done);
    std::shared_ptr<Graph> graph = std::make_shared<Graph>(OpRegistry::Global());
    OP_REQUIRES_ASYNC(ctx, graph != nullptr, errors::Internal("create tensorflow graph failed"), done);

    // Build GraphDef from FunctionDef
    GraphDef ori_graph_def;
    OP_REQUIRES_OK_ASYNC(ctx, BuildGraphDef(*flib_def, input_vec, ori_graph_def, is_initialized_graph_), done);

    /* if graph is init verify graph, return */
    if (this->is_initialized_graph_ == true) {
      Tensor initialized_tensor(ctx->expected_output_dtype(0), TensorShape({0}));
      ctx->set_output(0, initialized_tensor);
      done();
      return;
    }

    char *need_print = getenv("PRINT_MODEL");
    if (need_print != nullptr && strcmp("1", need_print) == 0) {
      string tmpmodel_path = GetDumpPath() + "TF_";
      string tmodel_path = tmpmodel_path + geop_name.c_str() + ".pbtxt";
      Status status_out = WriteTextProto(Env::Default(), tmodel_path, ori_graph_def);
    }
    int64 endTime = InferShapeUtil::GetCurrentTimestap();
    ADP_LOG(EVENT) << "[GEOP] In GEOP computeAsync, kernel_name:" << geop_name << " ,TFadapter cost time: ["
                   << ((endTime - startTime) / kMicrosToMillis) << " ms]";
    ADP_LOG(INFO) << "[GEOP] TFadpter process graph success, GE parser begin, kernel_name:" << geop_name
              << " ,tf session: " << tf_session_ << " ,graph id :" << cache_graph_id;
    // parser,  tensorflow graph to ge graph
    std::shared_ptr<domi::ModelParser> model_parser =
        domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
    OP_REQUIRES_ASYNC(ctx, model_parser != nullptr, errors::Unavailable("create model parser ret failed."), done);
    ge::ComputeGraphPtr compute_graph = nullptr;
    try {
      compute_graph = std::make_shared<ge::ComputeGraph>("ge_default_" + CurrentTimeInStr());
    } catch (...) { OP_REQUIRES_ASYNC(ctx, false, errors::Internal("make shared failed"), done); }

    OP_REQUIRES_ASYNC(ctx, compute_graph != nullptr, errors::InvalidArgument("create ComputeGraph failed"), done);

    auto build_sub_graph = [this, flib_def](const std::string &graph) -> std::string {
      // const tensorflow::GraphDef *graph_def_in = reinterpret_cast<const tensorflow::GraphDef *>(root_proto);
      ADP_LOG(INFO) << "[GEOP] build_sub_graph enter, sub graph name is " << graph;
      const FunctionDef *func_def = flib_def->Find(graph);
      if (func_def == nullptr) {
        ADP_LOG(ERROR) << "[GEOP] Sub graph not found in library, sub graph name is " << graph;
        LOG(ERROR) << "[GEOP] Sub graph not found in library, sub graph name is " << graph;
        return "";
      }
      // get infershape
      Graph subgraph(flib_def);
      Status status = InferShapeUtil::GetSubGraphFromFunctionDef(*flib_def, *func_def, &subgraph);
      if (status != Status::OK()) {
        ADP_LOG(ERROR) << "[GEOP] Get subgraph from functiondef fail:" << status.error_message();
        LOG(ERROR) << "[GEOP] Get subgraph from functiondef fail:" << status.error_message();
        return "";
      }
      ADP_LOG(INFO) << "[GEOP] Get subgraph from functiondef success.";
      char *enable_force_v2_control = getenv("ENABLE_FORCE_V2_CONTROL");
      if (enable_force_v2_control != nullptr && strcmp("1", enable_force_v2_control) == 0) {
        GraphDef graph_def;
        subgraph.ToGraphDef(&graph_def);
        WriteTextProto(Env::Default(), graph + "_graph.pbtxt", graph_def);
      }

      bool is_initialize = false;
      for (Node *node : subgraph.nodes()) {
        AddNodeAttrs(node, is_initialize);

        // Add Input&Output Desc into NodeDef
        if (GenerateDesc(node) != Status::OK()) {
          ADP_LOG(WARNING) << "[GEOP] name: " << node->name() << " op:" << node->type_string()
                           << " Generate desc failed in subgraph.";
          LOG(WARNING) << "[GEOP] name: " << node->name() << " op:" << node->type_string()
                       << " Generate desc failed in subgraph.";
        }
      }

      unique_ptr<GraphDef> sub_graph_def(new (std::nothrow) GraphDef());
      if (sub_graph_def == nullptr) {
        ADP_LOG(ERROR) << "[GEOP] Malloc memory for subgraph def fail.";
        LOG(ERROR) << "[GEOP] Malloc memory for subgraph def fail.";
        return "";
      }
      subgraph.ToGraphDef(sub_graph_def.get());
      if (enable_force_v2_control != nullptr && strcmp("1", enable_force_v2_control) == 0) {
        sub_graph_def->release_library();
        sub_graph_def->mutable_versions()->clear_min_consumer();
      }

      char *need_print = getenv("PRINT_MODEL");
      if (need_print != nullptr && strcmp("1", need_print) == 0) {
        string tmpmodel_path = GetDumpPath() + "TF_Subgraph_";
        string tmodel_path = tmpmodel_path + graph.c_str() + ".pbtxt";
        Status status_out = WriteTextProto(Env::Default(), tmodel_path, *sub_graph_def);
      }
      ADP_LOG(INFO) << "[GEOP] build_sub_graph exit, sub graph name is " << graph;
      return sub_graph_def->SerializeAsString();
    };

    ge::Status status = model_parser->ParseProtoWithSubgraph(ori_graph_def.SerializeAsString(),
                                                             build_sub_graph, compute_graph);
    if (status != ge::SUCCESS) {
      std::string error_message = ge::GEGetErrorMsg();
      std::stringstream ss;
      ss << "graph parse failed. ret : " << status << std::endl
         << "Error Message is : " << std::endl
         << error_message;
      OP_REQUIRES_ASYNC(ctx, status == ge::SUCCESS, errors::Internal(ss.str()), done);
    }

    domi::GetContext().format = ge::GetParserContext().format;

    ADP_LOG(INFO) << "[GEOP] Tensorflow graph parse to ge graph success, kernel_name:" << geop_name
              << " ,tf session: " << tf_session_ << " ,graph id: " << cache_graph_id;

    size_t nodes = compute_graph->GetAllNodesSize();
    if (nodes == 0) {
      build_flag_ = true;
      compute_graph_empty_ = true;
      int64 endTime = InferShapeUtil::GetCurrentTimestap();
      ADP_LOG(INFO) << "[GEOP] End GeOp::ComputeAsync, compute_graph is empty, kernel_name:" << geop_name
                << ", ret_status:" << ToString(ge::SUCCESS) << " , tf session: " << tf_session_
                << " ,graph id: " << cache_graph_id << " [" << ((endTime - startTime) / kMicrosToMillis) << " ms]";
      done();
      return;
    }

    // convert to ge::graph
    ge::Graph ge_graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    if (iteration_per_loop_ > 1) {
      ge_graph.SetNeedIteration(this->need_iteration_);
    }

    if (is_host_graph_) {
      ADP_LOG(INFO) << "[GEOP] set graph option.";
      graph_options_["ge.exec.placement"] = "HOST";
    }
    if (dynamic_input_ == "1") {
      graph_options_["ge.exec.dynamicInput"] = dynamic_input_;
      graph_options_["ge.exec.dynamicGraphExecuteMode"] = dynamic_graph_execute_mode_;
      graph_options_["ge.exec.dataInputsShapeRange"] = data_inputs_shape_range_;
    }
    if (is_tuning) {
      if (is_train_graph_ != "1" && init_options_["ge.jobType"] != "2") {
        ADP_LOG(INFO) << "[GEOP] in tune mode, nontraining graphs should be cache.";
        OP_REQUIRES_ASYNC(ctx, SessionManager::GetInstance().CacheGeGraphs(ge_session_, ge_graph),
          errors::Internal("[GEOP] cache ge session failed."), done);
        build_flag_ = true;
        BuildOutTensorInfo(ctx);
        done();
        return;
      } else {
        ADP_LOG(INFO) << "[GEOP] in tune mode, training graph handled by tools.";
        std::vector<ge::Graph> ge_graphs;
        OP_REQUIRES_ASYNC(ctx, SessionManager::GetInstance().GetGeGraphs(ge_session_, ge_graphs),
          errors::Internal("[GEOP] ge ge session nontraining graphs failed."), done);
        tune_options_.insert(graph_options_.begin(), graph_options_.end());
        AoeStatus tune_ret = (*aoe_tuning_)(ge_graph, ge_graphs, ge_session_, tune_options_);
        OP_REQUIRES_ASYNC(ctx, tune_ret == AOE_SUCCESS, errors::Internal("[GEOP] exec aoe tuning func failed."), done);
        ADP_LOG(INFO) << "[GEOP] msTuning success.";
        build_flag_ = true;
        BuildOutTensorInfo(ctx);
        done();
        return;
      }
    }

    // call ge session addGraph api
    status = ge_session_->AddGraph(cache_graph_id, ge_graph, graph_options_);
    if (status != ge::SUCCESS) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
      ADP_LOG(FATAL) << "[GEOP] call ge session add graph failed, kernel: " << geop_name << " ,tf session: "
                     << tf_session_ << ", graph id: " << cache_graph_id;

      std::string error_message = ge::GEGetErrorMsg();
      std::stringstream ss;
      ss << "[GEOP] call ge session add graph failed, kernel: " << geop_name
         << ", tf session: " << tf_session_
         << ", graph id: " << cache_graph_id << std::endl
         << "Error Message is : " << std::endl
         << error_message;
      LOG(FATAL) << ss.str();
      OP_REQUIRES_ASYNC(ctx, status == ge::SUCCESS, errors::Unavailable(ss.str()), done);
    } else {
      add_graph_flag_ = true;
      ADP_LOG(INFO) << "[GEOP] Add graph to ge session success, kernel_name:" << geop_name
                << " ,tf session: " << tf_session_ << " ,graph id:" << cache_graph_id;
    }
    build_flag_ = true;
    if (!is_set_dynamic_config && is_lazy_recompile_mode) {
      cache_graphs_.insert(std::make_pair(input_shapes, cache_graph_id));
      graph_counts_.push_back(std::make_pair(input_shapes, 1));
    }
    if (need_compile_graph_first_) {
      ge::Status status = ge_session_->BuildGraph(cache_graph_id, inputs);
      if (status != ge::SUCCESS) {
        std::string error_message = ge::GEGetErrorMsg();
        std::stringstream ss;
        ss << "[GEOP] GE session build graph failed, domi_ret : " << status << std::endl
           << "Error Message is : " << std::endl
           << error_message;
        OP_REQUIRES_ASYNC(ctx, status == ge::SUCCESS, errors::Unavailable(ss.str()), done);
      }

      ADP_LOG(INFO) << "[GEOP] Build graph success.";
      done();
      return;
    }
  } else {
    if (compute_graph_empty_) {
      int64 endTime = InferShapeUtil::GetCurrentTimestap();
      ADP_LOG(INFO) << "[GEOP] End GeOp::ComputeAsync, compute_graph is empty, kernel_name:" << geop_name
                << ", ret_status:" << ToString(ge::SUCCESS) << " , tf session: " << tf_session_
                << " ,graph id: " << cache_graph_id << " [" << ((endTime - startTime) / kMicrosToMillis) << " ms]";
      done();
      return;
    }
  }

  if (is_tuning) {
    ADP_LOG(INFO) << "in mstune mode, graph only execute once, The remaining steps return directly.";
    BuildOutTensorInfo(ctx);
    done();
    return;
  }

  int64 run_start_time = InferShapeUtil::GetCurrentTimestap();
  auto callback = [done, ctx, run_start_time](ge::Status ge_status, std::vector<ge::Tensor> &outputs) {
    if (ge_status == ge::SUCCESS) {
      if (BuildOutputTensorInfo(ctx, outputs) != Status::OK()) {
        ADP_LOG(FATAL) << ctx->op_kernel().name() << " GEOP::DoRunAsync get output failed.";
        LOG(FATAL) << ctx->op_kernel().name() << " GEOP::DoRunAsync get output failed.";
      }
    } else if (ge_status == ge::END_OF_SEQUENCE) {
      ctx->SetStatus(errors::OutOfRange("End of sequence"));
      ADP_LOG(WARNING) << "[GEOP] Out of range: End of sequence.";
      LOG(WARNING) << "[GEOP] Out of range: End of sequence.";
    } else if (ge_status != ge::SUCCESS) {
      tensorflow::Status tfStatus = errors::Unavailable(ToString(ge_status));
      ctx->CtxFailureWithWarning(tfStatus);
      std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
      ADP_LOG(FATAL) << ctx->op_kernel().name() << "GEOP::::DoRunAsync Failed";
      std::string error_message = ge::GEGetErrorMsg();
      LOG(FATAL) << ctx->op_kernel().name() << "GEOP::::DoRunAsync Failed" << std::endl
                 << "Error Message is : " << std::endl
                 << error_message;
    }
    int64 run_end_time = InferShapeUtil::GetCurrentTimestap();
    ADP_LOG(INFO) << "[GEOP] RunGraphAsync callback, status:" << ge_status << ", kernel_name:"
                  << ctx->op_kernel().name() << "[ " << (run_end_time - run_start_time) << "us]";
    done();
  };

  ADP_LOG(INFO) << "[GEOP] Call ge session RunGraphAsync, kernel_name:" << geop_name << " ,tf session: " << tf_session_
                << " ,graph id: " << cache_graph_id;
  // call ge session runGraphAsync api
  ge::Status status = ge_session_->RunGraphAsync(cache_graph_id, inputs, callback);
  if (status != ge::SUCCESS) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
    ADP_LOG(FATAL) << "[GEOP] call ge session RunGraphAsync Failed, kernel:" << geop_name << " ,tf session: "
                   << tf_session_ << " ,graph id: " << cache_graph_id;
    std::string error_message = ge::GEGetErrorMsg();
    std::stringstream ss;
    ss << "[GEOP] call ge session RunGraphAsync Failed, kernel:" << geop_name
       << ", tf session: " << tf_session_
       << ", graph id: " << cache_graph_id << std::endl
       << "Error Message is : " << std::endl
       << error_message;
    LOG(FATAL) << ss.str();
    OP_REQUIRES_ASYNC(ctx, status == ge::SUCCESS, errors::Unavailable(ss.str()), done);
  }

  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "[GEOP] End GeOp::ComputeAsync, kernel_name:" << geop_name << ", ret_status:" << ToString(status)
                << " ,tf session: " << tf_session_ << " ,graph id: " << cache_graph_id << " ["
                << ((endTime - startTime) / kMicrosToMillis) << " ms]";
  return;
}

void GeOp::AddNodeAttrs(Node *node, bool &is_initialize) {
  // Add dp custom kernel label
  if (node->type_string() == "IteratorGetNext") {
    node->AddAttr("_kernel", "dp");
    if (dynamic_input_ == "1") {
      node->AddAttr("_dynamic_graph_execute_mode", dynamic_graph_execute_mode_);
      node->AddAttr("_getnext_inputs_shape_range", getnext_inputs_shape_range_);
    }
  }
  if (node->type_string() == "Assert" || node->type_string() == "Print" || node->type_string() == "PrintV2") {
    node->AddAttr("_kernel", "extend");
  }
  NodeDef &node_def = const_cast<NodeDef &>(node->def());
  if (node_def.op() == "Where") { is_initialize = InferShapeUtil::IsInitializedGraph(node); }
  if (node->name() == "IterationOp") {
    this->need_iteration_ = true;
    ADP_LOG(INFO) << "subgraph  has iteration op.";
  }
  if (node->name().find("var_in_host") != std::string::npos) {
    is_host_graph_ = true;
    ADP_LOG(INFO) << "[GEOP] variable subgraph is initialized in host.";
  }
  if (!need_compile_graph_first_) {
    if (node->name().find("NpuCompile") != std::string::npos) {
      need_compile_graph_first_ = true;
      ADP_LOG(INFO) << "[GEOP] set subgraph compile first.";
    }
  }
  // clear device info && attr
  node_def.set_device("");
  if (node_def.op() == "Const") {
    node_def.mutable_attr()->erase("data_format");
    node_def.mutable_attr()->erase("cce_format");
    node_def.mutable_attr()->erase("output_type");
  }
}

// Build GraphDef from FunctionDef.
Status GeOp::BuildGraphDef(FunctionLibraryDefinition &flib_def,
                           const std::vector<Tensor> &input_vec, GraphDef &graph_def, bool &is_initialize) {
  const FunctionDef *function_def = flib_def.Find(function_.name());
  if (function_def == nullptr) {
    return errors::Internal("%s: fdef is nullptr", function_.name());
  }
  // get infershape
  Graph graph(OpRegistry::Global());
  Status ret = InferShapeUtil::InferShape(input_vec, &flib_def, function_def, &graph);
  if (!ret.ok()) {
    ADP_LOG(ERROR) << "[GEOP] InferShape failed, " << ret.error_message();
    LOG(ERROR) << "[GEOP] InferShape failed, " << ret.error_message();
    return ret;
  }

  bool is_set_dynamic_config = !sess_options_["ge.inputShape"].empty() && !sess_options_["ge.dynamicDims"].empty() &&
                               !sess_options_["ge.dynamicNodeType"].empty();
  if (is_set_dynamic_config) { BuildShapeNodeAndCacheArgNodes(graph); }

  bool is_tuning = !init_options_["ge.jobType"].empty() && !init_options_["ge.tuningPath"].empty();
  for (Node *node : graph.nodes()) {
    AddNodeAttrs(node, is_initialize);
    // Add Input&Output Desc into NodeDef
    ret = this->GenerateDesc(node);
    if (!ret.ok()) {
      ADP_LOG(ERROR) << "[GEOP] node: " << node->name() << " GenerateDesc failed, "
                     << ret.error_message();
      LOG(ERROR) << "[GEOP] node: " << node->name() << " GenerateDesc failed, "
                 << ret.error_message();
      return ret;
    }
    if (is_tuning) {
      // output handle
      NodeDef &node_def = const_cast<NodeDef &>(node->def());
      if (node->type_string() == "_Retval") {
        int index = node_def.attr().at("index").i();
        // format: AttrValue.list(ListValue).func(repeated NameAttrList)
        NameAttrList desc_attr = node_def.attr().at(INPUT_DESC).list().func(0);

        std::vector<int64> dims;
        int dim_num = desc_attr.attr().at(SERIALIZE_SHAPE).list().i_size();
        for (int t = 0; t < dim_num; t++) {
          int64 dim_i = (int64_t) desc_attr.attr().at(SERIALIZE_SHAPE).list().i(t);
          if (dim_i < 0) { dim_i = 1; }
          dims.push_back(dim_i);
        }

        TensorShape out_shape(dims);
        outputs_shape_.insert(std::map<int, TensorShape>::value_type(index, out_shape));
      }
    }
  }
  // set input_shape to dynamic nodes shape desc
  if (is_set_dynamic_config) {
    ret = ChangeInputsShapeDesc();
    if (!ret.ok()) {
      ADP_LOG(ERROR) << "[GEOP] ChangeInputsShapeDesc failed, " << ret.error_message();
      LOG(ERROR) << "[GEOP] ChangeInputsShapeDesc failed, " << ret.error_message();
      return ret;
    }
  }
  graph.ToGraphDef(&graph_def);
  char *enable_force_v2_control = getenv("ENABLE_FORCE_V2_CONTROL");
  if (enable_force_v2_control != nullptr && strcmp("1", enable_force_v2_control) == 0) {
    WriteTextProto(Env::Default(), function_.name() + "_v1.pbtxt", graph_def);

    Status status = FunctionalizeControlFlow(&graph, &flib_def);
    if (status != Status::OK()) {
      LOG(WARNING) << "[GEOP] Failed functionalize control flow: " << status.error_message();
      return Status::OK();
    }
    graph.ToGraphDef(&graph_def);
    WriteTextProto(Env::Default(), function_.name() + "_v2.pbtxt", graph_def);
  }
  return Status::OK();
}

void GeOp::BuildShapeNodeAndCacheArgNodes(Graph &graph) {
  std::string dynamic_node_type = sess_options_["ge.dynamicNodeType"];
  for (Node *node : graph.nodes()) {
    // add shape node to get getnext node real shape
    if (dynamic_node_type == "0" && node->type_string() == "IteratorGetNext") {
      dynamic_shape_nodes_.emplace_back(node);
      int i = 0;
      for (auto out_edge : node->out_edges()) {
        if (!out_edge->IsControlEdge()) {
          std::string shape_name = "getnext_shape_" + std::to_string(i);
          Node *shape_node = nullptr;
          TF_CHECK_OK(NodeBuilder(shape_name, "Shape")
                      .Input(node, out_edge->src_output())
                      .Device(node->def().device())
                      .Finalize(&graph, &shape_node));
          std::string identity_name = "shape_identity_" + std::to_string(i);
          Node *identity_node = nullptr;
          TF_CHECK_OK(NodeBuilder(identity_name, "Identity")
                      .Input(shape_node, 0)
                      .Device(shape_node->def().device())
                      .Finalize(&graph, &identity_node));
        }
        i++;
      }
    }
    // count data args and getnext args for dynamic dims
    if (node->type_string() == "_Arg") {
      if (node->name().find("IteratorGetNext_") != std::string::npos) {
        if (dynamic_node_type == "0") { dynamic_shape_nodes_.emplace_back(node); }
      } else {
        if (dynamic_node_type == "1") { dynamic_shape_nodes_.emplace_back(node); }
      }
    }
  }
  // sort dynamic nodes to match input_shapes
  std::sort(dynamic_shape_nodes_.begin(), dynamic_shape_nodes_.end(), CmpVecValue);
}

Status GeOp::ChangeInputsShapeDesc() {
  std::vector<std::string> result;
  std::string input_shapes = sess_options_["ge.inputShape"];
  Split(input_shapes, result, ";"); //e.g. result:["data:2,3", "data1:3,4"]

  if (dynamic_shape_nodes_.size() == 1 && dynamic_shape_nodes_[0]->type_string() == "IteratorGetNext") {
    ADP_LOG(INFO) << "[GEOP] change " << dynamic_shape_nodes_[0]->name() << " shape desc.";
    if (dynamic_shape_nodes_[0]->num_outputs() != static_cast<int32>(result.size())) {
      return errors::InvalidArgument("input_shape is not match inputs num in graph");
    }
    NodeDef &node_def = const_cast<NodeDef &>(dynamic_shape_nodes_[0]->def());
    AttrValue &output_tensor_descs = (*node_def.mutable_attr())[OUTPUT_DESC];
    for (int32 i = 0; i < dynamic_shape_nodes_[0]->num_outputs(); ++i) {
      AttrValue attr_shape_value;
      attr_shape_value.set_type(DT_INT32);
      SetShapesToOutputDesc(result, i, attr_shape_value);
      (*output_tensor_descs.mutable_list()->mutable_func(i)->mutable_attr())[SERIALIZE_SHAPE] = attr_shape_value;
    }
  } else {
    if (!dynamic_shape_nodes_.empty()) {
      if (dynamic_shape_nodes_.size() != result.size()) {
        return errors::InvalidArgument("input_shape is not match inputs num in graph");
      }
    }
    for (size_t i = 0; i < dynamic_shape_nodes_.size(); ++i) {
      ADP_LOG(INFO) << "[GEOP] change " << dynamic_shape_nodes_[i]->name() << " shape desc.";
      NodeDef &node_def = const_cast<NodeDef &>(dynamic_shape_nodes_[i]->def());
      AttrValue &output_tensor_descs = (*node_def.mutable_attr())[OUTPUT_DESC];
      AttrValue attr_shape_value;
      attr_shape_value.set_type(DT_INT32);
      SetShapesToOutputDesc(result, i, attr_shape_value);
      (*output_tensor_descs.mutable_list()->mutable_func(0)->mutable_attr())[SERIALIZE_SHAPE] = attr_shape_value;
    }
  }
  ADP_LOG(INFO) << "[GEOP] change input shapes desc success.";
  return Status::OK();
}

void GeOp::SetShapesToOutputDesc(const std::vector<std::string> &input_shapes,
                                 const int &index, AttrValue &attr_shape_value) {
  if (input_shapes.empty()) {
    ADP_LOG(ERROR) << "[GEOP] input_shapes is empty.";
    LOG(ERROR) << "[GEOP] input_shapes is empty.";
    return;
  }
  if (index < 0) {
    ADP_LOG(ERROR) << "[GEOP] index must more than 0.";
    LOG(ERROR) << "[GEOP] index must more than 0.";
    return;
  }
  ADP_LOG(INFO) << "[GEOP] get input: " << index << " input shape is: " << input_shapes[index];
  std::vector<std::string> shape;
  Split(input_shapes[index], shape, ":"); // e.g. shape:["data", "2,3,4"]
  if (shape.empty() || shape.size() != 2) {
    ADP_LOG(ERROR) << "[GEOP] shape is empty or shape size is not 2.";
    LOG(ERROR) << "[GEOP] shape is empty or shape size is not 2.";
    return;
  }
  if (shape[1] == "0") {
    // scale node has no shape.
    return;
  }
  std::vector<std::string> dims;
  Split(shape[1], dims, ","); // e.g. dims:["2", "3", "4"]
  for (auto dim : dims) {
    attr_shape_value.mutable_list()->add_i(std::atoi(dim.c_str()));
  }
}

void GeOp::AnalyzeInputDesc(void *tensor_ptr, ge::Tensor &input, ge::DataType type,
                            std::vector<std::string> &input_shapes) {
  ADP_LOG(INFO) << "[GEOP] Start analyze input tensor.";
  NpuGetNextOutputInfo *output_info = static_cast<NpuGetNextOutputInfo *>(tensor_ptr);
  std::vector<int64> tmp_dims;
  for (int64_t dim : output_info->dims_) {
    tmp_dims.push_back(dim);
  }

  TensorShape input_shape(tmp_dims);
  input_shapes.push_back(input_shape.DebugString());

  ge::Shape ge_shape(output_info->dims_);
  ge::TensorDesc ge_tensor_desc(ge_shape);
  ge_tensor_desc.SetDataType(type);
  ge_tensor_desc.SetPlacement(output_info->placement_);
  input.SetTensorDesc(ge_tensor_desc);

  uint8_t* data = output_info->data_.get();
  input.SetData(output_info->data_.get(), output_info->output_size_, output_info->data_.get_deleter());
  ADP_LOG(INFO) << "[GEOP] Get input shape:" << input_shape.DebugString()
                << ", input placement:" << output_info->placement_
                << ", input length:" << output_info->output_size_
                << ", input data addr:" << reinterpret_cast<uintptr_t>(data);
}

Status GeOp::BuildInputTensorInfo(OpKernelContext *ctx,
                                  std::vector<Tensor> &input_vec,
                                  std::vector<std::string> &input_shapes,
                                  std::vector<ge::Tensor> &inputs) {
  // ctx is not nullptr
  int num_inputs = ctx->num_inputs();
  std::string cur_input_shapes;

  // populate inputs
  for (int i = 0; i < num_inputs; i++) {
    Tensor tensor(ctx->input(i));
    ADP_LOG(INFO) << "[GEOP] Input tensor " << i << " shape: " << tensor.shape().DebugString();
    DataType data_type = tensor.dtype();
    size_t total_bytes = tensor.TotalBytes();
    void *tensor_ptr = DMAHelper::base(&tensor);

    ge::Tensor input;
    std::shared_ptr<domi::ModelParser> model_parser =
        domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
    REQUIRES_NOT_NULL(model_parser);
    ge::DataType type = model_parser->ConvertToGeDataType(static_cast<uint32_t>(data_type));
    if (type == ge::DT_UNDEFINED) {
      ADP_LOG(ERROR) << "[GEOP] No Supported datatype : " << data_type;
      LOG(ERROR) << "[GEOP] No Supported datatype : " << data_type;
      return errors::InvalidArgument("No Supported datatype : ", data_type);
    }
    if (is_dynamic_getnext_ == "1" && (placeholder_index_.find(std::to_string(i)) == std::string::npos)) {
      REQUIRES_NOT_NULL(tensor_ptr);
      AnalyzeInputDesc(tensor_ptr, input, type, input_shapes);
    } else {
      std::vector<int64_t> dims;
      std::string input_shape = tensor.shape().DebugString();
      for (uint32_t dim : tensor.shape().dim_sizes()) { dims.push_back(static_cast<int64_t>(dim)); }
      ge::Shape ge_shape(dims);
      ge::TensorDesc ge_tensor_desc(ge_shape);
      ge_tensor_desc.SetDataType(type);
      input.SetTensorDesc(ge_tensor_desc);
      input.SetData(static_cast<uint8_t *>(tensor_ptr), total_bytes, [](uint8_t *) {});
      input_shapes.push_back(input_shape);
      cur_input_shapes += input_shape;
    }
    inputs.push_back(input);
    input_vec.push_back(tensor);
  }
  if (sess_options_["ge.inputShape"].empty()) {
    if (!cur_input_shapes.empty() && input_shapes_.empty()) {
      input_shapes_ = cur_input_shapes;
    } else if (input_shapes_ != cur_input_shapes && dynamic_input_ != "1") {
      return errors::Internal("The input shape of ", ctx->op_kernel().name(),
                              " is dynamic, please ensure that npu option[dynamic_input] is set"
                              " correctly, for more details please refer to the migration guide.");
    }
  }
  return Status::OK();
}

Status GeOp::BuildOutTensorInfo(OpKernelContext *ctx) {
  int num_outputs = ctx->num_outputs();
    // populate outputs
  for (int i = 0; i < num_outputs; i++) {
    TensorShape out_shape = outputs_shape_.at(i);
    Tensor *tensor = nullptr;
    TF_RETURN_IF_ERROR(ctx->allocate_output(i, out_shape, &tensor));
  }
  return Status::OK();
}

// For each NodeDef, Create Input&Output Desc(shape,format,dataType)
Status GeOp::GenerateDesc(Node *&node) {
  REQUIRES_NOT_NULL(node);
  NodeDef &node_def = const_cast<NodeDef &>(node->def());
  const OpDef &op_def = node->op_def();
  if (dynamic_input_ == "1" && node->type_string() == "IteratorGetNext") {
    node_def.set_op("DynamicGetNext");
  }

  std::string format = this->data_format_;  // format
  int32_t domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_RESERVED;
  TF_RETURN_IF_ERROR(this->DomiFormatFromString(format, domi_format));

  // Get signature(dataType) from the OpDef & NodeDef
  DataTypeVector inputs;
  DataTypeVector outputs;
  TF_RETURN_IF_ERROR(tensorflow::InOutTypesForNode(node_def, op_def, &inputs, &outputs));

  int num;
  Node *in_node = nullptr;
  const Edge *in_edge = nullptr;

  if (inputs.size() > INT_MAX) { return errors::InvalidArgument("inputs size should be less than INT_MAX."); }

  // Create input Desc
  int inputs_size = static_cast<int>(inputs.size());
  if (inputs_size > 0) {
    AttrValue input_tensor_descs;
    AttrValue input_tensor_descs_s;
    num = 0;
    for (; num < inputs_size;) {
      node->input_node(num, &in_node);
      node->input_edge(num, &in_edge);
      REQUIRES_NOT_NULL(in_node);
      REQUIRES_NOT_NULL(in_edge);
      int src_output = in_edge->src_output();
      if (in_node->def().attr().find(OUTPUT_DESC) != in_node->def().attr().end()) {
        const AttrValue_ListValue &attr_list = in_node->def().attr().at(OUTPUT_DESC).list();
        if (attr_list.func_size() > src_output) {
          NameAttrList desc_attr = in_node->def().attr().at(OUTPUT_DESC).list().func(src_output);
          *(input_tensor_descs.mutable_list()->add_func()) = desc_attr;
        } else {
          NameAttrList name_attr_list;
          name_attr_list.set_name(std::to_string(0));
          AttrValue attr_format_value;
          attr_format_value.set_i((int64_t)domi_format);
          name_attr_list.mutable_attr()->insert({SERIALIZE_FORMAT, attr_format_value});
          AttrValue attr_datatype_value;
          attr_datatype_value.set_i((int64_t)inputs[num]);
          name_attr_list.mutable_attr()->insert({SERIALIZE_DATATYPE, attr_datatype_value});
          AttrValue attr_shape_value;
          attr_shape_value.set_type(DT_INT32);
          name_attr_list.mutable_attr()->insert({SERIALIZE_SHAPE, attr_shape_value});
          *(input_tensor_descs.mutable_list()->add_func()) = name_attr_list;
        }
      } else {
        ADP_LOG(INFO) << "[GEOP] no OUTPUT_DESC: " << node->name() << " <-- " << in_node->name();
        if (num > 0 && node->type_string() == "Merge" && in_node->type_string() == "NextIteration") {
          node->input_node(num - 1, &in_node);
          node->input_edge(num - 1, &in_edge);
          REQUIRES_NOT_NULL(in_node);
          REQUIRES_NOT_NULL(in_edge);
          int src_output = in_edge->src_output();
          NameAttrList desc_attr = in_node->def().attr().at(OUTPUT_DESC).list().func(src_output);
          *(input_tensor_descs.mutable_list()->add_func()) = desc_attr;
        }
      }
      num++;
    }
    REQUIRES_NOT_NULL(node_def.mutable_attr());
    node_def.mutable_attr()->insert({INPUT_DESC, input_tensor_descs});
  }

  // Create output Desc
  if (outputs.size() > 0) {
    // Get infershape
    const std::string KEY_SHAPE = tensorflow::KEY_SHAPE;
    AttrValue shape_value;
    const auto &it = node_def.attr().find(KEY_SHAPE);
    if (it == node_def.attr().end()) {  // no find
      ADP_LOG(WARNING) << "[GEOP] There is no infershape of node : " << node_def.name();
      LOG(WARNING) << "[GEOP] There is no infershape of node : " << node_def.name();
    } else {
      shape_value = node_def.attr().at(KEY_SHAPE);
      uint32_t shape_size = static_cast<uint32_t>(shape_value.list().shape_size());
      if (shape_size != outputs.size()) {
        ADP_LOG(ERROR) << "[GEOP] size not equal, shape_size : " << shape_size << " outputs size:" << outputs.size();
        LOG(ERROR) << "[GEOP] size not equal, shape_size : " << shape_size << " outputs size:" << outputs.size();
        shape_value.clear_list();
      }
    }
    // Create output Desc
    AttrValue output_tensor_descs;
    AttrValue output_tensor_descs_s;
    int i = 0;
    num = 0;
    for (DataType data_type : outputs) {
      string desc_string_s;
      AttrValue attr_format_value;
      attr_format_value.set_i((int64_t) domi_format);
      AttrValue attr_datatype_value;
      attr_datatype_value.set_i((int64_t) data_type);

      // shape
      AttrValue attr_shape_value;
      attr_shape_value.set_type(DT_INT32);
      if (shape_value.has_list()) {
        TensorShapeProto shape_proto = shape_value.list().shape(num);
        for (int j = 0; j < shape_proto.dim_size(); j++) {
          attr_shape_value.mutable_list()->add_i(shape_proto.dim(j).size());
        }
      }

      NameAttrList name_attr_list;
      name_attr_list.set_name(std::to_string(i));
      REQUIRES_NOT_NULL(name_attr_list.mutable_attr());
      name_attr_list.mutable_attr()->insert({SERIALIZE_FORMAT, attr_format_value});
      name_attr_list.mutable_attr()->insert({SERIALIZE_DATATYPE, attr_datatype_value});
      name_attr_list.mutable_attr()->insert({SERIALIZE_SHAPE, attr_shape_value});
      REQUIRES_NOT_NULL(output_tensor_descs.mutable_list());
      *(output_tensor_descs.mutable_list()->add_func()) = name_attr_list;

      num++;
      i++;
    }
    node_def.mutable_attr()->erase(KEY_SHAPE);
    node_def.mutable_attr()->insert({OUTPUT_DESC, output_tensor_descs});
  }
  string op_def_string;
  op_def.SerializeToString(&op_def_string);

  tensorflow::AttrValue value;
  value.set_s(op_def_string);
  node_def.mutable_attr()->insert({"op_def", value});
  return tensorflow::Status::OK();
}

Status GeOp::DomiFormatFromString(std::string format, int32_t &domi_format) {
  if (format == "NCHW") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_NCHW;
    return Status::OK();
  } else if (format == "NHWC") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_NHWC;
    return Status::OK();
  } else if (format == "NC1HWC0") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_NC1HWC0;
    return Status::OK();
  } else if (format == "NDHWC") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_NDHWC;
    return Status::OK();
  } else if (format == "NCDHW") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_NCDHW;
    return Status::OK();
  } else if (format == "DHWCN") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_DHWCN;
    return Status::OK();
  } else if (format == "DHWNC") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_DHWNC;
    return Status::OK();
  } else if (format == "FRACTALZ") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_FRACTAL_Z;
  }
  return errors::Unavailable("DomiFormatFromString, not supported format, format = ", format);
}
}  // namespace tensorflow

namespace tensorflow {
mutex GeOp::mu_(LINKER_INITIALIZED);

const std::string GeOp::INPUT_DESC = "input_tensor_desc";
const std::string GeOp::OUTPUT_DESC = "output_tensor_desc";
const std::string GeOp::SERIALIZE_FORMAT = "serialize_format";
const std::string GeOp::SERIALIZE_DATATYPE = "serialize_datatype";
const std::string GeOp::SERIALIZE_SHAPE = "serialize_shape";
const std::string GeOp::SubGraph = "SubGraph";
std::unordered_map<std::string, uint32_t> GeOp::session_and_graph_id_map_;

REGISTER_KERNEL_BUILDER(Name("GeOp").Device(DEVICE_CPU), GeOp);
}  // namespace tensorflow
