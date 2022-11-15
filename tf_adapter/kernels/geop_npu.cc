/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
#include <algorithm>

#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/ge_plugin.h"
#include "tf_adapter/util/infershape_util.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/generate_report.h"
#include "tf_adapter/util/npu_ops_identifier.h"
#include "tf_adapter/util/session_manager.h"

#ifdef TF_VERSION_TF2
#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"
#endif
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/env_var.h"

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/scope_guard.h"
#include "framework/common/string_util.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_api.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "parser/onnx_parser.h"
#include "ge/ge_api.h"
#include "ge/ge_api_types.h"

#include "graph/ascend_string.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/def_types.h"
#include "graph/model.h"
#include "tf_adapter_2.x/npu_device/core/npu_micros.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"

namespace tensorflow {
#ifdef TF_VERSION_TF2
Status FunctionalizeControlFlow(Graph *graph, FunctionLibraryDefinition *library, const NodeFilter &node_filter = {},
                                bool include_functions = false);
#else
Status FunctionalizeControlFlow(Graph *graph, FunctionLibraryDefinition *library);
#endif
namespace {
const std::string ATTR_NAME_CONST_INPUT_NAME = "_const_input";
const std::string kMdatTuning = "mdat";
const std::string kAutoRecompute = "auto";
using geDataUniquePtr = std::unique_ptr<uint8_t[], std::function<void(uint8_t *)>>;

class NpuHostFixedAllocator : public tensorflow::Allocator, public tensorflow::core::RefCounted {
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
  std::string Name() override {
    return "NpuHostFixedAllocator";
  }
  void *AllocateRaw(size_t alignment, size_t num_bytes) override {
    (void) alignment;
    (void) num_bytes;
    return ptr_.get();
  }
  void DeallocateRaw(void *ptr) override {
    (void) ptr;
    Unref();
  }
  geDataUniquePtr ptr_;
};

class NpuGetNextOutputInfo {
 public:
  NpuGetNextOutputInfo(ge::Placement placement, std::vector<int64_t> &dims, size_t output_size, geDataUniquePtr data)
      : placement_(placement), dims_(dims), output_size_(output_size), data_(std::move(data)) {}
  ~NpuGetNextOutputInfo() {
    ADP_LOG(INFO) << "[GEOP] Release NpuGetNextOutputInfo.";
  }
  ge::Placement placement_;
  std::vector<int64_t> dims_;
  size_t output_size_;
  geDataUniquePtr data_;
};

class NpuHostGetNextAllocator : public tensorflow::Allocator, public tensorflow::core::RefCounted {
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
  std::string Name() override {
    return "NpuHostGetNextAllocator";
  }
  void *AllocateRaw(size_t alignment, size_t num_bytes) override {
    (void) alignment;
    (void) num_bytes;
    return output_.get();
  }
  void DeallocateRaw(void *ptr) override {
    (void) ptr;
    Unref();
  }
  std::unique_ptr<NpuGetNextOutputInfo> output_;
};

inline string ToString(ge::Status status) {
  return ::ge::StatusFactory::Instance()->GetErrDesc(status);
}

Status BuildStringOutput(geDataUniquePtr data_ptr, size_t output_size, Tensor &cpu_tensor) {
  TensorShape out_shape = cpu_tensor.shape();
  if ((out_shape.num_elements() * sizeof(ge::StringHead)) >= output_size) {
    LOG(ERROR) << "[GEOP] Graph engine process success but output string format is not right";
    return errors::Internal("Graph engine process graph success but output string format is not right.");
  }
  auto tensor_flat = cpu_tensor.flat<tstring>();
  tstring* tensor_data = tensor_flat.data();
  ge::StringHead *string_head = reinterpret_cast<ge::StringHead *>(reinterpret_cast<char *>(data_ptr.get()));
  for (int64_t j = 0; j < out_shape.num_elements(); j++) {
    int64_t offset = string_head[j].addr;
    int64_t string_len = string_head[j].len;
    const char* temp_string = reinterpret_cast<const char*>(data_ptr.get()) + offset;
    tensor_data[j] = tstring(temp_string, string_len);
    ADP_LOG(INFO) << "[GEOP] output string data " << tensor_data[j];
  }
  return Status::OK();
}

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
    std::transform(ge_output_dims.begin(), ge_output_dims.end(), std::back_inserter(dims),
                   [](const int64_t dim) { return dim; });
    TensorShape out_shape(dims);
    const DataType out_type = ctx->op_kernel().output_type(i);
    size_t output_size = output.GetSize();
    geDataUniquePtr data_ptr = std::move(output.ResetData());
    ADP_LOG(INFO) << "[GEOP] Get ge output: " << i << " tensor shape is: " << out_shape.DebugString()
                  << ", data placement is: " << data_placement << ", output_size is: " << output_size
                  << ", data addr is: " << reinterpret_cast<uintptr_t>(data_ptr.get());

    if (data_placement != ge::kPlacementDevice) {
      const static int64_t kTensorAlignBytes = 64;
      if (reinterpret_cast<uintptr_t>(data_ptr.get()) % kTensorAlignBytes == 0) {
        ADP_LOG(INFO) << "[GEOP] Zero copy ge tensor " << reinterpret_cast<uintptr_t>(data_ptr.get())
                      << " as aligned with " << kTensorAlignBytes << " bytes";

        if (out_type == DT_STRING) {  // string type op is not sink now
          Tensor cpu_tensor = Tensor(out_type, out_shape);
          if (BuildStringOutput(std::move(data_ptr), output_size, cpu_tensor) != Status::OK()) {
            return errors::Internal("The output string data analyze failed.");
          }
          ctx->set_output(i, cpu_tensor);
        } else {
          Allocator *allocator = NpuHostFixedAllocator::Create(std::move(data_ptr));
          Tensor cpu_tensor(allocator, out_type, out_shape);
          if (output_size != cpu_tensor.TotalBytes()) {
            LOG(ERROR) << "[GEOP] Graph engine process graph success but output " << i << " total bytes "
                       << output_size << " mismatched with expected " << cpu_tensor.TotalBytes();
            return errors::Internal("Graph engine process graph success but output length mismatched with expected.");
          }
          ctx->set_output(i, cpu_tensor);
        }
      } else {
        ADP_LOG(ERROR) << "[GEOP] Skip zero copy as ge tensor, " << reinterpret_cast<uintptr_t>(data_ptr.get())
                       << " not aligned with " << kTensorAlignBytes << " bytes";
        return errors::Internal("[GEOP] Skip zero copy ge tensor, bytes not aligned with expected.");
      }
    } else {
      ADP_LOG(INFO) << "[GEOP] GE output data placement is device, construct output info tensor.";
      auto getnext_output_info = std::unique_ptr<NpuGetNextOutputInfo>(
          new NpuGetNextOutputInfo(data_placement, ge_output_dims, output_size, std::move(data_ptr)));
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

bool CmpVecValue(const Node *const node1, const Node *const node2) {
  if (node1 == nullptr || node2 == nullptr) {
    ADP_LOG(ERROR) << "node1 or node2 is nullptr.";
    LOG(ERROR) << "node1 or node2 is nullptr.";
    return false;
  }
  return node1->name() < node2->name();
}

bool CmpNodeIndex(const std::pair<Node *, uint32_t> &p1, const std::pair<Node *, uint32_t> &p2) {
  return p1.second < p2.second;
}
}  // namespace

std::string CurrentTimeInStr() {
  std::time_t now = std::time(nullptr);
  std::tm *ptm = std::localtime(&now);
  if (ptm == nullptr) { return ""; }

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
const std::string kAllReduce = "HcomAllReduce";

GeOp::GeOp(OpKernelConstruction *ctx)
    : AsyncOpKernel(ctx), init_flag_(false), build_flag_(false), add_graph_flag_(false), sess_init_flag_(false),
      compute_graph_empty_(false), is_input_convert_(false), data_format_(""), graph_id_(0),
      is_initialized_graph_(false), need_iteration_(false), tf_session_(""), ge_session_(nullptr), job_type_(""),
      is_host_graph_(false), handle_(nullptr), need_compile_graph_first_(false), tuned_flag_(ATOMIC_FLAG_INIT),
      jit_compile_(false), session_id_(0), aoe_initialize_(nullptr), aoe_finalize_(nullptr),
      aoe_create_session_(nullptr), aoe_destroy_session_(nullptr), aoe_set_gesession_(nullptr),
      aoe_set_dependgraphs_(nullptr), aoe_set_tuninggraph_(nullptr), aoe_tuning_graph_(nullptr),
      aoe_set_depend_graphs_inputs_(nullptr), aoe_set_tuning_graph_input_(nullptr) {
  Initialize(ctx);
}

GeOp::~GeOp() {
  Finalize();
}

void GeOp::Initialize(OpKernelConstruction *ctx) {
  mutex_lock lock{mu_};
  int64 startTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "[GEOP] Begin GeOp initialize.";
  if (init_flag_) {
    ADP_LOG(WARNING) << "[GEOP] GEOP already Initialize.";
    return;
  }

  CHECK_NOT_NULL(ctx);
  const NameAttrList *func = nullptr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("function", &func));
  function_ = *func;
  std::string data_format;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
  ADP_LOG(INFO) << "Attr 'data_format' of " << ctx->def().name() << " is " << data_format;
  this->data_format_ = data_format;

  Status s = ctx->GetAttr("_session", &tf_session_);
  if (s.ok()) {
    ADP_LOG(INFO) << "[GEOP] get session info from attr, tf session: " << tf_session_;
  }

  ctx->GetAttr("_enable_graph_parallel", &enable_graph_parallel_);
  ctx->GetAttr("_graph_parallel_option_path", &graph_parallel_option_path_);
  ctx->GetAttr("_recompute_mode", &recompute_mode_);
  ctx->GetAttr("_dynamic_input", &dynamic_input_);
  if (!dynamic_input_.empty() && dynamic_input_ == "1") {
    jit_compile_ = true;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("_dynamic_graph_execute_mode", &dynamic_graph_execute_mode_));
    ctx->GetAttr("_getnext_inputs_shape_range", &getnext_inputs_shape_range_);
    ctx->GetAttr("_data_inputs_shape_range", &data_inputs_shape_range_);
    ctx->GetAttr("_is_dynamic_getnext", &is_dynamic_getnext_);
    ctx->GetAttr("_placeholder_index", &placeholder_index_);
  }
  ctx->GetAttr("_train_graph", &is_train_graph_);
  ctx->GetAttr("_is_var_init_graph", &is_var_init_graph_);
  ADP_LOG(INFO) << "[GEOP] dynamic_input: " << dynamic_input_
                << ", dynamic_graph_execute_mode: " << dynamic_graph_execute_mode_
                << ", jit_compile: " << jit_compile_
                << ", getnext_inputs_shape_range: " << getnext_inputs_shape_range_
                << ", data_inputs_shape_range: " << data_inputs_shape_range_ << ", is_train_graph: " << is_train_graph_
                << ", is_dynamic_getnext: " << is_dynamic_getnext_ << ", placeholder_index: " << placeholder_index_
                << ", is_var_init_graph: " << is_var_init_graph_;

  // global environment Initialize, invoke once for each process
  std::string sess_config = "";
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
  ADP_LOG(INFO) << "init options: ";
  NpuAttrs::LogOptions(init_options_);

  if ((!init_options_["ge.jobType"].empty()) && (!init_options_["ge.tuningPath"].empty())) {
    handle_ = mmDlopen("libaoe_tuning.so", MMPA_RTLD_NOW);
    OP_REQUIRES(ctx, handle_ != nullptr, errors::InvalidArgument("libaoe_tuning.so dlopen failed, ", mmDlerror()));
    // aoe init
    aoe_initialize_ = (AoeInitializeFunc) mmDlsym(handle_, "AoeInitialize");
    OP_REQUIRES(ctx, aoe_initialize_ != nullptr,
                errors::InvalidArgument("dlsym Aoe initialize API failed, ", mmDlerror()));
    // aoe finalize
    aoe_finalize_ = (AoeFinalizeFunc) mmDlsym(handle_, "AoeFinalize");
    OP_REQUIRES(ctx, aoe_initialize_ != nullptr,
                errors::InvalidArgument("dlsym Aoe Finalize API failed, ", mmDlerror()));
    // aoe create session
    aoe_create_session_ = (AoeCreateSessionFunc) mmDlsym(handle_, "AoeCreateSession");
    OP_REQUIRES(ctx, aoe_create_session_ != nullptr,
                errors::InvalidArgument("dlsym Aoe create session API failed, ", mmDlerror()));
    // aoe destroy session
    aoe_destroy_session_ = (AoeDestroySessionFunc) mmDlsym(handle_, "AoeDestroySession");
    OP_REQUIRES(ctx, aoe_destroy_session_ != nullptr,
                errors::InvalidArgument("dlsym Aoe destroy session API failed, ", mmDlerror()));
    // aoe set session
    aoe_set_gesession_ = (AoeSetGeSessionFunc) mmDlsym(handle_, "AoeSetGeSession");
    OP_REQUIRES(ctx, aoe_set_gesession_ != nullptr,
                errors::InvalidArgument("dlsym Aoe set session API failed, ", mmDlerror()));
    // aoe set depend graphs
    aoe_set_dependgraphs_ = (AoeSetDependGraphFunc) mmDlsym(handle_, "AoeSetDependGraphs");
    OP_REQUIRES(ctx, aoe_set_dependgraphs_ != nullptr,
                errors::InvalidArgument("dlsym Aoe set depend graphs API failed, ", mmDlerror()));
    // aoe set tuning graph
    aoe_set_tuninggraph_ = (AoeSetTuningGraphFunc) mmDlsym(handle_, "AoeSetTuningGraph");
    OP_REQUIRES(ctx, aoe_set_tuninggraph_ != nullptr,
                errors::InvalidArgument("dlsym Aoe aoe set tuning graph API failed, ", mmDlerror()));
    // aoe tuning
    aoe_tuning_graph_ = (AoeTuningGraphFunc) mmDlsym(handle_, "AoeTuningGraph");
    OP_REQUIRES(ctx, aoe_tuning_graph_ != nullptr,
                errors::InvalidArgument("dlsym Aoe tuning graph API failed, ", mmDlerror()));
    // aoe set tuning depend graphs inputs
    aoe_set_depend_graphs_inputs_ =
        reinterpret_cast<AoeSetDependGraphsInputsFunc>(mmDlsym(handle_, "AoeSetDependGraphsInputs"));
    OP_REQUIRES(ctx, aoe_set_depend_graphs_inputs_ != nullptr,
                errors::InvalidArgument("dlsym Aoe set tuning depend graphs inputs API failed, ", mmDlerror()));
    // aoe set tuning graph inputs
    aoe_set_tuning_graph_input_ =
        reinterpret_cast<AoeSetTuningGraphInputFunc>(mmDlsym(handle_, "AoeSetTuningGraphInput"));
    OP_REQUIRES(ctx, aoe_set_tuning_graph_input_ != nullptr,
                errors::InvalidArgument("dlsym Aoe set tuning graph inputs API failed, ", mmDlerror()));
  }

  sess_options_ = NpuAttrs::GetSessOptions(ctx);
  ADP_LOG(INFO) << "session options: ";
  NpuAttrs::LogOptions(sess_options_);

  input_shapes_vec_.resize(ctx->num_inputs() + 1, absl::nullopt);

  init_flag_ = true;
  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(EVENT) << "[GEOP] GeOp Initialize success, cost: "
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
        if (!ret) {
          ADP_LOG(ERROR) << "tf session " << tf_session_ << " sub graph id failed.";
          LOG(ERROR) << "tf session " << tf_session_ << " sub graph id failed.";
          return;
        }
        if (graph_id == kInvalidGraphId) {
          SessionManager::GetInstance().DestroyGeSession(tf_session_);
          ClearGraphIdCount();
        }
      }

      if (!SessionManager::GetInstance().IsGeSessionExist()) {
        if (!GePlugin::GetInstance()->IsGlobal()) {
          if (!init_options_["ge.jobType"].empty() && !init_options_["ge.tuningPath"].empty() &&
              aoe_finalize_ != nullptr && tuned_initialize_flag_) {
            Aoe::AoeStatus tune_ret = (*aoe_finalize_)();
            if (tune_ret != Aoe::AOE_SUCCESS) {
              ADP_LOG(ERROR) << "[GEOP] exec aoe finalize func failed.";
              LOG(ERROR) << "[GEOP] exec aoe finalize func failed.";
              return;
            }
          }
          tuned_initialize_flag_ = false;
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
          (void) mmDlclose(handle_);
        }
      }
    }
  }
  init_flag_ = false;
  ADP_LOG(INFO) << "[GEOP] GeOp Finalize success, tf session: " << tf_session_ << ", graph_id_: " << graph_id_;
  return;
}

int32_t GeOp::InitRebuildFlag(uint32_t cache_graph_id) {
  if (!build_flag_) {
    ADP_LOG(INFO) << "[GEOP] tf session " << tf_session_ << ", graph id: " << cache_graph_id
                  << " does not build yet, no need to check rebuild";
    return 0;
  }
  if (compute_graph_empty_) {
    ADP_LOG(INFO) << "[GEOP] tf session " << tf_session_ << ", graph id: " << cache_graph_id
                  << " is empty, no need to check rebuild";
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
               << error_message;
    return -1;
  }

  build_flag_ = false;
  compute_graph_empty_ = false;
  return 0;
}

bool GeOp::IncrementGraphIdCount(uint32_t &graph_id) {
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

bool GeOp::DecrementGraphIdCount(const std::string &tf_session, uint32_t &graph_id) {
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

void GeOp::ClearGraphIdCount() {
  auto it = session_and_graph_id_map_.find(tf_session_);
  if (it != session_and_graph_id_map_.end()) {
    session_and_graph_id_map_.erase(it);
  }
}

void GeOp::GetExecGraphId(uint32_t &cache_graph_id, std::vector<std::string> input_shapes) {
  size_t num = cache_graphs_.size();
  if (cache_graphs_.find(input_shapes) != cache_graphs_.end()) {
    auto iter = std::find_if(graph_counts_.begin(), graph_counts_.end(),
                             [&input_shapes](const std::pair<std::vector<std::string>, uint32_t> graph_count) {
                               return graph_count.first == input_shapes;
                             });
    if (iter != graph_counts_.end()) {
      iter->second += 1;
    }
    cache_graph_id = cache_graphs_[input_shapes];
    build_flag_ = true;
  } else {
    if (num >= kMaxCacheNum) {
      ADP_LOG(INFO) << "[GEOP] the cache vector size is : " << num << " , begin erase the least uesed";
      std::sort(graph_counts_.begin(), graph_counts_.end(), CmpValue);
      uint32_t erased_graph_id = cache_graphs_[graph_counts_[0].first];
      cache_graphs_.erase(graph_counts_[0].first);
      graph_counts_.erase(graph_counts_.cbegin());
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

PartialTensorShape GeOp::MakeCompatShape(const PartialTensorShape &a, const PartialTensorShape &b) {
  const static auto kUnknownRankShape = PartialTensorShape();
  if (a.dims() != b.dims()) {
    return kUnknownRankShape;
  }
  PartialTensorShape shape;
  static constexpr int64 kUnknownDim = -1;
  std::vector<int64> dims;
  for (int i = 0; i < a.dims(); i++) {
    dims.push_back(kUnknownDim);
  }
  auto status = PartialTensorShape::MakePartialShape(dims.data(), static_cast<int32_t>(dims.size()), &shape);
  return status.ok() ? shape : kUnknownRankShape;
}

bool GeOp::MaybeUpdateShape(OpKernelContext *const ctx) {
  bool updated = false;
  for (size_t i = 0UL; i < static_cast<size_t>(ctx->num_inputs()); i++) {
    auto &shape = input_shapes_vec_[i];
    auto &value_shape = ctx->input(static_cast<int32_t>(i)).shape();
    if (!shape.has_value()) {
      updated = true;
      shape = value_shape;
      ADP_LOG(INFO) << "Init input " << i << " shape " << shape.value().DebugString();
    } else {
      if (shape.value().IsCompatibleWith(value_shape)) {
        continue;
      } else {
        updated = true;
        ADP_LOG(INFO) << "Compat input " << i << " shape " << shape.value().DebugString() << " vs. "
                << value_shape.DebugString();
        shape = MakeCompatShape(shape.value(), value_shape);
        ADP_LOG(INFO) << "Refresh input " << i << " shape to " << shape.value().DebugString();
      }
    }
  }
  return updated;
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

      bool res = IncrementGraphIdCount(graph_id_);
      if (!res) {
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
        ADP_LOG(INFO) << "[GEOP] in tuning func, aoe_mode:" << init_options_["ge.jobType"]
                      << ", work_path:" << init_options_["ge.tuningPath"]
                      << ", distribute_config:" << init_options_["distribute_config"];
        tune_options_.insert(init_options_.cbegin(), init_options_.cend());
        tune_options_.insert({"devices", std::to_string(device_id)});
        tune_options_.insert(sess_options_.cbegin(), sess_options_.cend());
        tune_options_.insert({"work_path", init_options_["ge.tuningPath"]});
        tune_options_.insert({"job_type", init_options_["ge.jobType"]});
        // aoe ini
        if (!tuned_initialize_flag_) {
          std::map<Aoe::AscendString, Aoe::AscendString> global_options;
          global_options.insert(
              {Aoe::AscendString("work_path"), Aoe::AscendString(init_options_["ge.tuningPath"].c_str())});
          Aoe::AoeStatus init_ret = (*aoe_initialize_)(global_options);
          OP_REQUIRES_ASYNC(ctx, init_ret == Aoe::AOE_SUCCESS,
                            errors::Internal("[GEOP] exec aoe initialize func failed[", init_ret, "]."), done);
          tuned_initialize_flag_ = true;
        }
      }
      ADP_LOG(INFO) << "[GEOP] tf session: " << tf_session_ << " get ge session success.";
      sess_init_flag_ = true;
    }
  }

  // convert input to const
  OP_REQUIRES_OK_ASYNC(ctx, GraphInputConvertToConst(ctx), done);
  std::string geop_name = ctx->op_kernel().name();
  uint32_t num_inputs = static_cast<uint32_t>(ctx->num_inputs());
  ADP_LOG(INFO) << "[GEOP] Begin GeOp::ComputeAsync"
                << ", kernel_name:" << geop_name << ", num_inputs:" << num_inputs
                << ", num_outputs:" << ctx->num_outputs();
  int64 startTime = InferShapeUtil::GetCurrentTimestap();
  int64 endTime = 0;

  // To be compatible with old versions, we should check dynamic_input_
  if (dynamic_input_ != "1") {
    bool shape_changed = MaybeUpdateShape(ctx);
    if (build_flag_ && shape_changed) {
      ge::Status status = ge_session_->RemoveGraph(graph_id_);
      if (status != ge::SUCCESS) {
        ADP_LOG(WARNING) << "[GEOP] GE remove graph failed, ret : " << ToString(status) << ", graph_id: " << graph_id_;
      }
      build_flag_ = false;
    }
  }

  std::vector<Tensor> input_vec;
  std::vector<std::string> input_shapes;
  std::vector<ge::Tensor> inputs;
  OP_REQUIRES_OK_ASYNC(ctx, (BuildInputTensorInfo(ctx, input_vec, input_shapes, inputs)), done);

  // if input shapes changed, cache graphs
  uint32_t cache_graph_id = graph_id_;
  bool is_set_dynamic_config = IsDynamicConfig();
  bool is_tuning = (!init_options_["ge.jobType"].empty()) && (!init_options_["ge.tuningPath"].empty());
  bool is_lazy_recompile_mode = (dynamic_input_ == "1") && (dynamic_graph_execute_mode_ == "lazy_recompile");
  ADP_LOG(INFO) << "is_set_dynamic_config: " << is_set_dynamic_config << " is_tuning: " << is_tuning
                << " is_lazy_recompile_mode: " << is_lazy_recompile_mode;

  if (is_tuning) {
    if (is_set_dynamic_config) {
      ADP_LOG(ERROR) << "dynamic input config can not use with mstuning.";
      OP_REQUIRES_ASYNC(ctx, false, errors::Internal("dynamic input config can not use with mstuning."), done);
      return;
    }
    auto input_vec_aoe = input_vec;
    if (RunTuning(input_vec_aoe, inputs, ctx) != 0) {
      ADP_LOG(ERROR) << "RunTuning fail.";
      std::string error_message = ge::GEGetErrorMsg();
      std::stringstream ss;
      ss << std::endl << error_message;
      OP_REQUIRES_ASYNC(ctx, false, errors::Internal(ss.str()), done);
    }
    if (InitRebuildFlag(cache_graph_id) != 0) {
      OP_REQUIRES_ASYNC(ctx, false, errors::Internal("Failed to check rebuild flag"), done);
      return;
    }
    ADP_LOG(INFO) << geop_name << " RunTuning finish.";
  } else if (is_set_dynamic_config) {
    if (InitRebuildFlag(cache_graph_id) != 0) {
      OP_REQUIRES_ASYNC(ctx, false, errors::Internal("Failed to check rebuild flag"), done);
      return;
    }
  } else {
    // in dynamic input mode, cache graphs.
    if (is_lazy_recompile_mode) {
      GetExecGraphId(cache_graph_id, input_shapes);
    }
    if (InitRebuildFlag(cache_graph_id) != 0) {
      OP_REQUIRES_ASYNC(ctx, false, errors::Internal("Failed to check rebuild flag"), done);
      return;
    }
  }

  if (!build_flag_) {
    // Get Graph
    OP_REQUIRES_ASYNC(ctx, ctx->function_library() != nullptr, errors::Internal("function library is nullptr"), done);
    FunctionLibraryDefinition *flib_def =
        const_cast<FunctionLibraryDefinition *>(ctx->function_library()->GetFunctionLibraryDefinition());
    OP_REQUIRES_ASYNC(ctx, flib_def != nullptr, errors::Internal("flib_def is nullptr"), done);
    std::shared_ptr<Graph> graph = std::make_shared<Graph>(OpRegistry::Global());
    OP_REQUIRES_ASYNC(ctx, graph != nullptr, errors::Internal("create tensorflow graph failed"), done);

    // Build GraphDef from FunctionDef
    GraphDef ori_graph_def;
    bool is_allreduce = false;
    OP_REQUIRES_OK_ASYNC(ctx, BuildGraphDef(*flib_def, input_vec, ori_graph_def, is_initialized_graph_, is_allreduce),
                         done);

    /* if graph is init verify graph, return */
    if (this->is_initialized_graph_) {
      Tensor initialized_tensor(ctx->expected_output_dtype(0), TensorShape({0}));
      ctx->set_output(0, initialized_tensor);
      done();
      return;
    }
    if (kDumpGraph) {
      const std::string pbtxt_path = GetDumpPath() + "TF_" + geop_name.c_str() + ".pbtxt";
      (void) WriteTextProto(Env::Default(), pbtxt_path, ori_graph_def);
    }
    endTime = InferShapeUtil::GetCurrentTimestap();
    ADP_LOG(EVENT) << "[GEOP] In GEOP computeAsync, kernel_name: " << geop_name << " ,TFadapter cost time: ["
                   << ((endTime - startTime) / kMicrosToMillis) << " ms]";
    ADP_LOG(INFO) << "[GEOP] TFadpter process graph success, GE parser begin, kernel_name: " << geop_name
                  << " , tf session: " << tf_session_ << " , graph id: " << cache_graph_id;
    // parser,  tensorflow graph to ge graph
    std::shared_ptr<domi::ModelParser> model_parser =
        domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
    OP_REQUIRES_ASYNC(ctx, model_parser != nullptr, errors::Unavailable("create model parser ret failed."), done);
    ge::ComputeGraphPtr compute_graph = nullptr;
    try {
      compute_graph = std::make_shared<ge::ComputeGraph>("ge_default_" + CurrentTimeInStr());
    } catch (...) {
      OP_REQUIRES_ASYNC(ctx, false, errors::Internal("make shared failed"), done);
    }

    OP_REQUIRES_ASYNC(ctx, compute_graph != nullptr, errors::InvalidArgument("create ComputeGraph failed"), done);

    auto build_sub_graph = [this, flib_def](const std::string &graph) -> std::string {
      return this->BuildSubGraph(flib_def, graph);
    };
    ge::Status status =
        model_parser->ParseProtoWithSubgraph(ori_graph_def.SerializeAsString(), build_sub_graph, compute_graph);
    if (status != ge::SUCCESS) {
      std::string error_message = ge::GEGetErrorMsg();
      std::stringstream ss;
      ss << "graph parse failed. ret : " << status << std::endl
         << "Error Message is : " << std::endl
         << error_message;
      OP_REQUIRES_ASYNC(ctx, status == ge::SUCCESS, errors::Internal(ss.str()), done);
    }

    domi::GetContext().format = ge::GetParserContext().format;

    ADP_LOG(INFO) << "[GEOP] Tensorflow graph parse to ge graph success, kernel_name: " << geop_name
                  << " , tf session: " << tf_session_ << " , graph id: " << cache_graph_id
                  << ", iteration_per_loop: " << iteration_per_loop_ << ", need iteration: " << this->need_iteration_;

    size_t nodes = compute_graph->GetAllNodesSize();
    if (nodes == 0) {
      build_flag_ = true;
      compute_graph_empty_ = true;
      endTime = InferShapeUtil::GetCurrentTimestap();
      ADP_LOG(INFO) << "[GEOP] End GeOp::ComputeAsync, compute_graph is empty, kernel_name:" << geop_name
                    << ", ret_status:" << ToString(ge::SUCCESS) << " , tf session: " << tf_session_
                    << " ,graph id: " << cache_graph_id << " [" << ((endTime - startTime) / kMicrosToMillis) << " ms]";
      done();
      return;
    }

    // convert to ge::graph
    if (graph_options_.count("input_format") != 0) {
      ADP_LOG(INFO) << "graph_options_[\"input_format\"] = " << graph_options_["input_format"];
    } else {
      graph_options_["input_format"] = "NHWC";
      ADP_LOG(INFO) << "tensorflow_parser graph_options_[\"input_format\"] = " << graph_options_["input_format"];
    }
    ge::Graph ge_graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    if (iteration_per_loop_ > 1) {
      ge_graph.SetNeedIteration(this->need_iteration_);
    }

    if (is_host_graph_) {
      ADP_LOG(INFO) << "[GEOP] set graph option.";
      graph_options_["ge.exec.placement"] = "HOST";
    }
    graph_options_["ge.shape_generalized_build_mode"] = "shape_precise";
    if (!recompute_mode_.empty()) {
      graph_options_["ge.recompute"] = recompute_mode_;
    }
    if (!graph_parallel_option_path_.empty()) {
      graph_options_["ge.graphParallelOptionPath"] = graph_parallel_option_path_;
    }
    if (!enable_graph_parallel_.empty()) {
      graph_options_["ge.enableGraphParallel"] = enable_graph_parallel_;
    }
    SetDynamicInput();
    graph_options_["ge.exec.isVarInitGraph"] = is_var_init_graph_;
    graph_options_["ge.jit_compile"] = jit_compile_ ? "1" : "0";

    // call ge session addGraph api
    auto graph_options = graph_options_;
    if (is_tuning) {
      graph_options["ge.buildMode"] = "normal";
    }
    ADP_LOG(EVENT) << "[GEOP] call ge session add graph jit_compile: " << graph_options["ge.jit_compile"];
    status = ge_session_->AddGraph(cache_graph_id, ge_graph, graph_options);
    if (status != ge::SUCCESS) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
      ADP_LOG(FATAL) << "[GEOP] call ge session add graph failed, kernel: " << geop_name << " ,tf session: "
                     << tf_session_ << ", graph id: " << cache_graph_id;

      std::string error_message = ge::GEGetErrorMsg();
      std::stringstream ss;
      ss << "[GEOP] call ge session add graph failed, kernel: " << geop_name << ", tf session: " << tf_session_
         << ", graph id: " << cache_graph_id << std::endl
         << "Error Message is : " << std::endl
         << error_message;
      OP_REQUIRES_ASYNC(ctx, status == ge::SUCCESS, errors::Unavailable(ss.str()), done);
    } else {
      add_graph_flag_ = true;
      ADP_LOG(INFO) << "[GEOP] Add graph to ge session success, kernel_name: " << geop_name
                    << " , tf session: " << tf_session_ << " , graph id: " << cache_graph_id;
    }
    build_flag_ = true;
    if (!is_set_dynamic_config && is_lazy_recompile_mode) {
      cache_graphs_.insert(std::make_pair(input_shapes, cache_graph_id));
      graph_counts_.push_back(std::make_pair(input_shapes, 1));
    }
    if (need_compile_graph_first_) {
      ge::Status build_graph_status = ge_session_->BuildGraph(cache_graph_id, inputs);
      if (build_graph_status != ge::SUCCESS) {
        std::string error_message = ge::GEGetErrorMsg();
        std::stringstream ss;
        ss << "[GEOP] GE session build graph failed, domi_ret : " << build_graph_status << std::endl
           << "Error Message is : " << std::endl
           << error_message;
        OP_REQUIRES_ASYNC(ctx, build_graph_status == ge::SUCCESS, errors::Unavailable(ss.str()), done);
      }

      ADP_LOG(INFO) << "[GEOP] Build graph success.";
      done();
      return;
    }
    LOG(INFO) << "The model has been compiled on the Ascend AI processor, current graph id is: " << cache_graph_id;
  } else {
    if (compute_graph_empty_) {
      endTime = InferShapeUtil::GetCurrentTimestap();
      ADP_LOG(INFO) << "[GEOP] End GeOp::ComputeAsync, compute_graph is empty, kernel_name:" << geop_name
                    << ", ret_status:" << ToString(ge::SUCCESS) << " , tf session: " << tf_session_
                    << " ,graph id: " << cache_graph_id << " [" << ((endTime - startTime) / kMicrosToMillis) << " ms]";
      done();
      return;
    }
  }

  int64 run_start_time = InferShapeUtil::GetCurrentTimestap();
  auto callback = [done, ctx, run_start_time](ge::Status ge_status, std::vector<ge::Tensor> &outputs) {
    if (ge_status == ge::SUCCESS) {
      if (BuildOutputTensorInfo(ctx, outputs) != Status::OK()) {
        ADP_LOG(FATAL) << ctx->op_kernel().name() << " GEOP::DoRunAsync get output failed.";
        std::string error_message = ge::GEGetErrorMsg();
        std::stringstream ss;
        ss << ctx->op_kernel().name() << "GEOP::DoRunAsync get output failed." << std::endl
           << "Error Message is : " << std::endl
           << error_message;
        OP_REQUIRES_ASYNC(ctx, false, errors::Internal(ss.str()), done);
        return;
      }
    } else if (ge_status == ge::END_OF_SEQUENCE) {
      ctx->SetStatus(errors::OutOfRange("End of sequence"));
      ADP_LOG(WARNING) << "[GEOP] Out of range: End of sequence.";
      LOG(WARNING) << "[GEOP] Out of range: End of sequence.";
    } else if (ge_status != ge::SUCCESS) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
      ADP_LOG(FATAL) << ctx->op_kernel().name() << "GEOP::::DoRunAsync Failed";
      std::string error_message = ge::GEGetErrorMsg();
      std::stringstream ss;
      ss << ctx->op_kernel().name() << "GEOP::::DoRunAsync Failed" << std::endl
         << "Error Message is : " << std::endl << error_message;
      OP_REQUIRES_ASYNC(ctx, false, errors::Internal(ss.str()), done);
      return;
    }
    int64 run_end_time = InferShapeUtil::GetCurrentTimestap();
    ADP_LOG(EVENT) << "[GEOP] RunGraphAsync callback, status:" << ge_status
                   << ", kernel_name:" << ctx->op_kernel().name() << "[ " << (run_end_time - run_start_time) << "us]";
    done();
  };

  // call ge session runGraphAsync api
  ADP_LOG(INFO) << "[GEOP] Call ge session RunGraphAsync, kernel_name:" << geop_name << " ,tf session: " << tf_session_
                << " ,graph id: " << cache_graph_id;
  ge::Status run_graph_status = ge_session_->RunGraphAsync(cache_graph_id, inputs, callback);
  if (run_graph_status != ge::SUCCESS) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
    ADP_LOG(FATAL) << "[GEOP] call ge session RunGraphAsync Failed, kernel:" << geop_name << " ,tf session: "
                   << tf_session_ << " ,graph id: " << cache_graph_id;
    std::string error_message = ge::GEGetErrorMsg();
    std::stringstream ss;
    ss << "[GEOP] call ge session RunGraphAsync Failed, kernel:" << geop_name << ", tf session: " << tf_session_
       << ", graph id: " << cache_graph_id << std::endl
       << "Error Message is : " << std::endl
       << error_message;
    OP_REQUIRES_ASYNC(ctx, run_graph_status == ge::SUCCESS, errors::Unavailable(ss.str()), done);
  }

  endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "[GEOP] End GeOp::ComputeAsync, kernel_name:" << geop_name
                << ", ret_status:" << ToString(run_graph_status) << " ,tf session: " << tf_session_
                << " ,graph id: " << cache_graph_id << " [" << ((endTime - startTime) / kMicrosToMillis) << " ms]";
  return;
}

void GeOp::ChangeChannelNameAttr(NodeDef &node_def) const {
  const std::string pre_channel_name = node_def.attr().at("channel_name").s();
  uint32_t device_id = 0;
  (void) GetEnvDeviceID(device_id);
  AttrValue channel_name = AttrValue();
  channel_name.set_s(std::to_string(
      std::hash<std::string>{}(tf_session_ + pre_channel_name + "_device_" + std::to_string(device_id))));
  (*node_def.mutable_attr())["channel_name"] = channel_name;
  ADP_LOG(INFO) << "[GEOP] changed the value of channel_name attr of node:" << node_def.name() << " to "
                << channel_name.s();
}

void GeOp::ProcessDpOpFuncDef(const Node &node) const {
  const std::string func_name = node.def().attr().at("function").func().name();
  const std::string org_func_def_lib = node.def().attr().at("func_def").s();
  FunctionDefLibrary func_def_lib;
  func_def_lib.ParseFromString(org_func_def_lib);
  bool is_new_transfer_mode = NpuAttrs::GetNewDataTransferFlag();
  for (auto &func_def : *func_def_lib.mutable_function()) {
    if (func_def.signature().name() == func_name) {
      for (auto &node_def : *func_def.mutable_node_def()) {
        if (!NpuAttrs::IsDatasetExecuteInDevice(tf_session_ + node_def.name()) &&
            (node_def.op() == "IteratorV2" || node_def.op() == "Iterator")) {
          NpuAttrs::SetDatasetExecuteInDeviceStatus(tf_session_ + node_def.name(), true);
        }
        if (node_def.op() == "DeviceQueueDataset") {
          if (is_new_transfer_mode) {
            ChangeChannelNameAttr(node_def);
          }
          tensorflow::AttrValue value;
          value.set_b(is_new_transfer_mode);
          node_def.mutable_attr()->insert({"_is_new_data_transfer", value});
        }
      }
    }
  }
  std::string new_func_def_lib;
  func_def_lib.SerializeToString(&new_func_def_lib);
  AttrValue func_def_value = AttrValue();
  func_def_value.set_s(new_func_def_lib);
  NodeDef &node_def = const_cast<NodeDef &>(node.def());
  (*node_def.mutable_attr())["func_def"] = func_def_value;
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
  if (node_def.op() == "Where") {
    is_initialize = InferShapeUtil::IsInitializedGraph(node);
  }
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

void GeOp::BuildQueueDataAndGetNextFromQueue(Graph &graph, const Node &getnext_node,
                                             const std::string &channel_name) const {
  Node *get_next_from_queue = nullptr;
  Node *queue_data = nullptr;
  std::string get_next_from_queue_name = "get_next_from_queue_" + getnext_node.name();
  std::string queue_data_name = "queue_data_" + getnext_node.name();
  auto get_next_attrs = getnext_node.def().attr();
  TF_CHECK_OK(NodeBuilder(queue_data_name, "QueueData")
                  .Device(getnext_node.def().device())
                  .Attr("index", 0)
                  .Attr("T", DT_UINT8)
                  .Attr("queue_name", channel_name)
                  .Attr("output_types", get_next_attrs["output_types"])
                  .Attr("output_shapes", get_next_attrs["output_shapes"])
                  .Finalize(&graph, &queue_data));

  TF_CHECK_OK(NodeBuilder(get_next_from_queue_name, "GetNextFromQueue")
                  .Input(NodeBuilder::NodeOut(queue_data, 0))
                  .Device(getnext_node.def().device())
                  .Attr("output_types", get_next_attrs["output_types"])
                  .Attr("output_shapes", get_next_attrs["output_shapes"])
                  .Finalize(&graph, &get_next_from_queue));

  for (auto out_edge : getnext_node.out_edges()) {
    CHECK_NOT_NULL(out_edge);
    graph.AddEdge(get_next_from_queue, out_edge->src_output(), out_edge->dst(), out_edge->dst_input());
  }

  const OpDef &queue_data_op_def = queue_data->op_def();
  NodeDef &queue_data_node_def = const_cast<NodeDef &>(queue_data->def());
  std::string queue_data_op_def_string;
  queue_data_op_def.SerializeToString(&queue_data_op_def_string);
  tensorflow::AttrValue queue_data_attr;
  queue_data_attr.set_s(queue_data_op_def_string);
  queue_data_node_def.mutable_attr()->insert({"op_def", queue_data_attr});

  const OpDef &get_next_op_def = get_next_from_queue->op_def();
  NodeDef &get_next_node_def = const_cast<NodeDef &>(get_next_from_queue->def());
  std::string get_next_op_def_string;
  get_next_op_def.SerializeToString(&get_next_op_def_string);
  tensorflow::AttrValue get_next_attr;
  get_next_attr.set_s(get_next_op_def_string);
  get_next_node_def.mutable_attr()->insert({"op_def", get_next_attr});
}

void GeOp::HandleDpOpAndGetNextNodes(Graph &graph) {
  std::vector<Node *> remove_nodes;
  for (Node *node : graph.nodes()) {
    CHECK_NOT_NULL(node);
    if (node->type_string() == "DPOP") {
      ProcessDpOpFuncDef(*node);
    } else if (node->type_string() == "IteratorGetNext") {
      Node *iterator_node = nullptr;
      std::string iterator_name;
      NodeDef &node_def = const_cast<NodeDef &>(node->def());
      for (auto in_edge : node->in_edges()) {
        CHECK_NOT_NULL(in_edge);
        CHECK_NOT_NULL(in_edge->src());
        if (in_edge->src()->type_string() == "IteratorV2" || in_edge->src()->type_string() == "Iterator") {
          iterator_name = in_edge->src()->name();
          iterator_node = in_edge->src();
        }
      }
      uint32_t device_id = 0;
      (void) GetEnvDeviceID(device_id);
      std::string channel_name = std::to_string(
          std::hash<std::string>{}(tf_session_ + iterator_name + "_device_" + std::to_string(device_id)));
      if (kIsHeterogeneous) {
        BuildQueueDataAndGetNextFromQueue(graph, *node, channel_name);
        remove_nodes.push_back(node);
        remove_nodes.push_back(iterator_node);
      } else if (NpuAttrs::IsDatasetExecuteInDevice(tf_session_ + iterator_name)) {
        if (dynamic_input_ == "1") {
          node_def.set_op("DynamicGetNext");
        }
      } else {
        Node *aicpu_getnext = nullptr;
        std::string aicpu_getnext_name = "aicpu_getnext_" + node->name();
        auto getnext_attrs = node->def().attr();
        std::string aicpu_getnext_type = dynamic_input_ == "1" ? "DynamicGetNextV2" : "GetNext";
        TF_CHECK_OK(NodeBuilder(aicpu_getnext_name, aicpu_getnext_type)
                        .Device(node->def().device())
                        .Attr("channel_name", channel_name)
                        .Attr("output_types", getnext_attrs["output_types"])
                        .Attr("output_shapes", getnext_attrs["output_shapes"])
                        .Finalize(&graph, &aicpu_getnext));
        for (auto out_edge : node->out_edges()) {
          CHECK_NOT_NULL(out_edge);
          graph.AddEdge(aicpu_getnext, out_edge->src_output(), out_edge->dst(), out_edge->dst_input());
        }
        const OpDef &getnext_op_def = aicpu_getnext->op_def();
        NodeDef &getnext_node_def = const_cast<NodeDef &>(aicpu_getnext->def());
        std::string op_def_s;
        getnext_op_def.SerializeToString(&op_def_s);
        tensorflow::AttrValue value;
        value.set_s(op_def_s);
        getnext_node_def.mutable_attr()->insert({"op_def", value});
        remove_nodes.push_back(node);
        remove_nodes.push_back(iterator_node);
      }
      if (dynamic_input_ == "1" && dynamic_graph_execute_mode_ == "lazy_recompile") {
        graph_options_["ge.exec.enableCopyOutputAddr"] = "1";
      }
    }
  }
  for (Node *node : remove_nodes) {
    ADP_LOG(INFO) << "[GEOP] Remove node: " << node->name();
    graph.RemoveNode(node);
  }
}

void GeOp::UpdateInputsShapeDesc(Graph &graph) {
  for (auto node : graph.op_nodes()) {
    if (!node->IsArg()) {
      continue;
    }
    size_t index = static_cast<size_t>(node->attrs().Find("index")->i());
    node->ClearAttr("_output_shapes");
    node->AddAttr("_output_shapes", std::vector<PartialTensorShape>{input_shapes_vec_[index].value()});

    NodeDef &node_def = const_cast<NodeDef &>(node->def());
    AttrValue &output_tensor_descs = (*node_def.mutable_attr())[OUTPUT_DESC];
    auto &shape = input_shapes_vec_[index].value();
    AttrValue attr_shape_value;
    attr_shape_value.set_type(DT_INT32);
    for (auto i = 0; i < shape.dims(); ++i) {
      attr_shape_value.mutable_list()->add_i(shape.dim_size(i));
    }
    (*output_tensor_descs.mutable_list()->mutable_func(0)->mutable_attr())[SERIALIZE_SHAPE] = attr_shape_value;
  }
}

// Build GraphDef from FunctionDef.
Status GeOp::BuildGraphDef(FunctionLibraryDefinition &flib_def, const std::vector<Tensor> &input_vec,
                           GraphDef &graph_def, bool &is_initialize, bool &is_allreduce) {
  const FunctionDef *function_def = flib_def.Find(function_.name());
  NPU_REQUIRES(function_def != nullptr, errors::Internal("Function:", function_.name(), " fdef is nullptr"));
  // get infershape
  Graph graph(OpRegistry::Global());
  Status ret = InferShapeUtil::InferShape(input_vec, &flib_def, function_def, &graph);
  if (!ret.ok()) {
    ADP_LOG(ERROR) << "[GEOP] InferShape failed, " << ret.error_message();
    LOG(ERROR) << "[GEOP] InferShape failed, " << ret.error_message();
    return ret;
  }

  bool is_set_dynamic_config = IsDynamicConfig();
  if (is_set_dynamic_config) {
    jit_compile_ = true;
    BuildShapeNodeAndCacheArgNodes(graph);
  }

  for (Node *node : graph.nodes()) {
    if (node->type_string() == kAllReduce) {
      is_allreduce = true;
    }
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

    if (node->type_string() == "NpuOnnxGraphOp") {
      ret = this->ParseOnnxGraphOpAttr(node);
      graph_options_["input_format"] = "NCHW";
      ADP_LOG(INFO) << "onnx_graph_parser graph_options_[\"input_format\"] = " << graph_options_["input_format"];
      if (!ret.ok()) {
        LOG(ERROR) << "[GEOP]node: " << node->name() << " Parse Node with Onnx Model failed, " << ret.error_message();
        return ret;
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
  HandleDpOpAndGetNextNodes(graph);

  // 二进制场景,更新输入shape
  if (!jit_compile_) {
    UpdateInputsShapeDesc(graph);
  }

  graph.ToGraphDef(&graph_def);
  std::string enable_force_v2_control;
  (void) ReadStringFromEnvVar("ENABLE_FORCE_V2_CONTROL", "", &enable_force_v2_control);
  if (enable_force_v2_control == "1") {
    Status status = FunctionalizeControlFlow(&graph, &flib_def);
    if (status != Status::OK()) {
      LOG(WARNING) << "[GEOP] Failed functionalize control flow: " << status.error_message();
      return Status::OK();
    }
    graph.ToGraphDef(&graph_def);
  }
  return Status::OK();
}

Status GeOp::ParseOnnxGraphOpAttr(Node *&node) const {
  NodeDef &node_def = const_cast<NodeDef &>(node->def());

  // Get input and output numbers of NpuOnnxGraphOp op
  AttrValue in_value;
  int32_t inout_nums = node->num_inputs();
  in_value.set_i(static_cast<int32_t>(inout_nums));
  node_def.mutable_attr()->insert({"_input_num", in_value});
  inout_nums = node->num_outputs();
  AttrValue ot_value;
  ot_value.set_i(static_cast<int32_t>(inout_nums));
  node_def.mutable_attr()->insert({"_output_num", ot_value});

  std::string model_path = node_def.attr().find("model_path")->second.s();
  ge::Graph sub_graph("onnx_compute_graph_" + node->name());
  std::map<ge::AscendString, ge::AscendString> parser_params;
  std::string subgrph_name("onnx_compute_graph_" + node->name() + '_' + CurrentTimeInStr());
  parser_params.insert({ge::AscendString(ge::ir_option::OUTPUT), ge::AscendString(subgrph_name.c_str())});
  NPU_REQUIRES(ge::aclgrphParseONNX(model_path.c_str(), parser_params, sub_graph) == ge::SUCCESS,
               errors::Internal("[GEOP] node:", node->name(), " Onnx Model Parse Failed."));

  // rename the nodes in subgraph of onnx model
  for (auto &sub_node : sub_graph.GetAllNodes()) {
    auto snode = ge::NodeAdapter::GNode2Node(sub_node);
    auto orig_name = snode->GetName();
    auto modi_name = node->name() + '_' + orig_name;
    snode->GetOpDesc()->SetName(modi_name);
  }

  ge::Model onnx_model("onnx_compute_model_" + node->name(), "");
  onnx_model.SetGraph(sub_graph);
  ge::Buffer model_buf;
  NPU_REQUIRES(onnx_model.Save(model_buf, false) == ge::SUCCESS,
               errors::Internal("[GEOP] node:", node->name(), " Onnx Model Serialized Failed."));

  std::string model_str(reinterpret_cast<const char *>(model_buf.GetData()), model_buf.GetSize());
  AttrValue attr_value;
  attr_value.set_s(model_str);
  node_def.mutable_attr()->insert({"_external_model", attr_value});
  return Status::OK();
}

void GeOp::BuildShapeNodeAndCacheArgNodes(Graph &graph) {
  if (kIsHeterogeneous) {
    ADP_LOG(INFO) << "Is heterogeneous, no need to build shape node and cache arg nodes.";
    return;
  }
  std::string dynamic_node_type = sess_options_["ge.dynamicNodeType"];
  for (Node *node : graph.nodes()) {
    // add shape node to get getnext node real shape
    if (dynamic_node_type == "0" && node->type_string() == "IteratorGetNext") {
      dynamic_shape_nodes_.emplace_back(node);
      ADP_LOG(INFO) << "push in dynamic shape nodes, node: " << node->name() << ", type:" << node->type_string();
      std::set<int32_t> out_index;
      for (auto out_edge : node->out_edges()) {
        if (!out_edge->IsControlEdge()) {
          std::string msg = "Src:" + out_edge->src()->name() + ":" + std::to_string(out_edge->src_output()) +
              ", Dst:" + out_edge->dst()->name() + ":" + std::to_string(out_edge->dst_input());
          ADP_LOG(INFO) << "[GEOP] GetNext node in out info:" << msg;
          out_index.insert(out_edge->src_output());
        }
      }
      for (int32_t idx : out_index) {
        std::string shape_name = "getnext_shape_" + std::to_string(idx);
        Node *shape_node = nullptr;
        TF_CHECK_OK(NodeBuilder(shape_name, "Shape")
                        .Input(node, idx)
                        .Device(node->def().device())
                        .Finalize(&graph, &shape_node));
        std::string identity_name = "shape_identity_" + std::to_string(idx);
        Node *identity_node = nullptr;
        TF_CHECK_OK(NodeBuilder(identity_name, "Identity")
                        .Input(shape_node, 0)
                        .Device(shape_node->def().device())
                        .Finalize(&graph, &identity_node));
      }
    }
    // count data args and getnext args for dynamic dims
    if (node->type_string() == "_Arg") {
      if (node->name().find("IteratorGetNext_") != std::string::npos) {
        if (dynamic_node_type == "0") {
          dynamic_shape_nodes_.emplace_back(node);
          ADP_LOG(INFO) << "push in dynamic shape nodes, node: " <<  node->name() << ", type:" << node->type_string();
        }
      } else {
        if (dynamic_node_type == "1") {
          dynamic_shape_nodes_.emplace_back(node);
          ADP_LOG(INFO) << "push in dynamic shape nodes, node: " << node->name() << ", type:" << node->type_string();
        }
      }
    }
  }
  // sort dynamic nodes to match input_shapes
  std::sort(dynamic_shape_nodes_.begin(), dynamic_shape_nodes_.end(), CmpVecValue);
}

Status GeOp::ChangeInputsShapeDesc() {
  if (kIsHeterogeneous) {
    ADP_LOG(INFO) << "Is heterogeneous, no need to change inputs shape desc.";
    return Status::OK();
  }
  std::vector<std::string> result;
  std::string input_shapes = sess_options_["ge.inputShape"];
  Split(input_shapes, result, ";");  // e.g. result:["data:2,3", "data1:3,4"]

  if (dynamic_shape_nodes_.size() == 1U && dynamic_shape_nodes_[0]->type_string() == "IteratorGetNext") {
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
    for (size_t i = 0U; i < dynamic_shape_nodes_.size(); ++i) {
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

void GeOp::SetShapesToOutputDesc(const std::vector<std::string> &input_shapes, const int &index,
                                 AttrValue &attr_shape_value) const {
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
  Split(input_shapes[index], shape, ":");  // e.g. shape:["data", "2,3,4"]
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
  Split(shape[1], dims, ",");  // e.g. dims:["2", "3", "4"]
  for (auto dim : dims) {
    attr_shape_value.mutable_list()->add_i(std::atoi(dim.c_str()));
  }
}

int GeOp::RunTuning(std::vector<Tensor> &input_vec, std::vector<ge::Tensor> &inputs, const OpKernelContext *const ctx) {
  if (tuned_flag_.test_and_set()) {
    ADP_LOG(INFO) << ctx->op_kernel().name() << " has tuned.";
    return 0;
  }
  ADP_LOG(INFO) << "[GEOP] " << ctx->op_kernel().name() << " begin tune.";

  // Get Graph
  if (ctx->function_library() == nullptr) {
    ADP_LOG(ERROR) << "function library is nullptr";
    return -1;
  }
  FunctionLibraryDefinition *flib_def =
      const_cast<FunctionLibraryDefinition *>(ctx->function_library()->GetFunctionLibraryDefinition());
  if (flib_def == nullptr) {
    ADP_LOG(ERROR) << "flib_def is nullptr";
    return -1;
  }
  std::shared_ptr<Graph> graph = std::make_shared<Graph>(OpRegistry::Global());
  if (graph == nullptr) {
    ADP_LOG(ERROR) << "create tensorflow graph failed";
    return -1;
  }

  // Build GraphDef from FunctionDef
  bool is_allreduce = false;
  GraphDef ori_graph_def;
  Status s = BuildGraphDef(*flib_def, input_vec, ori_graph_def, is_initialized_graph_, is_allreduce);
  if (!s.ok()) {
    ADP_LOG(ERROR) << "BuildGraphDef error";
    return -1;
  }

  if (is_initialized_graph_) {
    ADP_LOG(INFO) << ctx->op_kernel().name() << " graph is initialized";
    return 0;
  }

  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + "TF_" + ctx->op_kernel().name() + "_AOE.pbtxt";
    (void) WriteTextProto(Env::Default(), pbtxt_path, ori_graph_def);
  }

  // parser,  tensorflow graph to ge graph
  std::shared_ptr<domi::ModelParser> model_parser =
      domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  if (model_parser == nullptr) {
    ADP_LOG(ERROR) << "create model parser ret failed.";
    return -1;
  }
  ge::ComputeGraphPtr compute_graph = nullptr;
  compute_graph = std::make_shared<ge::ComputeGraph>("ge_default_" + CurrentTimeInStr());
  if (compute_graph == nullptr) {
    ADP_LOG(ERROR) << "create ComputeGraph failed";
    return -1;
  }

  auto build_sub_graph = [this, flib_def](const std::string &graph) -> std::string {
    return this->BuildSubGraph(flib_def, graph);
  };
  ge::Status status =
      model_parser->ParseProtoWithSubgraph(ori_graph_def.SerializeAsString(), build_sub_graph, compute_graph);
  if (status != ge::SUCCESS) {
    std::stringstream ss;
    ss << "graph parse failed. ret : " << status << std::endl
       << "Error Message is : " << std::endl
       << ge::GEGetErrorMsg();
    ADP_LOG(ERROR) << ss.str();
    return -1;
  }
  domi::GetContext().format = ge::GetParserContext().format;
  ADP_LOG(INFO) << "[GEOP] Tensorflow graph parse to ge graph success.";

  // convert to ge::graph
  ge::Graph ge_graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  if (iteration_per_loop_ > 1) {
    ge_graph.SetNeedIteration(this->need_iteration_);
  }
  if (is_host_graph_) {
    graph_options_["ge.exec.placement"] = "HOST";
  }
  SetDynamicInput();

  // run aoe tuning
  bool is_mdat_tuning = (init_options_["ge.jobType"] == kMdatTuning) && (recompute_mode_ == kAutoRecompute);
  if ((init_options_["ge.jobType"] == "1") || (init_options_["ge.jobType"] == "2") ||
      ((init_options_["ge.jobType"] == "4") && is_allreduce) || is_mdat_tuning) {
    std::function<void()> callback = [this]() {
      if (aoe_destroy_session_ != nullptr) {
        Aoe::AoeStatus aoe_destroy_ret = (*aoe_destroy_session_)(session_id_);
        if (aoe_destroy_ret != Aoe::AOE_SUCCESS) {
          ADP_LOG(ERROR) << "exec aoe destroy func failed[" << aoe_destroy_ret << "].";
          return;
        }
        ADP_LOG(INFO) << "[GEOP] aoe destroy success[" << aoe_destroy_ret << "].";
      }
    };
    ADP_LOG(INFO) << "[GEOP] in tune mode, training graph handled by tools.";

    // aoe create session
    std::map<Aoe::AscendString, Aoe::AscendString> session_options;
    session_options.insert({Aoe::AscendString("job_type"), Aoe::AscendString(init_options_["ge.jobType"].c_str())});
    session_options.insert({Aoe::AscendString("ge.resourceConfigPath"),
                            Aoe::AscendString(sess_options_["ge.resourceConfigPath"].c_str())});
    Aoe::AoeStatus session_ret = (*aoe_create_session_)(session_options, session_id_);
    if (session_ret != Aoe::AOE_SUCCESS) {
      ADP_LOG(ERROR) << "exec aoe create session func failed[" << session_ret << "].";
      return -1;
    }
    {
      GE_MAKE_GUARD(destroy, callback);
      // aoe set ge session
      Aoe::AoeStatus set_ret = (*aoe_set_gesession_)(session_id_, ge_session_);
      if (set_ret != Aoe::AOE_SUCCESS) {
        ADP_LOG(ERROR) << "exec aoe set session func failed[" << set_ret << "].";
        return -1;
      }
      // set tuning graph
      Aoe::AoeStatus tune_ret = (*aoe_set_tuninggraph_)(session_id_, ge_graph);
      if (tune_ret != Aoe::AOE_SUCCESS) {
        ADP_LOG(ERROR) << "exec aoe set graph func failed[" << tune_ret << "].";
        return -1;
      }
      // set tuning inputs
      Aoe::AoeStatus set_inputs_ret = (*aoe_set_tuning_graph_input_)(session_id_, inputs);
      if (set_inputs_ret != Aoe::AOE_SUCCESS) {
        ADP_LOG(ERROR) << "exec aoe set tuning inputs func failed[" << set_inputs_ret << "].";
        return -1;
      }
      // aoe tuning
      std::map<Aoe::AscendString, Aoe::AscendString> tuingOptions;
      tuingOptions.insert({Aoe::AscendString("ge.recompute"), Aoe::AscendString(recompute_mode_.c_str())});
      tuingOptions.insert(
          {Aoe::AscendString("ge.aoe_config_file"), Aoe::AscendString(init_options_["ge.aoe_config_file"].c_str())});
      Aoe::AoeStatus aoe_tune_ret = (*aoe_tuning_graph_)(session_id_, tuingOptions);
      if ((aoe_tune_ret != Aoe::AOE_SUCCESS) && (aoe_tune_ret != Aoe::AOE_ERROR_NO_AICORE_GRAPH)) {
        ADP_LOG(ERROR) << "exec aoe tuning func failed[" << aoe_tune_ret << "].";
        return -1;
      }
      ADP_LOG(INFO) << "[GEOP] aoe success[" << aoe_tune_ret << "].";
    }
  }
  return 0;
}

std::string GeOp::BuildSubGraph(FunctionLibraryDefinition *flib_def, const std::string &graph) {
  ADP_LOG(INFO) << "[GEOP] build_sub_graph enter, sub graph name is " << graph;
  const FunctionDef *func_def = flib_def->Find(graph);
  if (func_def == nullptr) {
    ADP_LOG(ERROR) << "[GEOP] Sub graph not found in library, sub graph name is " << graph;
    return "";
  }
  // get infershape
  Graph subgraph(flib_def);
  Status status = InferShapeUtil::GetSubGraphFromFunctionDef(*flib_def, *func_def, &subgraph);
  if (status != Status::OK()) {
    ADP_LOG(ERROR) << "[GEOP] Get subgraph from functiondef fail:" << status.error_message();
    return "";
  }
  ADP_LOG(INFO) << "[GEOP] Get subgraph from functiondef success.";
  std::string enable_force_v2_control;
  (void) ReadStringFromEnvVar("ENABLE_FORCE_V2_CONTROL", "", &enable_force_v2_control);
  if (enable_force_v2_control == "1") {
    GraphDef graph_def;
    subgraph.ToGraphDef(&graph_def);
    WriteTextProto(Env::Default(), GetDumpPath() + graph + "_graph.pbtxt", graph_def);
  }
  bool is_initialize = false;
  for (Node *node : subgraph.nodes()) {
    AddNodeAttrs(node, is_initialize);
    // Add Input&Output Desc into NodeDef
    if (GenerateDesc(node) != Status::OK()) {
      ADP_LOG(WARNING) << "[GEOP] name: " << node->name() << " op:" << node->type_string()
                       << " Generate desc failed in subgraph.";
    }
  }
  unique_ptr<GraphDef> sub_graph_def(new (std::nothrow) GraphDef());
  if (sub_graph_def == nullptr) {
    ADP_LOG(ERROR) << "[GEOP] Malloc memory for subgraph def fail.";
    return "";
  }
  subgraph.ToGraphDef(sub_graph_def.get());
  if (enable_force_v2_control == "1") {
    sub_graph_def->release_library();
    sub_graph_def->mutable_versions()->clear_min_consumer();
  }

  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + "TF_Subgraph_" + graph.c_str() + ".pbtxt";
    (void) WriteTextProto(Env::Default(), pbtxt_path, *sub_graph_def);
  }
  ADP_LOG(INFO) << "[GEOP] build_sub_graph exit, sub graph name is " << graph;
  return sub_graph_def->SerializeAsString();
}

void GeOp::SetDynamicInput() {
  if (dynamic_input_ == "1") {
    graph_options_["ge.exec.dynamicInput"] = dynamic_input_;
    graph_options_["ge.exec.dynamicGraphExecuteMode"] = dynamic_graph_execute_mode_;
    graph_options_["ge.exec.dataInputsShapeRange"] = data_inputs_shape_range_;
    if (dynamic_graph_execute_mode_ == "dynamic_execute" && data_inputs_shape_range_.empty() &&
        getnext_inputs_shape_range_.empty()) {
      graph_options_["ge.shape_generalized_build_mode"] = "shape_generalized";
    }
  }
}

void GeOp::AnalyzeInputDesc(void *tensor_ptr, ge::Tensor &input, ge::DataType type,
                            std::vector<std::string> &input_shapes) const {
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
  ge_tensor_desc.SetOriginShape(ge_shape);
  ge_tensor_desc.SetDataType(type);
  ge_tensor_desc.SetPlacement(output_info->placement_);
  input.SetTensorDesc(ge_tensor_desc);

  uint8_t *data = output_info->data_.release();
  input.SetData(data, output_info->output_size_, output_info->data_.get_deleter());
  ADP_LOG(INFO) << "[GEOP] Get input shape:" << input_shape.DebugString()
                << ", input placement:" << output_info->placement_ << ", input length:" << output_info->output_size_
                << ", input data addr:" << reinterpret_cast<uintptr_t>(data);
}

Status GeOp::AnalyzeStringInput(ge::Tensor &input, uint64_t count, const std::string *string_vector) const {
  uint64_t total_size = 0U;
  for (uint64_t i = 0U; i < count; i++) {
    total_size += (string_vector[i].size() + sizeof(ge::StringHead) + 1U);
  }

  std::unique_ptr<char[]> addr(new (std::nothrow) char[total_size]());
  REQUIRES_NOT_NULL(addr);
  ge::StringHead *string_head = ge::PtrToPtr<char, ge::StringHead>(addr.get());
  char *data_addr = addr.get() + count * sizeof(ge::StringHead);
  int64_t offset = static_cast<int64_t>(count * sizeof(ge::StringHead));
  for (uint64_t i = 0U; i < count; ++i) {
    string_head[i].addr = offset;
    const string &str = string_vector[i];
    string_head[i].len = static_cast<int64_t>(str.size());
    size_t str_size = str.size();
    const char *string_addr = str.c_str();
    while (str_size >= SECUREC_MEM_MAX_LEN) {
      const auto ret = memcpy_s(data_addr, SECUREC_MEM_MAX_LEN, string_addr, SECUREC_MEM_MAX_LEN);
      NPU_REQUIRES(ret == EOK, errors::Internal("call memcpy_s failed, ret:", ret));
      str_size -= SECUREC_MEM_MAX_LEN;
      offset += SECUREC_MEM_MAX_LEN;
      data_addr += SECUREC_MEM_MAX_LEN;
      string_addr += SECUREC_MEM_MAX_LEN;
    }
    auto remain_size = ((total_size - offset) > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : (total_size - offset);
    const auto ret = memcpy_s(data_addr, remain_size, string_addr, str_size + 1U);
    NPU_REQUIRES(ret == EOK, errors::Internal("call memcpy_s failed, ret:", ret));
    data_addr += (str_size + 1U);
    offset += (static_cast<int64_t>(str_size) + 1);
  }
  input.SetData(ge::PtrToPtr<char, const uint8_t>(addr.get()), total_size);
  return Status::OK();
}

Status GeOp::GraphInputConvertToConst(OpKernelContext *ctx) {
  mutex_lock lock{mu_};
  if (is_input_convert_) {
    return Status::OK();
  }
  ADP_LOG(INFO) << "Begin to convet input to const.";
  is_input_convert_ = true;
  NPU_REQUIRES(ctx->function_library() != nullptr,
               errors::Internal("Function:", function_.name(), " ctx function is nullptr"));
  FunctionLibraryDefinition *flib_def =
      const_cast<FunctionLibraryDefinition *>(ctx->function_library()->GetFunctionLibraryDefinition());
  NPU_REQUIRES(flib_def != nullptr,
               errors::Internal("Function:", function_.name(), " flib def is nullptr"));
  const FunctionDef *function_def = flib_def->Find(function_.name());
  NPU_REQUIRES(function_def != nullptr,
               errors::Internal("Function:", function_.name(), " fdef is nullptr"));

  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(InferShapeUtil::GetSubGraphFromFunctionDef(*flib_def, *function_def, &graph));
  for (Node *node : graph.nodes()) {
    if (node->type_string() != "_Arg") {
      continue;
    }

    bool check_value = false;
    for (auto out : node->out_edges()) {
      if (out->dst()->attrs().Find(ATTR_NAME_CONST_INPUT_NAME) == nullptr) {
        continue;
      }
      std::vector<std::string> attr_consts;
      TF_RETURN_IF_ERROR(GetNodeAttr(out->dst()->attrs(), ATTR_NAME_CONST_INPUT_NAME, &attr_consts));
      if (attr_consts.at(out->dst_input()) != "") {
        check_value = true;
      }
    }

    if (check_value) {
      int32_t index = 0;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      Tensor tensor(ctx->input(index));
      std::string const_input_name = "Const" + std::to_string(index);
      Node *const_node = nullptr;
      TF_CHECK_OK(NodeBuilder(const_input_name, "Const")
                  .Device(node->def().device())
                  .Attr("dtype", tensor.dtype())
                  .Attr("value", tensor)
                  .Finalize(&graph, &const_node));
      for (auto out_edge : node->out_edges()) {
        REQUIRES_NOT_NULL(out_edge);
        graph.AddEdge(const_node, out_edge->src_output(), out_edge->dst(), out_edge->dst_input());
      }
      graph.RemoveNode(node);
      remove_index_.push_back(std::make_pair(tensor, index));
    }
  }

  if (remove_index_.size() == 0) {
    ADP_LOG(INFO) << "[GEOP]Return for dont have const input.";
    return Status::OK();
  }

  // refresh the index attr for _Arg
  std::vector<std::pair<Node *, int32_t>> input_infos;
  for (Node *node : graph.nodes()) {
    if (node->type_string() != "_Arg") {
      continue;
    }
    int32_t index = 0;
    (void) GetNodeAttr(node->attrs(), "index", &index);
    input_infos.emplace_back(std::make_pair(node, index));
  }
  std::sort(input_infos.begin(), input_infos.end(), CmpNodeIndex);
  int32_t input_index = 0;
  for (auto &input_info : input_infos) {
    input_info.first->AddAttr("index", input_index);
    input_index++;
  }

  FunctionDefLibrary fdeflib;
  FunctionDef *const_fdef = fdeflib.add_function();
  NPU_REQUIRES_OK(GraphToFunctionDef(graph, function_.name(), const_fdef));
  NPU_REQUIRES_OK(flib_def->RemoveFunction(function_.name()));
  NPU_REQUIRES_OK(flib_def->AddFunctionDef(*const_fdef));
  ADP_LOG(INFO) << "[GEOP] Convert input to const success.";
  return Status::OK();
}

Status GeOp::GraphCheckInputEqualConstOp(Tensor &tensor, int32_t index, bool &is_equal) {
  if (remove_index_.size() == 0) {
    return Status::OK();
  }

  for (auto it : remove_index_) {
    if (it.second != index) {
      continue;
    }
    char *tensor_const = ge::PtrToPtr<void, char>(DMAHelper::base(&it.first));
    char *tensor_input = ge::PtrToPtr<void, char>(DMAHelper::base(&tensor));
    is_equal = ((it.first.TotalBytes() == tensor.TotalBytes()) &&
                (memcmp(tensor_const, tensor_input, tensor.TotalBytes()) == 0));
    if (!is_equal) {
      return errors::Internal("Const input not equal with the input tensor value.");
    }
  }
  ADP_LOG(INFO) << "[GEOP] The input with const flag equal const op value.";
  return Status::OK();
}

Status GeOp::BuildInputTensorInfo(OpKernelContext *const ctx, std::vector<Tensor> &input_vec,
                                  std::vector<std::string> &input_shapes, std::vector<ge::Tensor> &inputs) {
  // ctx is not nullptr
  int num_inputs = ctx->num_inputs();
  std::string cur_input_shapes;

  // populate inputs
  for (int i = 0; i < num_inputs; i++) {
    Tensor tensor(ctx->input(i));
    bool is_equal = false;
    if (GraphCheckInputEqualConstOp(tensor, i, is_equal) != Status::OK()) {
      return errors::Internal("Const op value not equal with tensor :", i);
    } else if (is_equal) {
      continue;
    }
    ADP_LOG(INFO) << "[GEOP] Kernel name: " << ctx->op_kernel().name()
                  << ", Input tensor " << i << " shape: " << tensor.shape().DebugString();
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
      for (uint32_t dim : tensor.shape().dim_sizes()) {
        dims.push_back(static_cast<int64_t>(dim));
      }
      ge::Shape ge_shape(dims);
      ge::TensorDesc ge_tensor_desc(ge_shape);
      ge_tensor_desc.SetDataType(type);
      ge_tensor_desc.SetOriginShape(ge_shape);
      input.SetTensorDesc(ge_tensor_desc);
      if (type == ge::DT_STRING) {
        REQUIRES_NOT_NULL(tensor_ptr);
        const uint64_t count = static_cast<uint64_t>(tensor.NumElements());
        std::string *string_vector = static_cast<std::string *>(tensor_ptr);
        if (AnalyzeStringInput(input, count, string_vector) != Status::OK()) {
          return errors::Internal("The input string data analyze failed.");
        }
      } else {
        input.SetData(static_cast<uint8_t *>(tensor_ptr), total_bytes, [](uint8_t *) {});
      }
      input_shapes.push_back(input_shape);
      cur_input_shapes += input_shape;
    }
    inputs.push_back(input);
    input_vec.push_back(tensor);
  }
  if (sess_options_["ge.inputShape"].empty()) {
    if (!cur_input_shapes.empty() && input_shapes_.empty()) {
      input_shapes_ = cur_input_shapes;
    } else if (jit_compile_ && input_shapes_ != cur_input_shapes && dynamic_input_ != "1") {
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

  std::string format = this->data_format_;
  int32_t domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_RESERVED;
  TF_RETURN_IF_ERROR(this->DomiFormatFromString(format, domi_format));

  // Get signature(dataType) from the OpDef & NodeDef
  DataTypeVector inputs;
  DataTypeVector outputs;
  TF_RETURN_IF_ERROR(tensorflow::InOutTypesForNode(node_def, op_def, &inputs, &outputs));

  int num;
  Node *in_node = nullptr;
  const Edge *in_edge = nullptr;

  if (inputs.size() > INT_MAX) {
    return errors::InvalidArgument("inputs size should be less than INT_MAX.");
  }

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
          attr_format_value.set_i(static_cast<int64_t>(domi_format));
          name_attr_list.mutable_attr()->insert({SERIALIZE_FORMAT, attr_format_value});
          AttrValue attr_datatype_value;
          attr_datatype_value.set_i(static_cast<int64_t>(inputs[num]));
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
          int pre_src_output = in_edge->src_output();
          NameAttrList desc_attr = in_node->def().attr().at(OUTPUT_DESC).list().func(pre_src_output);
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
    const std::string key_shape = tensorflow::KEY_SHAPE;
    AttrValue shape_value;
    const auto &it = node_def.attr().find(key_shape);
    if (it == node_def.attr().end()) {  // no find
      ADP_LOG(WARNING) << "[GEOP] There is no infershape of node : " << node_def.name();
    } else {
      shape_value = node_def.attr().at(key_shape);
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
      attr_format_value.set_i(static_cast<int64_t>(domi_format));
      AttrValue attr_datatype_value;
      attr_datatype_value.set_i(static_cast<int64_t>(data_type));

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
    node_def.mutable_attr()->erase(key_shape);
    node_def.mutable_attr()->insert({OUTPUT_DESC, output_tensor_descs});
  }
  string op_def_string;
  op_def.SerializeToString(&op_def_string);

  tensorflow::AttrValue value;
  value.set_s(op_def_string);
  node_def.mutable_attr()->insert({"op_def", value});
  return tensorflow::Status::OK();
}

Status GeOp::DomiFormatFromString(std::string format, int32_t &domi_format) const {
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
    return Status::OK();
  } else if (format == "ND") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_ND;
    return Status::OK();
  }
  return errors::Unavailable("DomiFormatFromString, not supported format, format = ", format);
}

bool GeOp::IsDynamicConfig() {
  const bool result = !sess_options_["ge.inputShape"].empty() && !sess_options_["ge.dynamicDims"].empty() &&
      !sess_options_["ge.dynamicNodeType"].empty();
  ADP_LOG(INFO) << "[GEOP] IsDynamicConfig result is." << result;
  return result;
}
}  // namespace tensorflow

namespace tensorflow {
mutex GeOp::mu_(LINKER_INITIALIZED);
bool GeOp::tuned_initialize_flag_(false);

const std::string GeOp::INPUT_DESC = "input_tensor_desc";
const std::string GeOp::OUTPUT_DESC = "output_tensor_desc";
const std::string GeOp::SERIALIZE_FORMAT = "serialize_format";
const std::string GeOp::SERIALIZE_DATATYPE = "serialize_datatype";
const std::string GeOp::SERIALIZE_SHAPE = "serialize_shape";
const std::string GeOp::SubGraph = "SubGraph";
std::unordered_map<std::string, uint32_t> GeOp::session_and_graph_id_map_;

REGISTER_KERNEL_BUILDER(Name("GeOp").Device(DEVICE_CPU), GeOp);
}  // namespace tensorflow
