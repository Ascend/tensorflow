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
#include <functional>

#include "dataset_function.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/scope_guard.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_api.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/parser_inner_ctx.h"

#include "tf_adapter/common/common.h"
#include "tf_adapter/util/util.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/infershape_util.h"
#include "tf_adapter/util/ge_plugin.h"
#include "graph/utils/graph_utils.h"
#include "runtime/dev.h"
#include "runtime/mem.h"

namespace tensorflow {
namespace data {
class TensorTransAllocator {
 public:
  virtual void *ReAlloc(void *addr, size_t len) = 0;
  virtual std::function<void(void *)> FreeFunction() = 0;
};

class TfTensorTransToHostAllocator : public TensorTransAllocator {
 public:
  void *ReAlloc(void *addr, size_t len) override {
    return addr;
  }
  std::function<void(void *)> FreeFunction() override {
    return [](void *addr) { (void)addr; };
  }
};

class TfTensorTransToDeviceAllocator : public TensorTransAllocator {
 public:
  void *ReAlloc(void *addr, size_t len) override {
    void *ret_addr;
    rtError_t rt = rtMalloc(&ret_addr, len, RT_MEMORY_HBM);
    if (rt != RT_ERROR_NONE) {
      return nullptr;
    }
    rt = rtMemcpy(ret_addr, len, addr, len, RT_MEMCPY_HOST_TO_DEVICE);
    if (rt != RT_ERROR_NONE) {
      rtFree(ret_addr);
      return nullptr;
    }
    return ret_addr;
  }

  std::function<void(void *)> FreeFunction() override {
    return [](void *addr) { rtFree(addr); };
  }
};

static Status TransTfTensorToGeTensor(std::shared_ptr<domi::ModelParser> &model_parser, Tensor &tf_tensor,
    ge::Tensor &ge_tensor, TensorTransAllocator &allocater) {
  void *tensor_ptr = DMAHelper::base(&tf_tensor);
  REQUIRES_NOT_NULL(tensor_ptr);

  ge::DataType type = model_parser->ConvertToGeDataType(static_cast<uint32_t>(tf_tensor.dtype()));
  if (type == ge::DT_UNDEFINED) {
    ADP_LOG(ERROR) << "[GEOP] No Supported datatype : " << tf_tensor.dtype();
    LOG(ERROR) << "[GEOP] No Supported datatype : " << tf_tensor.dtype();
    return errors::InvalidArgument("No Supported datatype : ", tf_tensor.dtype());
  }

  std::vector<int64_t> dims = DatasetFunction::GetTfShapeDims(tf_tensor.shape());
  ge::Shape ge_shape(dims);
  ge::TensorDesc ge_tensor_desc(ge_shape);
  ge_tensor_desc.SetDataType(type);
  ge_tensor_desc.SetOriginShape(ge_shape);
  ge_tensor.SetTensorDesc(ge_tensor_desc);
  if (type == ge::DT_STRING) {
    return errors::Internal("The ge_tensor string data analyze failed.");
  } else {
    void *paddr = allocater.ReAlloc(tensor_ptr, tf_tensor.TotalBytes());
    ge_tensor.SetData(static_cast<uint8_t *>(paddr), tf_tensor.TotalBytes(), allocater.FreeFunction());
  }
  return Status::OK();
}

static Status TransTfTensorsToGeTensors(std::vector<Tensor> &tf_tensors, std::vector<ge::Tensor> &ge_tensors,
    TensorTransAllocator &allocater) {
  // populate inputs
  std::shared_ptr<domi::ModelParser> model_parser =
        domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  DATASET_REQUIRES(model_parser != nullptr, errors::Unavailable("create model parser ret failed."));

  for (auto tensor : tf_tensors) {
    ge::Tensor ge_tensor;
    Status status = TransTfTensorToGeTensor(model_parser, tensor, ge_tensor, allocater);
    DATASET_REQUIRES(status.ok(), status);
    ge_tensors.push_back(ge_tensor);
  }
  return Status::OK();
}

DatasetFunction::~DatasetFunction() {
  ADP_LOG(INFO) << "~DatasetFunction";
}

void DatasetFunction::DumpTfGraph(const std::string &procPrifex, const std::string &func_name, const GraphDef &graph) {
  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + GetPrefix() + "_" + procPrifex + "_tf_" + func_name + ".pbtxt";
    (void)WriteTextProto(Env::Default(), pbtxt_path, graph);
  }
}

void DatasetFunction::DumpGeComputeGraph(const std::string &procPrifex, const std::string &func_name,
    const ge::ComputeGraphPtr &graph) {
  if (kDumpGraph) {
    const std::string fileName = GetPrefix() + "_" + procPrifex + "_ge_" + func_name;
    ge::GraphUtils::DumpGEGraph(graph, fileName);
    ge::GraphUtils::DumpGEGraphToOnnx(*graph, fileName);
  }
}

Status DatasetFunction::GeError(std::string errorDesc, ge::Status status) const {
  std::stringstream error;
  error << errorDesc << " ret : " << status << std::endl
        << "Error message is : " << std::endl
        << ge::GEGetErrorMsg();
  return errors::Internal(error.str());
}

Status DatasetFunction::AddOpDef(Node &node) const {
  const OpDef &op_def = node.op_def();
  NodeDef &node_def = const_cast<NodeDef &>(node.def());

  std::string op_def_string;
  op_def.SerializeToString(&op_def_string);

  tensorflow::AttrValue value;
  value.set_s(op_def_string);
  node_def.mutable_attr()->insert({"op_def", value});
  return Status::OK();
}

Status DatasetFunction::RefreshNodeDesc(Node &node) const {
  return AddOpDef(node);
}

std::string DatasetFunction::BuildSubGraph(FunctionLibraryDefinition &flib_def, const std::string &func_name) {
  const FunctionDef *func_def = flib_def.Find(func_name);
  DATASET_REQUIRES(func_def != nullptr, "");

  Graph sub_graph(&flib_def);

  Status status = InferShapeUtil::GetSubGraphFromFunctionDef(flib_def, *func_def, &sub_graph);
  if (!status.ok()) {
    ADP_LOG(ERROR) << status.ToString();
    return "";
  }

  for (Node *node : sub_graph.nodes()) {
    status = RefreshNodeDesc(*node);
    DATASET_REQUIRES(status.ok(), "");
  }

  GraphDef sub_graph_def;
  sub_graph.ToGraphDef(&sub_graph_def);

  DumpTfGraph(std::string("Build"), func_name, sub_graph_def);
  return sub_graph_def.SerializeAsString();
}

Status DatasetFunction::CreateGeGraph(const std::shared_ptr<domi::ModelParser> &model_parser,
    FunctionLibraryDefinition &flib_def) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>(funcName_);
  DATASET_REQUIRES(model_parser != nullptr, errors::Internal("Create compute graph failed."));

  auto build_sub_graph = [this, &flib_def](const std::string &graph) -> std::string {
    return this->BuildSubGraph(flib_def, graph);
  };
  auto graph_def = build_sub_graph(funcName_);
  ge::Status status = model_parser->ParseProtoWithSubgraph(graph_def, build_sub_graph, compute_graph);
  DATASET_REQUIRES(status == ge::SUCCESS, GeError("Parse proto with graph failed.", status));

  DumpGeComputeGraph(std::string("Build"), funcName_, compute_graph);

  ge_graph_ = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  return Status::OK();
}

std::vector<int64_t> DatasetFunction::GetTfShapeDims(const PartialTensorShape &tf_shape) {
  std::vector<int64_t> dims;
  const std::vector<int64_t> kUnknowShape = {-2};
  if (!tf_shape.unknown_rank()) {
    dims.clear();
    for (auto it : tf_shape) { dims.push_back(it.size); }
  } else {
    dims = kUnknowShape;
  }
  return dims;
}

std::vector<int64_t> DatasetFunction::GetTfShapeDims(const TensorShape &tf_shape) {
  std::vector<int64_t> dims;
  const std::vector<int64_t> kUnknowShape = {-2};
  if (!tf_shape.unknown_rank()) {
    dims.clear();
    for (auto it : tf_shape) { dims.push_back(it.size); }
  } else {
    dims = kUnknowShape;
  }
  return dims;
}

ge::InputTensorInfo DatasetFunction::BuildTensorInfo(const std::shared_ptr<domi::ModelParser> &model_parser,
    DataType type, const PartialTensorShape &shape) const {
  ge::InputTensorInfo tensorInfo =
    {static_cast<uint32_t>(model_parser->ConvertToGeDataType(static_cast<uint32_t>(type))), {}, nullptr, 0};
  tensorInfo.dims = GetTfShapeDims(shape);
  return tensorInfo;
}

std::vector<ge::InputTensorInfo> DatasetFunction::BuildInputTensorInfos(
    const std::shared_ptr<domi::ModelParser> &model_parser) const {
  std::vector<ge::InputTensorInfo> inputTensorInfos;
  int input_num = input_types_.size();
  for (int i = 0; i < input_num; i++) {
    inputTensorInfos.push_back(BuildTensorInfo(model_parser, input_types_[i], input_shape_[i]));
  }
  return inputTensorInfos;
}

Status DatasetFunction::BuildGeGraph(const Instance &instance,
    const std::shared_ptr<domi::ModelParser> &model_parser) {
  std::vector<ge::InputTensorInfo> inputs = BuildInputTensorInfos(model_parser);
  ge::Status status = ge_session_->BuildGraph(instance, inputs);
  DATASET_REQUIRES(status == ge::SUCCESS, GeError("Build graph failed.", status));
  return Status::OK();
}

Status DatasetFunction::InitGeGraph(FunctionLibraryDefinition &flib_def) {
  std::shared_ptr<domi::ModelParser> model_parser =
        domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  DATASET_REQUIRES(model_parser != nullptr, errors::Unavailable("create model parser ret failed."));

  return CreateGeGraph(model_parser, flib_def);
}

Status DatasetFunction::Instantialte(Instance &instance) {
  std::shared_ptr<domi::ModelParser> model_parser =
      domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  DATASET_REQUIRES(model_parser != nullptr, errors::Unavailable("create model parser ret failed."));

  std::map<ge::AscendString, ge::AscendString> graph_options;
  instance = NewGraphId();
  ge::Status status = ge_session_->AddGraphWithCopy(instance, ge_graph_, graph_options);
  DATASET_REQUIRES(status == ge::SUCCESS, GeError("Add graph failed.", status));

  return BuildGeGraph(instance, model_parser);
}

void DatasetFunction::LogOptions(const std::map<std::string, std::string> &options) {
  for (const auto& [key, value] : options) {
    ADP_LOG(INFO) << "  name: " << key << ", value = " << value;
  }
}

Status DatasetFunction::Initialize(const std::map<std::string, std::string> &session_options,
    FunctionLibraryDefinition *flib_def) {
  session_options_ = session_options;

  if (!GePlugin::GetInstance()->IsGlobal()) {
    ADP_LOG(INFO) << "init_options:";
    LogOptions(init_options_);
    GePlugin::GetInstance()->Init(init_options_);
  }

  std::shared_ptr<domi::ModelParser> model_parser =
        domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  DATASET_REQUIRES(model_parser != nullptr, errors::Unavailable("create model parser ret failed."));

  std::transform(output_types_.begin(), output_types_.end(), std::back_inserter(ge_output_types_),
      [&model_parser](const DataType type) { return model_parser->ConvertToGeDataType(static_cast<uint32_t>(type)); });

  ADP_LOG(INFO) << "session_options:";
  LogOptions(session_options);
  ge_session_.reset(new (std::nothrow)ge::Session(session_options));

  return InitGeGraph(*flib_def);
}

Status DatasetFunction::Run(Instance instance, std::vector<ge::Tensor> &in_tensors,
    std::vector<ge::Tensor> &out_tensors) {
  ge::Status status = ge_session_->RunGraph(instance, in_tensors, out_tensors);
  return (status == ge::SUCCESS) ? Status::OK() : GeError("Run graph failed.", status);
}

Status DatasetFunction::Run(Instance instance, std::vector<Tensor> &in_tensors,
    std::vector<ge::Tensor> &out_tensors) {
  std::vector<ge::Tensor> inputs;
  TfTensorTransToHostAllocator trans_alloc;
  Status status = TransTfTensorsToGeTensors(in_tensors, inputs, trans_alloc);
  DATASET_REQUIRES(status.ok(), status);

  return Run(instance, inputs, out_tensors);
}

Status DatasetFunction::RunWithStreamAsyn(Instance instance, void *stream, std::vector<ge::Tensor> &in_tensors,
    std::vector<ge::Tensor> &out_tensors) {
  ge::Status status = ge_session_->RunGraphWithStreamAsync(instance, stream, in_tensors, out_tensors);
  return (status == ge::SUCCESS) ? Status::OK() : GeError("Run graph with stream failed.", status);
}

Status DatasetFunction::RunWithStreamAsyn(Instance instance, void *stream, std::vector<Tensor> &in_tensors,
    std::vector<ge::Tensor> &out_tensors) {
  std::vector<ge::Tensor> inputs;
  TfTensorTransToDeviceAllocator trans_alloc;
  Status status = TransTfTensorsToGeTensors(in_tensors, inputs, trans_alloc);
  DATASET_REQUIRES(status.ok(), status);

  return RunWithStreamAsyn(instance, stream, inputs, out_tensors);
}

bool DatasetFunction::IsUnknowShape(const PartialTensorShape &tf_shape) {
  if (tf_shape.unknown_rank()) {
    return true;
  }

  for (auto it : tf_shape) {
    if (it.size < 0) {
      return true;
    }
  }
  return false;
}

bool DatasetFunction::HaveUnknowShape(const std::vector<PartialTensorShape> tf_shapes) {
  for (auto it : tf_shapes) {
    if (IsUnknowShape(it)) {
      return true;
    }
  }
  return false;
}

Status DatasetFunction::RegisterNpuCancellation(std::function<void()> callback, std::function<void()>* deregister_fn) {
  return RegisterNpuCancellationCallback(callback, deregister_fn);
}
}  // namespace data
}  // namespace tensorflow