/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/memory/memory_api.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/parser_api.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/omg/parser/parser_inner_ctx.h"
#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "graph/tensor.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_adapter.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "graph/buffer.h"
#include "graph/model.h"

#include <iostream>

namespace ge {
namespace {
const std::map<uint32_t, ge::DataType> data_type_map = {
  {tensorflow::DataType::DT_FLOAT, ge::DataType::DT_FLOAT},
  {tensorflow::DataType::DT_HALF, ge::DataType::DT_FLOAT16},
  {tensorflow::DataType::DT_INT8, ge::DataType::DT_INT8},
  {tensorflow::DataType::DT_INT16, ge::DataType::DT_INT16},
  {tensorflow::DataType::DT_UINT16, ge::DataType::DT_UINT16},
  {tensorflow::DataType::DT_UINT8, ge::DataType::DT_UINT8},
  {tensorflow::DataType::DT_INT32, ge::DataType::DT_INT32},
  {tensorflow::DataType::DT_INT64, ge::DataType::DT_INT64},
  {tensorflow::DataType::DT_UINT32, ge::DataType::DT_UINT32},
  {tensorflow::DataType::DT_UINT64, ge::DataType::DT_UINT64},
  {tensorflow::DataType::DT_BOOL, ge::DataType::DT_BOOL},
  {tensorflow::DataType::DT_DOUBLE, ge::DataType::DT_DOUBLE},
  {tensorflow::DataType::DT_COMPLEX64, ge::DataType::DT_COMPLEX64},
  {tensorflow::DataType::DT_QINT8, ge::DataType::DT_INT8},
  {tensorflow::DataType::DT_QUINT8, ge::DataType::DT_UINT8},
  {tensorflow::DataType::DT_QINT32, ge::DataType::DT_INT32},
  {tensorflow::DataType::DT_QINT16, ge::DataType::DT_INT16},
  {tensorflow::DataType::DT_QUINT16, ge::DataType::DT_UINT16},
  {tensorflow::DataType::DT_COMPLEX128, ge::DataType::DT_COMPLEX128},
  {tensorflow::DataType::DT_RESOURCE, ge::DataType::DT_RESOURCE},
  {tensorflow::DataType::DT_BFLOAT16, ge::DataType::DT_FLOAT16},
  {tensorflow::DataType::DT_STRING, ge::DataType::DT_STRING},
  {tensorflow::DataType::DT_FLOAT_REF, ge::DataType::DT_FLOAT},
  {tensorflow::DataType::DT_DOUBLE_REF, ge::DataType::DT_DOUBLE},
  {tensorflow::DataType::DT_INT32_REF, ge::DataType::DT_INT32},
  {tensorflow::DataType::DT_INT8_REF, ge::DataType::DT_INT8},
  {tensorflow::DataType::DT_UINT8_REF, ge::DataType::DT_UINT8},
  {tensorflow::DataType::DT_INT16_REF, ge::DataType::DT_INT16},
  {tensorflow::DataType::DT_UINT16_REF, ge::DataType::DT_UINT16},
  {tensorflow::DataType::DT_COMPLEX64_REF, ge::DataType::DT_COMPLEX64},
  {tensorflow::DataType::DT_QINT8_REF, ge::DataType::DT_INT8},
  {tensorflow::DataType::DT_QUINT8_REF, ge::DataType::DT_UINT8},
  {tensorflow::DataType::DT_QINT32_REF, ge::DataType::DT_INT32},
  {tensorflow::DataType::DT_QINT16_REF, ge::DataType::DT_INT16},
  {tensorflow::DataType::DT_QUINT16_REF, ge::DataType::DT_UINT16},
  {tensorflow::DataType::DT_COMPLEX128_REF, ge::DataType::DT_COMPLEX128},
  {tensorflow::DataType::DT_RESOURCE_REF, ge::DataType::DT_RESOURCE},
  {tensorflow::DataType::DT_BFLOAT16_REF, ge::DataType::DT_FLOAT16},
  {tensorflow::DataType::DT_UINT32_REF, ge::DataType::DT_UINT32},
  {tensorflow::DataType::DT_UINT64_REF, ge::DataType::DT_UINT64},
  {tensorflow::DataType::DT_INT64_REF, ge::DataType::DT_INT64},
  {tensorflow::DataType::DT_BOOL_REF, ge::DataType::DT_BOOL},
  {tensorflow::DataType::DT_HALF_REF, ge::DataType::DT_FLOAT16},
  {tensorflow::DataType::DT_STRING_REF, ge::DataType::DT_STRING},
  {tensorflow::DataType::DT_VARIANT, ge::DataType::DT_VARIANT},
};
} // end 
class TensorFlowModelParser : public domi::ModelParser {
 public:
  TensorFlowModelParser() {}
  virtual ~TensorFlowModelParser() {}

  Status Parse(const char *file, ge::Graph &graph) override;

  Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) override;

  Status ParseFromMemory(const char *data, uint32_t size, ge::Graph &graph) override {
    return domi::SUCCESS;
  }

  Status ToJson(const char *model_file, const char *json_file) override;

  Status ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph) override;

  Status ParseProtoWithSubgraph(const google::protobuf::Message *proto, domi::GetGraphCallback callback,
                                ge::ComputeGraphPtr &graph) override;

  Status ParseProtoWithSubgraph(const std::string &serialized_proto, domi::GetGraphCallbackV2 callback,
                                ge::ComputeGraphPtr &graph) override;

  ge::DataType ConvertToGeDataType(const uint32_t type) override;

  Status ParseAllGraph(const google::protobuf::Message *root_proto, ge::ComputeGraphPtr &root_graph) override ;
};
ge::DataType TensorFlowModelParser::ConvertToGeDataType(const uint32_t type) {
  auto search = data_type_map.find(type);
  if (search != data_type_map.end()) {
    return search->second;
  } else {
    return ge::DataType::DT_UNDEFINED;
  }
}

Status TensorFlowModelParser::Parse(const char *file, ge::Graph &graph) { return ge::SUCCESS; }

Status TensorFlowModelParser::ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph) { return ge::SUCCESS; }

Status TensorFlowModelParser::ToJson(const char *model_file, const char *json_file) { return ge::SUCCESS; }

Status TensorFlowModelParser::ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph) { return ge::SUCCESS; }

Status TensorFlowModelParser::ParseProtoWithSubgraph(const std::string &serialized_proto,
                                                     domi::GetGraphCallbackV2 callback,
                                                     ge::ComputeGraphPtr &graph) {
  callback("finall_branch1_Y3CNZMF9Vv8");
  return ge::SUCCESS;
}

Status TensorFlowModelParser::ParseProtoWithSubgraph(const google::protobuf::Message *proto, domi::GetGraphCallback callback,
                                                     ge::ComputeGraphPtr &graph) { return ge::SUCCESS; }

Status TensorFlowModelParser::ParseAllGraph(const google::protobuf::Message *root_proto, ge::ComputeGraphPtr &root_graph) { return ge::SUCCESS; }

static std::map<uint32_t, ge::Graph> graphs_map;
static std::atomic<bool> is_ge_init(false);
static std::atomic<bool> is_parser_init(false);

GE_FUNC_VISIBILITY Status InitRdmaPool(size_t size, rtMemType_t mem_type) {
  if (size == 0) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

GE_FUNC_VISIBILITY Status RdmaRemoteRegister(const std::vector<HostVarInfo> &var_info,
                                             rtMemType_t mem_type) {
  if (var_info.empty()) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

GE_FUNC_VISIBILITY Status MallocSharedMemory(const TensorInfo &tensor_info, uint64_t &dev_addr, uint64_t &memory_size) {
  if (tensor_info.var_name.empty()) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

GE_FUNC_VISIBILITY Status GetVarBaseAddrAndSize(const std::string &var_name, uint64_t &base_addr, uint64_t &var_size) {
  if (var_name.empty()) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

GE_FUNC_VISIBILITY Status GEInitialize(const std::map<std::string, std::string> &options) {
  if (options.empty()) {
    return ge::FAILED;
  }
  is_ge_init = true;
  return ge::SUCCESS;
}

GE_FUNC_VISIBILITY Status GEFinalize() {
  if (!is_ge_init) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

Status ParserInitialize(const std::map<std::string, std::string> &options) {
  if (options.empty()) {
    return ge::FAILED;
  }
  is_parser_init = true;
  return ge::SUCCESS;
}

Status ParserFinalize() {
  if (!is_parser_init) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

GE_FUNC_VISIBILITY std::string GEGetErrorMsg() { return "ERROR";}

Session::Session(const std::map<string, string> &options) {}

Session::~Session() {
  graphs_map.clear();
}

Status Session::RemoveGraph(uint32_t graphId) {
  auto ret = graphs_map.find(graphId);
  if (ret != graphs_map.end()) {
    graphs_map.erase(ret);
    return ge::SUCCESS;
  }
  return ge::FAILED;
}

bool Session::IsGraphNeedRebuild(uint32_t graphId) {
  auto ret = graphs_map.find(graphId);
  if (ret != graphs_map.end()) {
    return false;
  }
  return true;
}

Status Session::AddGraph(uint32_t graphId, const Graph &graph, const std::map<std::string, std::string> &options) {
  auto ret = graphs_map.find(graphId);
  if (ret != graphs_map.end()) {
    return ge::FAILED;
  }
  graphs_map[graphId] = graph;
  return ge::SUCCESS;
}

Status Session::BuildGraph(uint32_t graphId, const std::vector<ge::Tensor> &inputs) {
  auto ret = graphs_map.find(graphId);
  if (ret == graphs_map.end()) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

Status Session::RunGraphAsync(uint32_t graphId, const std::vector<ge::Tensor> &inputs, RunAsyncCallback callback) {
  ge::Status ret;
  std::vector<ge::Tensor> outputs;
  outputs.push_back(ge::Tensor());
  auto res = graphs_map.find(graphId);
  if (res == graphs_map.end()) {
    ret = ge::FAILED;
  } else {
    ret = ge::SUCCESS;
  }
  callback(ret, outputs);
  return ret;
}

ComputeGraph::ComputeGraph(const std::string &name)
    : name_(name), nodes_(), input_nodes_(), sub_graph_(), is_valid_flag_(false), need_iteration_(false) {
}

ComputeGraph::~ComputeGraph() {}

ProtoAttrMapHelper ComputeGraph::MutableAttrMap() { return attrs_; }

ConstProtoAttrMapHelper ComputeGraph::GetAttrMap() const {
  return ConstProtoAttrMapHelper(attrs_.GetProtoOwner(), attrs_.GetProtoMsg());
}

size_t ComputeGraph::GetAllNodesSize() const {
  if (name_ == "total_0") { return 0; }
  return 1;
}

Graph::Graph(const std::string& grph) {}
Graph::Graph(char const* name) {}

Graph GraphUtils::CreateGraphFromComputeGraph(const ComputeGraphPtr compute_graph) { return Graph("ge"); }

void Graph::SetNeedIteration(bool need_iteration) {}

GNode::GNode() {}

std::vector<GNode> Graph::GetAllNodes() const {
  std::vector<GNode> res;
  GNode node;
  res.push_back(node);
  return res;
}

NodePtr NodeAdapter::GNode2Node(ge::GNode const &node) {
  return nullptr;
}

std::string Node::GetName() const {
  return "";
}

OpDescPtr Node::GetOpDesc() const {
  return nullptr;
}

void OpDesc::SetName(std::string const &name) {}

graphStatus aclgrphParseONNX(const char *model_file,
    const std::map<ge::AscendString, ge::AscendString> &parser_params, ge::Graph &graph) {
  std::string model_(model_file);
  if(model_ == "no_model") {
    return FAILED;
  }
  return SUCCESS;
}

Buffer::Buffer() {}
std::size_t Buffer::GetSize() const {
  return sizeof("_external_model");
}
std::uint8_t *Buffer::GetData() {
  std::string *buf_ = new std::string("_external_model");
  return reinterpret_cast<std::uint8_t *> (buf_);
}
const std::uint8_t *Buffer::GetData() const {
  std::string *buf_ = new std::string("_external_model");
  return reinterpret_cast<const std::uint8_t *> (buf_);
}

Model::Model() {}
Model::Model(const string &name, const string &custom_version) {}
void Model::SetGraph(const Graph& graph) {}
graphStatus Model::Save(Buffer &buffer, bool is_dump) const {
  return GRAPH_SUCCESS;
}
ConstProtoAttrMapHelper Model::GetAttrMap() const {
  return ConstProtoAttrMapHelper();
}
ProtoAttrMapHelper Model::MutableAttrMap() {
  return ProtoAttrMapHelper();
}

Tensor::Tensor() {}

graphStatus Tensor::SetTensorDesc(const TensorDesc &tensorDesc) { return GRAPH_SUCCESS; }

TensorDesc Tensor::GetTensorDesc() const { return TensorDesc(); }

graphStatus Tensor::SetData(const uint8_t *data, size_t size) { return GRAPH_SUCCESS; }

graphStatus Tensor::SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func) { return GRAPH_SUCCESS; }

size_t Tensor::GetSize() const { return 4; }

Shape::Shape(const std::vector<int64_t> &dims) {}

Shape::Shape() {}

std::vector<int64_t> Shape::GetDims() const {
  std::vector<int64_t> dims;
  return dims;
}

TensorDesc::TensorDesc() {}

TensorDesc::TensorDesc(Shape shape, Format format, DataType dt) {}

void TensorDesc::SetDataType(DataType dt) {}

Shape TensorDesc::GetShape() const { return Shape(); }

static Placement ge_placement = ge::kPlacementHost;

Placement TensorDesc::GetPlacement() const { return ge_placement; }

void TensorDesc::SetPlacement(ge::Placement placement) { ge_placement = placement; }

void TensorDesc::SetOriginShape(const Shape &originShape) {}

std::unique_ptr<uint8_t[], Tensor::DeleteFunc> Tensor::ResetData() { return nullptr; }

ParserContext &GetParserContext() {
  static ParserContext context;
  return context;
}
} // end ge

namespace domi {
ge::OmgContext &GetContext() {
  static ge::OmgContext context;
  return context;
}

ModelParserFactory *ModelParserFactory::Instance() {
  static ModelParserFactory instance;
  return &instance;
}

std::shared_ptr<domi::ModelParser> ModelParserFactory::CreateModelParser(const domi::FrameworkType type) {
  return std::make_shared<ge::TensorFlowModelParser>();
}

ModelParserFactory::~ModelParserFactory() {}
} // end domi
