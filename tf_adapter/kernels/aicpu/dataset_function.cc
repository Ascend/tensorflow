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
#include "dataset_function.h"

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
#include <cstdlib>

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
#include "graph/utils/graph_utils_ex.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "runtime/dev.h"
#include "runtime/mem.h"
#include "tf_adapter/util/infershape_util.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"

namespace tensorflow {
namespace data {
namespace {
static constexpr const char* const kInputDesc = "input_tensor_desc";
static constexpr const char* const kOutputDesc = "output_tensor_desc";
static constexpr const char* const kFormat = "serialize_format";
static constexpr const char* const kType = "serialize_datatype";
static constexpr const char* const kShape = "serialize_shape";
static constexpr const char* const kPreProcessOps = "preprocess_ops";
static constexpr const char* const kControlOps = "control_ops";
static constexpr const char* const kTfaSplitGraphJsonPath = "/built-in/framework/tensorflow/dp_split_graph_ops.json";
// Indicates the chips that allows split graph
// rule of this map : {"real chip name", "dvpp ops ini file name"}
const static std::map<std::string, std::string> kDvppAccelerateChips = {
    {"Ascend910B1", "ascend910b"},
    {"Ascend910B2", "ascend910b"},
    {"Ascend910B3", "ascend910b"}};
} // namespace

void *DatasetFunction::ReAllocDeviceMem(void *addr, size_t len) {
  void *ret_addr;
  aclError ret = aclrtMalloc(&ret_addr, len, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != ACL_SUCCESS) {
    return nullptr;
  }
  ret = aclrtMemcpy(ret_addr, len, addr, len, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_ERROR_NONE) {
    (void)aclrtFree(ret_addr);
    return nullptr;
  }
  return ret_addr;
}

template <typename T>
tensorflow::AttrValue DatasetFunction::BuildDescAttr(T shapes, TensorDataTypes types) const {
  tensorflow::AttrValue desc_attr;
  for (size_t i = 0; i < types.size(); i++) {
    auto desc = desc_attr.mutable_list()->add_func();
    desc->set_name(std::to_string(i));

    tensorflow::AttrValue shape_value;
    if (shapes[i].unknown_rank()) {
      const int kUnknownRankDimSize = -2;
      shape_value.mutable_list()->add_i(kUnknownRankDimSize);
    } else {
      for (int j = 0; j < shapes[i].dims(); j++) {
        shape_value.mutable_list()->add_i(shapes[i].dim_size(j));
      }
    }
    (void)desc->mutable_attr()->insert({kShape, shape_value});

    tensorflow::AttrValue type_value;
    type_value.set_i(static_cast<int64_t>(types[i]));
    (void)desc->mutable_attr()->insert({kType, type_value});

    tensorflow::AttrValue format_value;
    format_value.set_i(static_cast<int>(ge::Format::FORMAT_NHWC));
    (void)desc->mutable_attr()->insert({kFormat, format_value});
  }
  return desc_attr;
}

void DatasetFunction::AssembleInputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node &n) const {
  n.AddAttr(kInputDesc, BuildDescAttr(std::move(shapes), std::move(types)));
}

void DatasetFunction::AssembleOutputDesc(TensorPartialShapes shapes, TensorDataTypes types, tensorflow::Node &n) const {
  n.AddAttr(kOutputDesc, BuildDescAttr(std::move(shapes), std::move(types)));
}

void DatasetFunction::AssembleOpDef(tensorflow::Node &n) const {
  const tensorflow::OpRegistrationData *op_reg_data;
  (void)tensorflow::OpRegistry::Global()->LookUp(n.type_string(), &op_reg_data);
  std::string serialized_op_def;
  (void)op_reg_data->op_def.SerializeToString(&serialized_op_def);
  n.AddAttr("op_def", serialized_op_def);
}

void DatasetFunction::AssembleParserAddons(const tensorflow::FunctionLibraryDefinition &lib_def,
    tensorflow::Graph &graph) const {
  tensorflow::ShapeRefiner shape_refiner(graph.versions(), &lib_def);
  auto node_shape_inference_lambda = [this, &shape_refiner](tensorflow::Node *node) {
    this->AssembleOpDef(*node);
    auto status = shape_refiner.AddNode(node);
    if (!status.ok()) {
      ADP_LOG(INFO) << "  " << node->name() << "[" << node->type_string() << "] Skip infer " << status.error_message();
      return;
    }
    auto node_ctx = shape_refiner.GetContext(node);

    TensorDataTypes input_types;
    TensorDataTypes output_types;
    (void)tensorflow::InOutTypesForNode(node->def(), node->op_def(), &input_types, &output_types);

    if (!input_types.empty()) {
      tensorflow::AttrValue input_desc_attrs;
      bool input_desc_incomplete = false;
      for (int i = 0; i < node->num_inputs(); i++) {
        const tensorflow::Edge *edge = nullptr;
        status = node->input_edge(i, &edge);
        if (!status.ok()) {
          ADP_LOG(ERROR) << status.ToString();
          return;
        }

        auto input_attr = edge->src()->attrs().Find(kOutputDesc);
        if (input_attr == nullptr) {
          input_desc_incomplete = true;
          ADP_LOG(WARNING) << node->DebugString() << " input node " << edge->src()->DebugString()
                           << " has no desc for output " << edge->src_output();
          break;
        }
        *input_desc_attrs.mutable_list()->add_func() =
          edge->src()->attrs().Find(kOutputDesc)->list().func(edge->src_output());
      }
      if (!input_desc_incomplete) {
        node->AddAttr(kInputDesc, input_desc_attrs);
      } else {
        TensorPartialShapes input_shapes;
        for (int i = 0; i < node_ctx->num_inputs(); ++i) {
          tensorflow::TensorShapeProto proto;
          node_ctx->ShapeHandleToProto(node_ctx->input(i), &proto);
          (void)input_shapes.emplace_back(proto);
        }
        this->AssembleInputDesc(input_shapes, input_types, *node);
      }
    }

    if (!output_types.empty()) {
      TensorPartialShapes output_shapes;
      for (int i = 0; i < node_ctx->num_outputs(); ++i) {
        tensorflow::TensorShapeProto proto;
        node_ctx->ShapeHandleToProto(node_ctx->output(i), &proto);
        (void)output_shapes.emplace_back(proto);
        ADP_LOG(INFO) << "    output " << i << ": " << tensorflow::DataTypeString(output_types[static_cast<size_t>(i)])
               << node_ctx->DebugString(node_ctx->output(i));
      }
      this->AssembleOutputDesc(output_shapes, output_types, *node);
    }
  };
  tensorflow::ReverseDFS(graph, {}, node_shape_inference_lambda);
}

void *DatasetFunction::ConvertDTStringTensor(const Tensor &tf_tensor, uint64_t &tensor_size) {
  const uint64_t count = static_cast<uint64_t>(tf_tensor.NumElements());

#if defined(TF_VERSION_TF2)
  const tstring *string_vector = tf_tensor.flat<tstring>().data();
#else
  const std::string *string_vector = static_cast<const std::string *>(DMAHelper::base(&tf_tensor));
#endif

  uint64_t total_size = 0U;
  uint64_t string_head_size = sizeof(ge::StringHead);
  for (uint64_t i = 0U; i < count; i++) {
    // add 1U for the end of string identifier '\0'
    total_size += (string_vector[i].size() + string_head_size + 1U);
  }
  DATASET_REQUIRES_RETURN_NULL(total_size != 0U,
      errors::Internal("Convert string data failed with total size equals 0."));

  std::unique_ptr<char[]> addr = absl::make_unique<char[]>(total_size);
  DATASET_REQUIRES_RETURN_NULL(addr != nullptr,
      errors::Internal("Malloc host memory failed."));
  ge::StringHead *string_head = ge::PtrToPtr<char, ge::StringHead>(addr.get());
  DATASET_REQUIRES_RETURN_NULL(!DatasetFunction::CheckMultiplyOverflow(count, string_head_size),
      errors::Unavailable("Wrong offset, count = ", count,
                          ", string_head_size = ", string_head_size));
  uint64_t offset = count * string_head_size;
  char *data_addr = addr.get() + offset;
  for (uint64_t i = 0U; i < count; ++i) {
    string_head[i].addr = offset;
    const string &str = string_vector[i];
    string_head[i].len = static_cast<int64_t>(str.size());
    size_t str_size = str.size();
    const char *string_addr = str.c_str();
    while (str_size >= SECUREC_MEM_MAX_LEN) {
      const auto ret = memcpy_s(data_addr, total_size - offset, string_addr, SECUREC_MEM_MAX_LEN);
      DATASET_REQUIRES_RETURN_NULL(ret == EOK, errors::Internal("call memcpy_s failed, ret:", ret));
      str_size -= SECUREC_MEM_MAX_LEN;
      offset += SECUREC_MEM_MAX_LEN;
      data_addr += SECUREC_MEM_MAX_LEN;
      string_addr += SECUREC_MEM_MAX_LEN;
    }
    auto remain_size = total_size - offset;
    const auto ret = memcpy_s(data_addr, remain_size, string_addr, str_size + 1U);
    DATASET_REQUIRES_RETURN_NULL(ret == EOK, errors::Internal("call memcpy_s failed, ret:", ret));
    data_addr += (str_size + 1U);
    offset += (static_cast<int64_t>(str_size) + 1);
  }

  tensor_size = total_size;
  return reinterpret_cast<void *>(addr.release());
}

Status DatasetFunction::TransTfTensorToDataBuffer(aclmdlDataset *input_dataset, Tensor &tf_tensor) {
  void *tensor_ptr = DMAHelper::base(&tf_tensor);
  REQUIRES_NOT_NULL(tensor_ptr);

  std::shared_ptr<domi::ModelParser> model_parser =
        domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  DATASET_REQUIRES(model_parser != nullptr, errors::Unavailable("create model parser ret failed."));

  ge::DataType type = model_parser->ConvertToGeDataType(static_cast<uint32_t>(tf_tensor.dtype()));
  if (type == ge::DT_UNDEFINED) {
    ADP_LOG(ERROR) << "[DatasetFunction] No Supported datatype : " << tf_tensor.dtype();
    LOG(ERROR) << "[DatasetFunction] No Supported datatype : " << tf_tensor.dtype();
    return errors::InvalidArgument("No Supported datatype : ", tf_tensor.dtype());
  }

  void *tensor_addr = nullptr;
  uint64_t tensor_size = 0ULL;
  if (type == ge::DT_STRING) {
    tensor_addr = DatasetFunction::ConvertDTStringTensor(tf_tensor, tensor_size);
  } else {
    tensor_addr = tensor_ptr;
    tensor_size = tf_tensor.TotalBytes();
  }
  DATASET_REQUIRES(tensor_addr != nullptr,
      errors::Internal("Convert input string data failed. tensor addr is nullptr."));
  DATASET_REQUIRES(tensor_size != 0ULL,
      errors::Internal("Convert input string data failed. tensor size is 0."));

  void *device_addr = ReAllocDeviceMem(tensor_addr, tensor_size);
  DATASET_REQUIRES(device_addr != nullptr, errors::Internal("Create device memory for input tensor failed."));
  aclDataBuffer* inputData = aclCreateDataBuffer(device_addr, tensor_size);
  if (inputData == nullptr) {
    (void)aclrtFree(device_addr);
    return errors::Internal("Create data buffer for input tensor failed.");
  }

  aclError ret = aclmdlAddDatasetBuffer(input_dataset, inputData);
  if (ret != ACL_SUCCESS) {
    (void)aclrtFree(device_addr);
    (void)aclDestroyDataBuffer(inputData);
    return errors::Internal("Can't add data buffer, create input failed.");
  }
  return Status::OK();
}

aclmdlDataset *DatasetFunction::CreateAclInputDatasetWithTFTensors(std::vector<Tensor> &tf_tensors) {
  aclmdlDataset* input_dataset = aclmdlCreateDataset();
  if (input_dataset == nullptr) {
    return nullptr;
  }

  for (size_t i = 0; i < tf_tensors.size(); i++) {
    Status status = TransTfTensorToDataBuffer(input_dataset, tf_tensors[i]);
    if (!status.ok()) {
      DestroyAclInputDataset(input_dataset);
      return nullptr;
    }
  }

  return input_dataset;
}

void DatasetFunction::DestroyAclInputDataset(aclmdlDataset *input) {
  if (input == nullptr) {
    return;
  }
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input); i++) {
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(input, i);
    void* data_addr = aclGetDataBufferAddr(data_buffer);
    CHECK_NOT_NULL(data_addr);
    aclError ret = aclrtFree(data_addr);
    if (ret != ACL_ERROR_NONE) {
      ADP_LOG(ERROR) << "Free acl device memory failed.";
      return;
    }
    ret = aclDestroyDataBuffer(data_buffer);
    if (ret != ACL_ERROR_NONE) {
      ADP_LOG(ERROR) << "Destory acl input dataset buffer failed.";
      return;
    }
  }
  aclError ret = aclmdlDestroyDataset(input);
  if (ret != ACL_ERROR_NONE) {
    ADP_LOG(ERROR) << "Destory acl input dataset failed.";
    return;
  }
  input = nullptr;
}

aclmdlDataset *DatasetFunction::CreateAclOutputDataset(ModelId model_id) {
  aclmdlDesc *model_desc = nullptr;
  model_desc = DatasetFunction::CreateAclModelDesc(model_id);
  DATASET_REQUIRES_RETURN_NULL(model_desc != nullptr, errors::Internal("No model description, create ouput failed."));

  aclmdlDataset* output = aclmdlCreateDataset();
  DATASET_REQUIRES_RETURN_NULL(output != nullptr, errors::Internal("Can't create dataset, create output failed."));

  size_t output_num = aclmdlGetNumOutputs(model_desc);
  for (size_t i = 0; i < output_num; ++i) {
    // create aclDataBuffer with nullptr, aclmdlExecute() will set device memory for it
    aclDataBuffer *outputData = aclCreateDataBuffer(nullptr, 0U);
    DATASET_REQUIRES_RETURN_NULL(outputData != nullptr,
                                 errors::Internal("Can't create data buffer, create output failed."));

    aclError ret = aclmdlAddDatasetBuffer(output, outputData);
    if (ret != ACL_SUCCESS) {
      DestroyAclOutputDataset(output, false);
      ADP_LOG(ERROR) << "Can't add data buffer, create output failed, errorCode is " << ret;
      return nullptr;
    }
    // 当前修改只针对同步接口+动态图，acl会反刷tensordesc; 同步接口+静态图，acl不会反刷tensordesc
    int64_t shape[1] = {-1};
    aclTensorDesc *output_desc = aclCreateTensorDesc(ACL_FLOAT, 1, shape, ACL_FORMAT_NCHW);
    ret = aclmdlSetDatasetTensorDesc(output, output_desc, i);
    if (ret != ACL_SUCCESS) {
      DestroyAclOutputDataset(output, false);
      ADP_LOG(ERROR) << "Add tensor desc failed, errorCode is " << ret;
      return nullptr;
    }
  }

  DestoryAclModelDesc(model_desc);
  return output;
}

void DatasetFunction::DestroyAclOutputDataset(aclmdlDataset *output, bool isFree) {
  if (output == nullptr) {
    return;
  }
  aclError ret = ACL_ERROR_NONE;
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); i++) {
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(output, i);
    if (isFree) {
      void* data_addr = aclGetDataBufferAddr(data_buffer);
      ret = aclrtFree(data_addr);
      if (ret != ACL_ERROR_NONE) {
        ADP_LOG(ERROR) << "Free acl device memory failed.";
      }
    }
    ret = aclDestroyDataBuffer(data_buffer);
    if (ret != ACL_ERROR_NONE) {
      ADP_LOG(ERROR) << "Destory acl output data buffer failed.";
    }
  }
  ret = aclmdlDestroyDataset(output);
  if (ret != ACL_ERROR_NONE) {
    ADP_LOG(ERROR) << "Destory acl output dataset failed.";
    return;
  }
  output = nullptr;
}

aclmdlDesc *DatasetFunction::CreateAclModelDesc(ModelId model_id) {
  aclmdlDesc* model_desc = aclmdlCreateDesc();
  DATASET_REQUIRES_RETURN_NULL(model_desc != nullptr, errors::Internal("Create model description failed."));

  aclError ret = aclmdlGetDesc(model_desc, model_id);
  DATASET_REQUIRES_RETURN_NULL(ret == ACL_SUCCESS, errors::Internal("Get model description failed."));

  return model_desc;
}

void DatasetFunction::DestoryAclModelDesc(aclmdlDesc *model_desc) {
  if (model_desc != nullptr) {
    aclError ret = aclmdlDestroyDesc(model_desc);
    if (ret != ACL_SUCCESS) {
      ADP_LOG(ERROR) << "Destory model desc failed.";
    }
    model_desc = nullptr;
  }
}

Status DatasetFunction::GetAclTenorDescDims(aclTensorDesc *desc, std::vector<int64_t> &ret_dims) {
  size_t dims = aclGetTensorDescNumDims(desc);
  for (size_t i = 0U; i < dims; i++) {
    int64_t dim = aclGetTensorDescDim(desc, i);
    if (dim == -1) {
      return errors::Internal("Get dim from acl tensor failed.");
    }
    ret_dims.emplace_back(dim);
  }
  return Status::OK();
}

DatasetFunction::~DatasetFunction() {
  ADP_LOG(EVENT) << "[DatasetFunction] ~DatasetFunction";
}

void DatasetFunction::DumpTfGraph(const std::string &procPrifex,
    const std::string &func_name, const GraphDef &graph) const {
  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + GetPrefix() + "_" + procPrifex + "_tf_" + func_name + ".pbtxt";
    (void)WriteTextProto(Env::Default(), pbtxt_path, graph);
  }
}

void DatasetFunction::DumpGeComputeGraph(const std::string &procPrifex, const std::string &func_name,
    const ge::ComputeGraphPtr &graph) const {
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

PartialTensorShape DatasetFunction::MakeCompatShape(const PartialTensorShape &a, const PartialTensorShape &b) const {
  const static auto kUnknownRankShape = PartialTensorShape();
  if (a.dims() != b.dims()) {
    return kUnknownRankShape;
  }
  PartialTensorShape shape;
  static constexpr int64 kUnknownDim = -1;
  std::vector<int64> dims;
  for (int i = 0; i < a.dims(); i++) {
    dims.push_back((a.dim_size(i) != b.dim_size(i)) ? kUnknownDim : a.dim_size(i));
  }
  auto status = PartialTensorShape::MakePartialShape(dims.data(), static_cast<int32_t>(dims.size()), &shape);
  return status.ok() ? shape : kUnknownRankShape;
}

void DatasetFunction::UpdateShapeForArgOp(Graph &graph) const {
  std::vector<tensorflow::Node *> args;
  std::vector<absl::optional<PartialTensorShape>> input_shapes;
  for (auto node : graph.op_nodes()) {
    if (!node->IsArg()) {
      continue;
    }
    size_t index = static_cast<size_t>(node->attrs().Find("index")->i());
    if (index >= args.size()) {
      args.resize(index + 1);
    }
    args[index] = node;
  }
  input_shapes.resize(args.size(), absl::nullopt);

  for (size_t i = 0UL; i < input_shape_.size(); i++) {
    auto &shape = input_shapes[i];
    auto &value_shape = input_shape_[i];
    if (!shape.has_value()) {
      shape = value_shape;
      ADP_LOG(INFO) << "Init input " << i << " shape " << shape.value().DebugString();
      args[i]->ClearAttr("_output_shapes");
      args[i]->AddAttr("_output_shapes", std::vector<PartialTensorShape>{shape.value()});
    } else {
      if (shape.value().IsCompatibleWith(value_shape)) {
        continue;
      } else {
        ADP_LOG(INFO) << "Compat input " << i << " shape " << shape.value().DebugString() << " vs. "
                << value_shape.DebugString();
        shape = MakeCompatShape(shape.value(), value_shape);
        ADP_LOG(INFO) << "Refresh input " << i << " shape to " << shape.value().DebugString();
        args[i]->ClearAttr("_output_shapes");
        args[i]->AddAttr("_output_shapes", std::vector<PartialTensorShape>{shape.value()});
      }
    }
  }
}

Status DatasetFunction::BuildSubGraph(FunctionLibraryDefinition &flib_def, tensorflow::Graph &sub_graph,
                                      const std::string &func_name) const {
  const FunctionDef *func_def = flib_def.Find(func_name);
  DATASET_REQUIRES(func_def != nullptr, errors::Internal("Can not find ", func_name));

  // step-1 create and initialize a graph
  Status status = InferShapeUtil::GetSubGraphFromFunctionDef(flib_def, *func_def, &sub_graph);
  DATASET_REQUIRES(status.ok(), status);

  // step-2 Add _output_shapes for arg op.
  // If we do not add "_output_shapes" for arg op, it's shape_inference (tensorflow/core/ops/function_ops.cc)
  // function will set UnknownShape output for this op. A static graph can be misinterpreted as a dynamic graph.
  UpdateShapeForArgOp(sub_graph);

  // step-3 Create Input&Output Desc for each node
  // this assemble process is referenced from TF2.X in npu_parse.cpp file
  AssembleParserAddons(flib_def, sub_graph);

  return Status::OK();
}

std::string DatasetFunction::SerializeGraph(tensorflow::Graph &input_graph) const {
  // convert graph to graphdef
  GraphDef graph_def;
  input_graph.ToGraphDef(&graph_def);

  DumpTfGraph(std::string("DpBuild"), "OriginalTFGraph", graph_def);
  return graph_def.SerializeAsString();
}

Status DatasetFunction::ReadJsonFile(const string &json_file_path, nlohmann::json &json_read) const {
  ADP_LOG(INFO) << "Read "<< json_file_path.c_str() << " json file.";
  Status status = tensorflow::Env::Default()->FileExists(json_file_path);
  DATASET_REQUIRES(status.ok(), status);

  std::ifstream ifs(json_file_path);
  DATASET_REQUIRES(ifs.is_open(), errors::Internal("Open json file failed."));

  try {
    ifs >> json_read;
    ifs.close();
  } catch (const nlohmann::json::exception &e) {
    ifs.close();
    return errors::Internal("Read json file exception : ", e.what());
  }

  ADP_LOG(INFO) << "Read " << json_file_path.c_str() << "json file, content is:" << json_read.dump().c_str();
  return Status::OK();
}

std::string DatasetFunction::GetSocVersion() const {
  const char *soc_name = aclrtGetSocName();
  std::string soc_version = "";
  if (soc_name == nullptr) { return soc_version; }
  soc_version = soc_name;
  if (kDvppAccelerateChips.count(soc_version)) {
    soc_version = kDvppAccelerateChips.at(soc_version);
    transform(soc_version.begin(), soc_version.end(), soc_version.begin(), ::tolower);
    return soc_version;
  }
  return "";
}

Status DatasetFunction::InitAccelateOpList(std::vector<std::string> &acc_while_list) const {
  std::string kOppInstallPath;
  (void)tensorflow::ReadStringFromEnvVar("ASCEND_OPP_PATH", "", &kOppInstallPath);
  std::string dp_split_graph_file = kOppInstallPath + kTfaSplitGraphJsonPath;
  nlohmann::json json_data;
  Status status = ReadJsonFile(dp_split_graph_file, json_data);
  DATASET_REQUIRES(status.ok(), status);

  std::string soc_version = GetSocVersion();
  DATASET_REQUIRES(json_data.contains(soc_version), errors::Internal("Target element does not exists : ", soc_version));
  DATASET_REQUIRES(json_data.contains(kControlOps), errors::Internal("Target element does not exists : ", kControlOps));

  auto parse_op_fun = [&acc_while_list, &json_data](std::string first_key, std::string second_key) -> bool {
    nlohmann::json first_json_data;
    first_json_data = json_data[first_key];
    if (!first_json_data.contains(second_key)) { return false; }
    nlohmann::json second_data = first_json_data[second_key];
    for (auto const& e : second_data) {
        acc_while_list.emplace_back(e);
        ADP_LOG(INFO) << "Add op " << e <<" to acc_while_list.";
    }
    return true;
  };
  // parse preprocess ops
  if (!parse_op_fun(soc_version, kPreProcessOps)) {
    acc_while_list.clear();
    return errors::InvalidArgument("Target element does not exists : ", kPreProcessOps);
  }

#if defined(TF_VERSION_TF2)
    const std::string tf_version = "TF2";
#else
    const std::string tf_version = "TF1";
#endif
  // parse control ops
  if (!parse_op_fun(kControlOps, tf_version)) {
    acc_while_list.clear();
    return errors::InvalidArgument("Target element does not exists : ", tf_version);
  }

  return Status::OK();
}

void DatasetFunction::MarkDvppGraphNodes(Graph &sub_graph_tf, std::vector<Node*> &acc_nodes,
                                         std::vector<Node*> &dvpp_graph_nodes,
                                         const std::vector<std::string> acc_while_list) {
  for (Node *node : sub_graph_tf.nodes()) {
    if (node->IsSource() || node->IsSink()) { continue; }

    auto leave = [&dvpp_graph_nodes](tensorflow::Node *node) {
      if (node->IsSource() || node->IsSink()) { return; }
      if (std::find(dvpp_graph_nodes.begin(), dvpp_graph_nodes.end(), node) == dvpp_graph_nodes.end()) {
        dvpp_graph_nodes.emplace_back(node);
        ADP_LOG(INFO) << "DFSMarkNodes insert node to dvpp_graph_nodes, node name is " << node->name();
      }
    };
    for (std::string item : acc_while_list) {
      if (node->type_string().find(item) != string::npos) {
        acc_nodes.emplace_back(node);
        tensorflow::DFSFrom(sub_graph_tf, {node}, {}, leave, {}, {});
      }
    }
  }
}

void DatasetFunction::MarkConstNodes(const Graph &sub_graph_tf, std::vector<Node*> &dvpp_graph_nodes) const {
  for (Node *node : sub_graph_tf.nodes()) {
    if (node->IsSource() || node->IsSink() || !node->IsConstant()) { continue; }
    if (std::find(dvpp_graph_nodes.begin(), dvpp_graph_nodes.end(), node) != dvpp_graph_nodes.end()) {
      // const node already in dvpp_graph_nodes, Identity -> Const with ControlEdge
      continue;
    }

    bool connect_host = false;
    for (Node *out_node : node->out_nodes()) {
      if (std::find(dvpp_graph_nodes.begin(), dvpp_graph_nodes.end(), out_node) == dvpp_graph_nodes.end()) {
        connect_host = true;
        break;
      }
    }
    // No connection to host graph, only connect to dvpp graph
    if (!connect_host) {
      dvpp_graph_nodes.emplace_back(node);
      ADP_LOG(INFO) << "Const node only connect dvpp graph, node name is " << node->name();
    }
  }
}

void DatasetFunction::MarkHostGraphNodes(const tensorflow::Graph &sub_graph_tf,
                                         const std::vector<Node*> dvpp_graph_nodes,
                                         std::vector<Node*> &host_graph_nodes) const {
  for (Node *node : sub_graph_tf.nodes()) {
    if (node->IsSource() || node->IsSink()) { continue;}

    if (std::find(dvpp_graph_nodes.begin(), dvpp_graph_nodes.end(), node) != dvpp_graph_nodes.end()) {
      continue;
    }
    ADP_LOG(INFO) << "Add node to host graph node list, node name: " << node->name()
      << " , node type:" << node->type_string();
    host_graph_nodes.emplace_back(node);
  }
}

tensorflow::DataType DatasetFunction::EdgeDataType(const tensorflow::Edge &edge) const {
  return edge.src()->output_type(edge.src_output());
}

bool DatasetFunction::CheckCorrectness(const tensorflow::Graph &sub_graph_tf,
                                       const std::vector<Node*> dvpp_graph_nodes,
                                       const std::vector<Node*> host_graph_nodes) const {
  for (Node *node : sub_graph_tf.nodes()) {
    if (node->IsSource() || node->IsSink()) { continue;}
    if ((std::find(dvpp_graph_nodes.begin(), dvpp_graph_nodes.end(), node) == dvpp_graph_nodes.end())
         && (std::find(host_graph_nodes.begin(), host_graph_nodes.end(), node) == host_graph_nodes.end())) {
      ADP_LOG(ERROR) << "Check results failed, node " << node->name() << " is lost.";
      return false;
    }
  }
  return true;
}

void DatasetFunction::UpdateAttrsForArgOp(tensorflow::Node *arg, const tensorflow::Edge *edge) const {
    // add op_def, input_tensor_desc, output_tensor_desc for arg node
    AssembleOpDef(*arg);
    auto output_attr = edge->src()->attrs().Find(kOutputDesc);
    if (output_attr != nullptr) {
      tensorflow::AttrValue input_desc_attrs;
      *input_desc_attrs.mutable_list()->add_func() = output_attr->list().func(edge->src_output());
      arg->AddAttr(kInputDesc, input_desc_attrs);
    }

    auto input_attr = edge->dst()->attrs().Find(kInputDesc);
    if (input_attr != nullptr) {
      tensorflow::AttrValue output_desc_attrs;
      *output_desc_attrs.mutable_list()->add_func() = input_attr->list().func(edge->dst_input());
      arg->AddAttr(kOutputDesc, output_desc_attrs);
    }
}

void DatasetFunction::CreateHostGraph(tensorflow::Graph &sub_graph_host, const std::vector<Node*> host_graph_nodes,
                                      std::vector<Node*> &dvpp_graph_nodes,
                                      std::map<tensorflow::Node*, int64> &dvpp_arg_indexs) const {
  std::map<tensorflow::Node *, tensorflow::Node *> node_map;
  std::vector<const tensorflow::Edge *> output_edges;
  std::vector<tensorflow::DataType> output_types;

  // exist_retvals stors exist retval for calculating new index for other retval node
  std::map<int64, tensorflow::Node*> exist_retvals;

  // Add new node to graph
  for (Node *node : host_graph_nodes) {
    // add independent retval node for graph output
    if (node->IsRetval()) {
      dvpp_graph_nodes.emplace_back(node);
    }
    Status status;
    node_map[node] = sub_graph_host.AddNode(node->def(), &status);
    TF_CHECK_OK(status);
    ADP_LOG(INFO) << "Create host graph, add node to sub_graph_host, node name: " << node_map[node]->name();
  }

  // Add edges to graph
  for (Node *node : host_graph_nodes) {
    // add edges
    for (auto edge : node->in_edges()) {
      if (edge->src()->IsSource()) { continue; }
      if (edge->IsControlEdge()) {
        ADP_LOG(INFO) << "sub_graph_host add control edge from " << node_map[edge->src()]->name()
            << " to " << node_map[node]->name();
        (void)sub_graph_host.AddControlEdge(node_map[edge->src()], node_map[node]);
        continue;
      }
      auto e = sub_graph_host.AddEdge(node_map[edge->src()], edge->src_output(), node_map[node], edge->dst_input());
      ADP_LOG(INFO) << "sub_graph_host add input edge " << e->DebugString() << " for node " << node->name();
    }

    // check if need add a retval node for current edge
    for (auto edge : node->out_edges()) {
      if (edge->dst()->IsSink() || edge->IsControlEdge()) { continue; }
      ADP_LOG(INFO) << "sub_graph_host node " << node->name() << " , node_map[node]=" << node_map[node]
        << " node_map[edge->dst()]=" << node_map[edge->dst()] << "  edge->dst()=" << edge->dst();
      if (edge->dst()->IsRetval()) {
        int64 index = static_cast<int64>(edge->dst()->attrs().Find("index")->i());
        exist_retvals[index] = edge->dst();
        dvpp_arg_indexs[edge->src()] = index;
        ADP_LOG(INFO) << "update dvpp_arg_indexs for node : " << edge->src()->name() << " index=" << index;
      }
      if (node_map[edge->dst()] == nullptr) {
        ADP_LOG(INFO) << "sub_graph_host node "<< node->name() << "  need add RetVal node.";
        output_edges.emplace_back(edge);
        output_types.emplace_back(EdgeDataType(*edge));
      }
    }
  }

  for (size_t i = 0U; i < output_edges.size(); i++) {
    auto &edge = output_edges[i];
    // calculate the "index" attr for current retval node
    int64 retval_index = 0;
    for (auto it = exist_retvals.cbegin(); it != exist_retvals.end(); it++) {
      if (exist_retvals[retval_index] == nullptr) {
          break;
      }
      retval_index++;
    }
    tensorflow::Node *ret;
    TF_CHECK_OK(tensorflow::NodeBuilder("_RetVal" + std::to_string(retval_index), "_Retval")
                      .Input(node_map[edge->src()], edge->src_output())
                      .Attr("index", retval_index)
                      .Attr("T", output_types[i])
                      .Finalize(&sub_graph_host, &ret));
    exist_retvals[retval_index] = ret;
    dvpp_arg_indexs[edge->src()] = retval_index;
    ADP_LOG(INFO) << "sub_graph_host edge src: " << edge->src()->name() << " , edge dst: " << edge->dst()->name()
        << "  retval_index=" << retval_index;
  }
}

void DatasetFunction::CreateDvppGraph(tensorflow::Graph &sub_graph_dvpp, const std::vector<Node*> dvpp_graph_nodes,
                                      const std::map<tensorflow::Node*, int64> dvpp_arg_indexs) const {
  std::map<tensorflow::Node *, tensorflow::Node *> node_map;
  std::vector<const tensorflow::Edge *> input_edges;
  TensorDataTypes input_types;

  for (Node *node : dvpp_graph_nodes) {
    Status status;
    node_map[node] = sub_graph_dvpp.AddNode(node->def(), &status);
    TF_CHECK_OK(status);
    ADP_LOG(INFO) << "Create dvpp graph, add node to sub_graph_dvpp, node name: " << node_map[node]->name()
      << "  originalnode=" << node << ",newnode=" << node_map[node];
  }

  for (Node *node : dvpp_graph_nodes) {
    for (auto edge : node->in_edges()) {
      if (edge->src()->IsSource() || edge->src()->IsSink() || edge->IsControlEdge()) { continue; }
      // record the edge where the arg node needs to be added
      if (node_map[edge->src()] == nullptr && !edge->IsControlEdge()) {
        ADP_LOG(INFO) << "node " << node->name() << "  need add Arg node.";
        input_edges.emplace_back(edge);
        input_types.emplace_back(EdgeDataType(*edge));
        continue;
      }
    }

    // add output edges
    for (auto edge : node->out_edges()) {
      if (edge->dst()->IsSink()) { continue; }
      if (edge->IsControlEdge()) {
        ADP_LOG(INFO) << "sub_graph_dvpp add control edge from " << node_map[node]->name()
            << " to " << node_map[edge->dst()]->name();
        (void)sub_graph_dvpp.AddControlEdge(node_map[node], node_map[edge->dst()]);
        continue;
      }
      auto e = sub_graph_dvpp.AddEdge(node_map[node], edge->src_output(), node_map[edge->dst()], edge->dst_input());
      ADP_LOG(INFO) << "sub_graph_dvpp add out edge " << e->DebugString() << " for node " << node->name();
    }
  }

  for (size_t i = 0U; i < input_edges.size(); i++) {
    auto edge = input_edges[i];
    int64 arg_index = -1;
    if (dvpp_arg_indexs.find(edge->src()) != dvpp_arg_indexs.end()) {
      arg_index = dvpp_arg_indexs.at(edge->src());
    }
    if (arg_index == -1) {
      ADP_LOG(ERROR) << "Sub_graph_dvpp can not found the index for arg node. edge->dst()=" << edge->dst()->name();
      return;
    }
    tensorflow::Node *arg;
    TF_CHECK_OK(tensorflow::NodeBuilder("arg" + std::to_string(arg_index), "_Arg")
                      .Attr("index", arg_index)
                      .Attr("T", input_types[i])
                      .Finalize(&sub_graph_dvpp, &arg));
    auto e = sub_graph_dvpp.AddEdge(arg, 0, node_map[edge->dst()], edge->dst_input());
    ADP_LOG(INFO) << "sub_graph_dvpp add input edge " << e->DebugString() << " with Arg node. arg_index=" << arg_index;

    UpdateAttrsForArgOp(arg, edge);
  }
}

std::string DatasetFunction::SplitSubGraph(FunctionLibraryDefinition &flib_def,
                                           const std::vector<std::string> acc_while_list) {
  // Init original tf sub graph
  tensorflow::Graph sub_graph_tf(&flib_def);
  TF_CHECK_OK(BuildSubGraph(flib_def, sub_graph_tf, funcName_));

  // mark all the nodes with white list in acc_nodes
  // dvpp_graph_nodes stores acc_nodes and nodes which depend on the output of nodes in acc_nodes
  std::vector<Node*> acc_nodes;
  std::vector<Node*> dvpp_graph_nodes;
  MarkDvppGraphNodes(sub_graph_tf, acc_nodes, dvpp_graph_nodes, acc_while_list);
  // handle const node, save the const node which connect host graph and dvpp graph,
  // we will copy this const node for dvpp graph
  MarkConstNodes(sub_graph_tf, dvpp_graph_nodes);
  // mark all the nodes which will add to host graph
  std::vector<Node*> host_graph_nodes;
  MarkHostGraphNodes(sub_graph_tf, dvpp_graph_nodes, host_graph_nodes);
  // Check whether any node is lost
  if (!CheckCorrectness(sub_graph_tf, dvpp_graph_nodes, host_graph_nodes)) { return ""; }
  // create host graph
  tensorflow::Graph sub_graph_host(tensorflow::OpRegistry::Global());
  std::map<tensorflow::Node*, int64> dvpp_arg_indexs;
  CreateHostGraph(sub_graph_host, host_graph_nodes, dvpp_graph_nodes, dvpp_arg_indexs);
  // create dvpp graph
  tensorflow::Graph sub_graph_dvpp(tensorflow::OpRegistry::Global());
  CreateDvppGraph(sub_graph_dvpp, dvpp_graph_nodes, dvpp_arg_indexs);
  // remove attrs for host graph, because these atts only used for GE parser
  for (Node *node : sub_graph_host.nodes()) {
    node->ClearAttr(kInputDesc);
    node->ClearAttr(kOutputDesc);
    node->ClearAttr("op_def");
  }

  // replace the old sub graph with new graph
  tensorflow::FunctionDefLibrary flib;
  TF_CHECK_OK(flib_def.RemoveFunction(funcName_));
  TF_CHECK_OK(tensorflow::GraphToFunctionDef(sub_graph_host, funcName_, flib.add_function()));
  TF_CHECK_OK(flib_def.AddLibrary(flib));

  GraphDef graph_def_tf;
  sub_graph_host.ToGraphDef(&graph_def_tf);
  DumpTfGraph(std::string("DpBuild"), "HostGraph", graph_def_tf);
  GraphDef graph_def_dvpp;
  sub_graph_dvpp.ToGraphDef(&graph_def_dvpp);
  DumpTfGraph(std::string("DpBuild"), "DvppGraph", graph_def_dvpp);

  return graph_def_dvpp.SerializeAsString();
}

Status DatasetFunction::CreateGeGraph(const std::shared_ptr<domi::ModelParser> &model_parser,
    FunctionLibraryDefinition &flib_def) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>(funcName_);
  DATASET_REQUIRES(model_parser != nullptr, errors::Internal("Create compute graph failed."));

  auto build_sub_graph = [this, &flib_def](const std::string &graph_name) -> std::string {
    Graph sub_graph(&flib_def);
    TF_CHECK_OK(this->BuildSubGraph(flib_def, sub_graph, graph_name));
    return this->SerializeGraph(sub_graph);
  };

  std::string graph_def;
  // read file and init op white list who can be accelated by device
  std::vector<std::string> acc_while_list;
  Status status = InitAccelateOpList(acc_while_list);
  if (!status.ok()) {
    ADP_LOG(WARNING) << "Init acc_while_list failed, run whole sub graph. " << status.ToString();
    graph_def = build_sub_graph(funcName_);
  } else {
    ADP_LOG(WARNING) << "Init acc_while_list success, run split graph process.";
    // split graph, and run part graph on device
    graph_def = SplitSubGraph(flib_def, acc_while_list);
    DATASET_REQUIRES(!graph_def.empty(), errors::Internal("Split graph failed."));
    run_split_graph_ = true;
  }

  ge::Status ge_status = model_parser->ParseProtoWithSubgraph(graph_def, build_sub_graph, compute_graph);
  DATASET_REQUIRES(ge_status == ge::SUCCESS, GeError("Parse proto with graph failed.", ge_status));

  // add performance priority tag for each node
  for (const auto &node : compute_graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    DATASET_REQUIRES(op_desc != nullptr, errors::Internal("Param op_desc is nullptr, check invalid."));
    (void)ge::AttrUtils::SetBool(op_desc, "_performance_prior", true);
  }

  DumpGeComputeGraph(std::string("Build"), funcName_, compute_graph);

  ge_graph_ = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  return Status::OK();
}

bool DatasetFunction::IsSplitGraph() {
  return run_split_graph_;
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

Status DatasetFunction::InitGeGraph(FunctionLibraryDefinition &flib_def) {
  std::shared_ptr<domi::ModelParser> model_parser =
        domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  DATASET_REQUIRES(model_parser != nullptr, errors::Unavailable("create model parser ret failed."));

  return CreateGeGraph(model_parser, flib_def);
}

Status DatasetFunction::LoadGeModelFromMem(ModelId &model_id) {
  aclError acl_error = aclmdlLoadFromMem(ge_model_.data.get(), ge_model_.length, &model_id);
  DATASET_REQUIRES(acl_error == ACL_ERROR_NONE, errors::Unavailable("ACL load model from mem failed.", acl_error));
  return Status::OK();
}

Status DatasetFunction::Instantialte() {
  std::shared_ptr<domi::ModelParser> model_parser =
      domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  DATASET_REQUIRES(model_parser != nullptr, errors::Unavailable("create model parser ret failed."));

  std::map<ge::AscendString, ge::AscendString> graph_options;
  // add graph level tag to exclude aicore engine
  graph_options[ge::AscendString("ge.exec.exclude_engines")] = ge::AscendString("AiCore");

  auto ret_build = aclgrphBuildModel(ge_graph_, graph_options, ge_model_);
  DATASET_REQUIRES(ret_build == ge::GRAPH_SUCCESS, errors::Unavailable("Build ge model failed.", ret_build));

  return Status::OK();
}

void DatasetFunction::LogOptions(const std::map<std::string, std::string> &options) {
  for (const auto option : options) {
    ADP_LOG(INFO) << "  name: " << option.first << ", value = " << option.second;
  }
}

Status DatasetFunction::Initialize(const std::map<std::string, std::string> &session_options,
    FunctionLibraryDefinition &flib_def) {
  session_options_ = session_options;

  if (!GePlugin::GetInstance()->IsGlobal()) {
    ADP_LOG(INFO) << "[DatasetFunction] init_options:";
    LogOptions(init_options_);
    GePlugin::GetInstance()->Init(init_options_);
  }

  std::shared_ptr<domi::ModelParser> model_parser =
        domi::ModelParserFactory::Instance()->CreateModelParser(domi::FrameworkType::TENSORFLOW);
  DATASET_REQUIRES(model_parser != nullptr, errors::Unavailable("create model parser ret failed."));

  std::transform(output_types_.begin(), output_types_.end(), std::back_inserter(ge_output_types_),
      [&model_parser](const DataType type) { return model_parser->ConvertToGeDataType(static_cast<uint32_t>(type)); });

  LogOptions(session_options);
  ge_session_.reset(new (std::nothrow)ge::Session(session_options));

  Status status = InitGeGraph(flib_def);
  DATASET_REQUIRES(status.ok(), status);
  return Instantialte();
}

Status DatasetFunction::Run(ModelId model_id, aclmdlDataset *in_dataset, aclmdlDataset *out_dataset) const {
  aclError ret = aclmdlExecute(model_id, in_dataset, out_dataset);
  return (ret == ACL_SUCCESS) ? Status::OK() : errors::Internal("Run graph failed with model_id=", model_id,
      " ret=", ret);
}

Status DatasetFunction::RunWithStreamAsyn(ModelId model_id, aclmdlDataset *in_dataset,
    aclmdlDataset *out_dataset, aclrtStream stream) const {
  aclError ret = aclmdlExecuteAsync(model_id, in_dataset, out_dataset, stream);
  return (ret == ACL_SUCCESS) ? Status::OK() : errors::Internal("Run graph with stream failed with model_id=",
      model_id, " ret=", ret);
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

TimeStatistic::TimeStatistic(int64_t total_threads) {
  stop_record = false;
  statis_threads.resize(total_threads);
  statis_threads_ge.resize(total_threads);
  max_threads = total_threads;
}

void TimeStatistic::RecordStartTime(Items &it) const {
  it.start_time = InferShapeUtil::GetCurrentTimestap();
}

void TimeStatistic::RecordEndTime(Items &it) const {
  it.end_time = InferShapeUtil::GetCurrentTimestap();
  it.Update();
}

void TimeStatistic::UpdateWithTimeTag(Items &it, std::shared_ptr<Items> &tag) const {
  // update it with tag
  RecordEndTime(*tag);
  int64_t interval_time = tag->min_time;
  // if data overflow, stop record
  if (DatasetFunction::CheckAddOverflow(it.total_time, interval_time)) {
    return;
  }
  it.total_time += interval_time;
  it.total_records++;
  it.min_time = std::min(it.min_time, interval_time);
  it.max_time = std::max(it.max_time, interval_time);
}

void TimeStatistic::ShowTimeStatistic() {
  if (stop_record || statis.total_records <= 0LL) {
    return;
  }

  int64_t kMicrosToMillis = 1LL;
  statis.avg_time = statis.total_time / statis.total_records;
  ADP_LOG(INFO) << "[TimeStatistic] Time statistics for GetNext (avg min max(unit:us)) : "
                << statis.avg_time / kMicrosToMillis
                << " " << statis.min_time / kMicrosToMillis
                << " " << statis.max_time / kMicrosToMillis;

  auto print_thread_info = [this, kMicrosToMillis](std::vector<Items> &stat, const std::string name) {
    for (int64_t i = 0; i < this->max_threads; i++) {
      if (stat[i].total_records <= 0LL) {
        stat[i].avg_time = 0LL;
        continue;
      }
      stat[i].avg_time = stat[i].total_time / stat[i].total_records;
      ADP_LOG(INFO) << "[TimeStatistic] Time statistics for " << name << " with Thread-" << i
                    << " (avg min max(unit:us)) : "
                    << stat[i].avg_time / kMicrosToMillis
                    << " " << stat[i].min_time / kMicrosToMillis
                    << " " << stat[i].max_time / kMicrosToMillis;
    }
  };
  print_thread_info(statis_threads, "MapFunc");
  print_thread_info(statis_threads_ge, "GE_Func");
}
}  // namespace data
}  // namespace tensorflow