/* Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "ge/ge_api.h"
#include "framework/omg/parser/parser_api.h"
#include "framework/omg/parser/model_parser.h"

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"

namespace domi {
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
  {tensorflow::DataType::DT_BFLOAT16, ge::DataType::DT_BF16},
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

ge::DataType ModelParser::ConvertToGeDataType(const uint32_t type) {
  auto search = data_type_map.find(type);
  if (search != data_type_map.end()) {
    return search->second;
  } else {
    return ge::DataType::DT_UNDEFINED;
  }
}

Status ModelParser::ParseProtoWithSubgraph(const std::string &serialized_proto, GetGraphCallbackV2 callback,
                                           ge::ComputeGraphPtr &graph) {
  std::vector<std::string> partitioned_serialized{serialized_proto};
  std::map<std::string, std::string> const_value_map;
  return ParseProtoWithSubgraph(partitioned_serialized, const_value_map, callback, graph);
}

Status ModelParser::ParseProtoWithSubgraph(const std::vector<std::string> &partitioned_serialized,
                                           const std::map<std::string, std::string> &const_value_map,
                                           GetGraphCallbackV2 callback,
                                           ge::ComputeGraphPtr &graph) {
  std::string graph_def_str = partitioned_serialized[0];
  tensorflow::GraphDef graph_def;
  graph_def.ParseFromString(graph_def_str);

  if (!const_value_map.empty()) {
    for (auto &node : *graph_def.mutable_node()) {
      if (node.op() == "Const") {
        std::string node_name = node.name();
        auto iter = const_value_map.find(node_name);
        if (iter == const_value_map.end()) {
          return ge::FAILED;
        }
        auto attr_iter = node.mutable_attr()->find("value");
        if (attr_iter == node.mutable_attr()->end()) {
          return ge::FAILED;
        }
        tensorflow::TensorProto *tensor = attr_iter->second.mutable_tensor();
        tensor->set_tensor_content(iter->second);
      }
    }
  }
  graph->graph = std::make_shared<tensorflow::Graph>(tensorflow::OpRegistry::Global());
  tensorflow::GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  tensorflow::ConvertGraphDefToGraph(opts, graph_def, graph->graph.get());
  for (const auto &node : graph->graph->op_nodes()) {
    for (const auto &attr : node->attrs()) {
      if (attr.second.has_func()) {
        callback(attr.second.func().name());
      }
    }
  }
  return ge::SUCCESS;
}
}  // namespace domi

namespace ge {
Status ParserInitialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

Status ParserFinalize() { return SUCCESS; }
}  // namespace ge