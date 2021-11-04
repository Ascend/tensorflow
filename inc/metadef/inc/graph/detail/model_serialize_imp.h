/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef INC_GRAPH_DETAIL_MODEL_SERIALIZE_IMP_H_
#define INC_GRAPH_DETAIL_MODEL_SERIALIZE_IMP_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "graph/anchor.h"
#include "graph/detail/attributes_holder.h"
#include "graph/ge_tensor.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/model.h"

namespace ge {
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;

struct NodeNameGraphReq {
    std::string node_name;
    int32_t index;
    ComputeGraphPtr graph;
};

struct NodeNameNodeReq {
    std::string src_node_name;
    int32_t src_out_index;
    NodePtr dst_node;
    int32_t dst_in_index;
    std::string dst_node_name;
};

class ModelSerializeImp {
 public:
  bool SerializeModel(const Model &model, proto::ModelDef *modeProto, bool is_dump = false);

  bool SerializeGraph(const ConstComputeGraphPtr &graph, proto::GraphDef *graphProto, bool is_dump = false);

  bool SerializeEdge(const NodePtr &node, proto::OpDef *opDefProto);

  bool SerializeOpDesc(const ConstOpDescPtr &node, proto::OpDef *opDefProto, bool is_dump = false);

  bool SerializeNode(const NodePtr &node, proto::OpDef *opDefProto, bool is_dump = false);

  bool SerializeTensor(const ConstGeTensorPtr &tensor, proto::TensorDef *tensorProto);

  bool UnserializeModel(Model &model, proto::ModelDef &modeProto);

  bool UnserializeGraphWithoutEdge(ComputeGraphPtr &graph, proto::GraphDef &graphProto);

  bool UnserializeGraph(ComputeGraphPtr &graph, proto::GraphDef &graphProto);

  bool HandleNodeNameRef();

  bool UnserializeOpDesc(OpDescPtr &opDesc, proto::OpDef &opDefProto);
  void AttrDefToOpDesc(OpDescPtr &op_desc, std::vector<std::string> &key_in, std::vector<std::string> &key_out,
                       std::vector<uint32_t> &value_in, std::vector<uint32_t> &value_out, std::vector<std::string> &opt);
  void OpDescToAttrDef(const ConstOpDescPtr &op_desc, proto::OpDef *op_def_proto, bool is_dump = false);

  bool UnserializeNode(ComputeGraphPtr &graph, proto::OpDef &opDefProto);

  bool UnserializeTensor(GeTensorPtr &tensor, proto::TensorDef &tensorProto);

  bool ParseNodeIndex(const std::string &node_index, std::string &nodeName, int32_t &index);

  void SetProtobufOwner(const ProtoMsgOwner &bufferProtobufOnwer) { protobuf_owner_ = bufferProtobufOnwer; }

  static bool SerializeAllAttrsFromAnyMap(
      const std::map<std::string, AnyValue> &, google::protobuf::Map<std::string, ::ge::proto::AttrDef> *);
  static bool DeserializeAllAttrsToAttrHolder(
      const google::protobuf::Map<std::string, ::ge::proto::AttrDef> &, AttrHolder *);

 private:
  bool RebuildOwnership(ComputeGraphPtr &compute_graph, std::map<std::string, ComputeGraphPtr> &subgraphs);

  void FixOpDefSubgraphInstanceName(const ConstOpDescPtr &op_desc);

  void ExtractMetaDataAttr(proto::OpDef &op_def_proto, std::vector<std::string> &opt_input,
                           std::vector<std::string> &key_in, std::vector<uint32_t> &value_in,
                           std::vector<std::string> &key_out, std::vector<uint32_t> &value_out) const;

  std::vector<NodeNameGraphReq> graph_input_node_names_;
  std::vector<NodeNameGraphReq> graph_output_node_names_;
  std::vector<NodeNameNodeReq> node_input_node_names_;
  std::map<std::string, NodePtr> node_map_;
  ProtoMsgOwner protobuf_owner_;
};
}  // namespace ge

#endif  // INC_GRAPH_DETAIL_MODEL_SERIALIZE_IMP_H_
