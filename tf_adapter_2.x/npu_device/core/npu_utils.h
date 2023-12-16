/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef NPU_DEVICE_CORE_NPU_UTILS_H
#define NPU_DEVICE_CORE_NPU_UTILS_H

#include <unordered_set>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/tensor_description.pb.h"

#include "acl/acl_base.h"
#include "graph/types.h"
#include "graph/ascend_string.h"

#include "npu_types.h"

namespace npu {
class NpuDevice;
class ScopeTensorHandleDeleter {
 public:
  ScopeTensorHandleDeleter() = default;
  ~ScopeTensorHandleDeleter();
  void Guard(TFE_TensorHandle *handle);

 private:
  std::unordered_set<TFE_TensorHandle *> handles_;
};

/**
 * @brief: map ge type to tf
 * @param ge_type: ge type
 * @param acl_type: tf type
 */
tensorflow::Status MapGeType2Tf(ge::DataType ge_type, tensorflow::DataType &tf_type);

tensorflow::Status SeparateGraphDef(tensorflow::GraphDef *def,
                                    std::vector<ge::AscendString> &partition_graph,
                                    std::map<ge::AscendString, ge::AscendString> &const_value_map);

/**
 * @brief: map tf type to ge
 * @param ge_type: tf type
 * @param acl_type: ge type
 */
tensorflow::Status MapTfType2Ge(tensorflow::DataType tf_type, ge::DataType &ge_type);

/**
 * @brief: map ge type to acl
 * @param ge_type: ge type
 * @param acl_type: acl type
 */
tensorflow::Status MapGeType2Acl(ge::DataType ge_type, aclDataType &acl_type);

/**
 * @brief: map ge format to acl
 * @param ge_format: ge format
 * @param acl_format: acl format
 */
tensorflow::Status MapGeFormat2Acl(ge::Format ge_format, aclFormat &acl_format);

std::string WrapResourceName(const std::string &name);

// specify the template in utils.cpp if need
template <typename T>
std::string ToString(T v) {
  return std::to_string(v);
}

template <typename T>
std::string VecToString(std::vector<T> vec) {
  if (vec.empty()) {
    return "[]";
  }
  std::string s = "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    s += ToString(vec[i]);
    if (i != vec.size() - 1) {
      s += ",";
    }
  }
  return s + "]";
}

std::map<ge::AscendString, ge::AscendString> StringToAscendString(
    const std::map<std::string, std::string> &string_map);

std::string SetToString(const std::set<std::string> &vec);

struct ResourceCompare {
  bool operator()(const tensorflow::ResourceHandle &left, const tensorflow::ResourceHandle &right) const {
    if (left.name() != right.name()) {
      return left.name() < right.name();
    } else if (left.container() != right.container()) {
      return left.container() < right.container();
    } else {
      return left.device() < right.device();
    }
  }
};

void PruneGraphByFunctionSignature(const tensorflow::FunctionDef &fdef, tensorflow::Graph &g,
                                   bool keep_signature = false);

void FixGraphArgRetvalIndex(const tensorflow::Graph &graph);

bool IsSubstituteNode(const tensorflow::Node &node);

bool IsSubstituteNode(const tensorflow::NodeDef &def);

bool IsNodeHasSubgraph(const tensorflow::Node &node);

bool IsNodeHasSubstituteInput(const tensorflow::Node &node);

tensorflow::DataType EdgeDataType(const tensorflow::Edge &edge);

std::set<std::string> GetNodeSubgraph(const tensorflow::Node &node);
tensorflow::Status GetSubgraphUnsupportedOps(const NpuDevice &device, const tensorflow::Node &node,
                                             const tensorflow::FunctionLibraryDefinition &lib_def,
                                             std::set<std::string> &unsupported_ops);
tensorflow::Status GetGraphUnsupportedOps(const NpuDevice &device, const tensorflow::Graph &graph,
                                          const tensorflow::FunctionLibraryDefinition &lib_def,
                                          std::set<std::string> &unsupported_ops);

bool IsGraphHasAnyUnknownShapeNode(const tensorflow::Graph &graph,
                                   const tensorflow::FunctionLibraryDefinition &lib_def);

bool IsGraphNeedLoop(const tensorflow::Graph &graph, tensorflow::Node *&key);

uint64_t NextUUID();

class OptimizeStageGraphDumper {
 public:
  explicit OptimizeStageGraphDumper(const std::string &graph);

  void Dump(const std::string &stage, const tensorflow::GraphDef &graph_def);

  void DumpWithSubGraphs(const std::string &stage, const tensorflow::GraphDef &graph_def,
                         const tensorflow::FunctionLibraryDefinition &lib_def);

 private:
  bool enabled_{false};
  std::string graph_;
  int counter_{0};
};

void NpuCustomizedOptimizeGraph(tensorflow::FunctionLibraryRuntime &lib, std::unique_ptr<tensorflow::Graph> *g);

tensorflow::Status LoopCopy(char *dst_ptr, size_t dst_size, char *src_ptr, size_t src_size);

int64_t CreateChannelCapacity(const npu::TensorPartialShapes &shapes, const npu::TensorDataTypes &types);

class NpuAllocatorUtils {
public:
  NpuAllocatorUtils() = default;
  ~NpuAllocatorUtils();

  static bool IsNpuAllocator(const std::string name) {
    // 这里判断Npu Cpu内存的原因和作用是:
    // map和mapandbatch算子内部组织的输出数据是连续的, 不需要外部再对输出数据做连续内存的重新组织
    return (name.compare(kNpuAllocatorName) == 0) ||
        (name.compare(kCpuAllocatorName) == 0);
  }

  static bool IsNpuAllocator(const tensorflow::Tensor &tensor) {
    tensorflow::TensorDescription tensorDesc;
    tensor.FillDescription(&tensorDesc);
    if (tensorDesc.has_allocation_description()) {
      return IsNpuAllocator(tensorDesc.allocation_description().allocator_name());
    }
    return false;
  }

private:
  static constexpr const char* const kNpuAllocatorName = "NpuAllocator";
  static constexpr const char* const kCpuAllocatorName = "CpuAllocator";
};
}  // namespace npu

#endif  // NPU_DEVICE_CORE_NPU_UTILS_H
