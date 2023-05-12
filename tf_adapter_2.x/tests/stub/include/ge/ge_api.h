/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef INC_EXTERNAL_GE_GE_API_H_
#define INC_EXTERNAL_GE_GE_API_H_

#include "stub/defines.h"

#include <map>
#include <string>
#include <vector>
#include <cstring>

#include "tensorflow/core/graph/graph.h"

namespace ge {
Status GEInitialize(const std::map<std::string, std::string> &options);

Status GEFinalize();

std::string GEGetErrorMsg();

class Shape {
 public:
  Shape() = default;
  ~Shape() = default;
  explicit Shape(const std::vector<int64_t> &dims) : dims_(dims) {}
  std::vector<int64_t> GetDims() const { return dims_; }

 private:
  std::vector<int64_t> dims_;
};

class TensorDesc {
 public:
  TensorDesc() = default;
  ~TensorDesc() = default;
  explicit TensorDesc(Shape shape, Format format = FORMAT_ND, DataType type = DT_FLOAT)
      : shape_(shape), format_(format), type_(type) {}
  Shape GetShape() const { return shape_; }
  void SetShape(const Shape &shape) { shape_ = shape; }
  DataType GetDataType() const { return type_; }

 private:
  DataType type_ = DT_FLOAT;
  Format format_ = FORMAT_ND;
  Shape shape_;
};

class Tensor {
  using DeleteFunc = std::function<void(uint8_t *)>;

 public:
  Tensor() = default;
  ~Tensor() = default;
  Tensor(const Tensor &other) {
    const static DeleteFunc deleter = [](uint8_t *p) { delete[] p; };
    desc_ = other.desc_;
    size_ = other.size_;
    data_ = std::unique_ptr<uint8_t[], DeleteFunc>(new uint8_t[size_], deleter);
    std::memcpy(data_.get(), other.GetData(), size_);
  }
  Tensor &operator=(const Tensor &) = default;

  TensorDesc GetTensorDesc() const { return desc_; }
  graphStatus SetTensorDesc(const TensorDesc &desc) {
    desc_ = desc;
    return GRAPH_SUCCESS;
  }

  const uint8_t *GetData() const { return data_.get(); }
  size_t GetSize() const { return size_; }
  std::unique_ptr<uint8_t[], DeleteFunc> ResetData() { return std::move(data_); }
  graphStatus SetData(const uint8_t *data, size_t size) {
    const static DeleteFunc deleter = [](uint8_t *p) { delete[] p; };
    data_ = std::unique_ptr<uint8_t[], DeleteFunc>(new uint8_t[size], deleter);
    std::memcpy(data_.get(), data, size);
    size_ = size;
    return GRAPH_SUCCESS;
  }

 private:
  TensorDesc desc_;
  std::unique_ptr<uint8_t[], DeleteFunc> data_;
  size_t size_;
};

class ComputeGraph {
 public:
  explicit ComputeGraph(const std::string &name) {}
  ~ComputeGraph() = default;
  std::shared_ptr<tensorflow::Graph> graph;
  size_t GetAllNodesSize() const;
  size_t GetInputSize() const;
  size_t GetOutputSize() const;
};

class Graph {
 public:
  Graph() = default;
  ~Graph() = default;
  void SetNeedIteration(bool v) { need_iteration = v; }
  bool need_iteration = false;
  std::shared_ptr<tensorflow::Graph> graph;
};

class Session {
  using RunAsyncCallback = std::function<void(Status, std::vector<ge::Tensor> &)>;

 public:
  explicit Session(const std::map<std::string, std::string> &options) {}

  ~Session() = default;

  Status AddGraph(uint32_t graphId, const Graph &graph);

  Status AddGraph(uint32_t graphId, const Graph &graph, const std::map<std::string, std::string> &options);

  Status RemoveGraph(uint32_t graphId);

  Status RunGraphAsync(uint32_t graphId, const std::vector<ge::Tensor> &inputs, RunAsyncCallback callback);

  bool IsGraphNeedRebuild(uint32_t graphId);

 private:
  std::map<uint32_t, std::shared_ptr<tensorflow::Graph>> graphs_;
  std::map<uint32_t, bool> graph_need_rebuild_;
};

struct GraphUtilsEx {
  static Graph CreateGraphFromComputeGraph(const ge::ComputeGraphPtr compute_graph);
  static ge::ComputeGraphPtr GetComputeGraph(const ge::Graph &compute_graph);
};  // namespace GraphUtilsEx

}  // namespace ge

#endif  // INC_EXTERNAL_GE_GE_API_H_