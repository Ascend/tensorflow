#include "stub/defines.h"

#include <map>
#include <string>
#include <vector>
#include <cstring>

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
  DataType type_;
  Format format_;
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
};

class Graph {
 public:
  Graph() = default;
  ~Graph() = default;
  void SetNeedIteration(bool need_iteration) {}
};

class Session {
  using RunAsyncCallback = std::function<void(Status, std::vector<ge::Tensor> &)>;

 public:
  explicit Session(const std::map<std::string, std::string> &options) {}

  ~Session() = default;

  Status AddGraph(uint32_t graphId, const Graph &graph) { return SUCCESS; }

  Status RemoveGraph(uint32_t graphId) { return SUCCESS; }

  Status RunGraphAsync(uint32_t graphId, const std::vector<ge::Tensor> &inputs, RunAsyncCallback callback) {
    return SUCCESS;
  }

  bool IsGraphNeedRebuild(uint32_t graphId) { return false; }
};

struct GraphUtils {
  static Graph CreateGraphFromComputeGraph(const ge::ComputeGraphPtr compute_graph) { return {}; }
};  // namespace GraphUtils

}  // namespace ge