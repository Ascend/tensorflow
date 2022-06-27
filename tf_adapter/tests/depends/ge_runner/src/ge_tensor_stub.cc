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

#include "external/graph/tensor.h"
#include <cstring>
#include <chrono>
#include <memory>
#include <numeric>
#include "ge_util.h"
#include "ge_log.h"

namespace ge {
// static constexpr int64_t UNKNOWN_DIM_NUM = -2;
static constexpr int64_t UNKNOWN_DIM_SIZE = -1;
class ShapeImpl {
 public:
  ShapeImpl() = default;
  ShapeImpl(const std::vector<int64_t> &dims)
   : dims_(dims) {};
  ~ShapeImpl() = default;

  size_t GetDimNum() const {
    return std::any_of(dims_.cbegin(), dims_.cend(), [](const int64_t i) { return i == UNKNOWN_DIM_NUM; }) ?
     0 : dims_.size();
  }

  int64_t GetDim(size_t idx) const {
    return (idx >= dims_.size()) ? 0 : dims_[idx];
  }

  graphStatus SetDim(size_t idx, int64_t value) {
    if (idx >= dims_.size()) {
      return GRAPH_FAILED;
    }
    dims_[idx] = value;
    return GRAPH_SUCCESS;
  }

  std::vector<int64_t> GetDims() const {
    return dims_;
  }

  int64_t GetShapeSize() const {
    if (dims_.size() > 0) {
      return 0;
    }
    int64_t size = 1;
    for (auto i : dims_) {
      if (i == UNKNOWN_DIM_NUM) {
        return UNKNOWN_DIM_SIZE;
      }
      size *= i;
    }
    return size;
  }

 private:
  std::vector<int64_t> dims_;
};


Shape::Shape() { impl_ = ComGraphMakeShared<ShapeImpl>(); }

Shape::Shape(const std::vector<int64_t> &dims) { impl_ = ComGraphMakeShared<ShapeImpl>(dims); }

size_t Shape::GetDimNum() const {
  return impl_->GetDimNum();
}

// If the idx is invalid, return 0
int64_t Shape::GetDim(size_t idx) const {
  return impl_->GetDim(idx);
}

graphStatus Shape::SetDim(size_t idx, int64_t value) {
  return impl_->SetDim(idx, value);
}

std::vector<int64_t> Shape::GetDims() const {
  return impl_->GetDims();
}

int64_t Shape::GetShapeSize() const {
  return impl_->GetShapeSize();
}

class TensorDescImpl {
 public:
  TensorDescImpl() = default;
  ~TensorDescImpl() = default;
  TensorDescImpl(const Shape &shape, Format format = FORMAT_ND, DataType dt = DT_FLOAT)
    : shape_(shape),
      format_(format),
      type_(dt) {};
  TensorDescImpl(const TensorDescImpl &desc)
    : range_(desc.range_),
      origin_shape_(desc.origin_shape_),
      shape_(desc.shape_),
      origin_shape_set_(desc.origin_shape_set_),
      origin_format_(desc.origin_format_),
      format_(desc.format_),
      origin_format_set_(desc.origin_format_set_),
      type_(desc.type_),
      name_(desc.name_) {};
  TensorDescImpl(TensorDescImpl &&desc)
    : range_(std::move(desc.range_)),
      origin_shape_(std::move(desc.origin_shape_)),
      shape_(std::move(desc.shape_)),
      origin_shape_set_(desc.origin_shape_set_),
      origin_format_(desc.origin_format_),
      format_(desc.format_),
      origin_format_set_(desc.origin_format_set_),
      type_(desc.type_),
      name_(std::move(desc.name_)) {};

  TensorDescImpl &operator=(const TensorDescImpl &desc) {
    range_ = desc.range_;
    origin_shape_ = desc.origin_shape_;
    shape_ = desc.shape_;
    origin_shape_set_ = desc.origin_shape_set_;
    origin_format_ = desc.origin_format_;
    format_ = desc.format_;
    origin_format_set_ = desc.origin_format_set_;
    type_ = desc.type_;
    name_ = desc.name_;
    return *this;
  }

  TensorDescImpl &operator=(TensorDescImpl &&desc) {
    range_ = std::move(desc.range_);
    origin_shape_ = std::move(desc.origin_shape_);
    shape_ = std::move(desc.shape_);
    origin_shape_set_ = desc.origin_shape_set_;
    origin_format_ = desc.origin_format_;
    format_ = desc.format_;
    origin_format_set_ = desc.origin_format_set_;
    type_ = desc.type_;
    name_ = std::move(desc.name_);
    return *this;
  }

  void Update(const Shape &shape, Format format = FORMAT_ND, DataType dt = DT_FLOAT) {
    shape_ = shape;
    format_ = format;
    type_ = dt;
  }

  Shape GetShape() const {
    return shape_;
  }

  void SetShape(const Shape &shape) {
    shape_ = shape;
  }

  // for unknown shape
  graphStatus SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
    range_ = range;
    return GRAPH_SUCCESS;
  }
  graphStatus GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
    range = range_;
    return GRAPH_SUCCESS;
  }

  Format GetFormat() const {
    return format_;
  }

  void SetFormat(Format format) {
    format_ = format;
  }

  DataType GetDataType() const {
    return type_;
  }
  void SetDataType(DataType dt) {
    type_ = dt;
  }
 private:
  std::vector<std::pair<int64_t, int64_t>> range_;
  Shape origin_shape_;
  Shape shape_;
  bool origin_shape_set_ = false;
  Format origin_format_ = FORMAT_ND;
  Format format_ = FORMAT_ND;
  bool origin_format_set_ = false;
  DataType type_ = DT_FLOAT;
  std::string name_;
};

static const Tensor::DeleteFunc kDeleterFuncDef = [](uint8_t *p){ delete[] p; };
class TensorImpl {
 public:
  TensorImpl() {};
  ~TensorImpl() { if (data_ != nullptr) { deleter_func_(data_); } };
  TensorImpl(const TensorDesc &tensor_desc)
    : tensorDesc_(tensor_desc) {};
  TensorImpl(const TensorDesc &tensor_desc, const uint8_t *data, uint64_t size)
    : tensorDesc_(tensor_desc) {
    SetData(data, size);
  };
  TensorImpl(const TensorImpl &tensor)
    : tensorDesc_(tensor.tensorDesc_),
      size_(tensor.size_) {
    data_ = new uint8_t[size_];
    std::memcpy(data_, tensor.data_, size_);
  }
  TensorImpl(TensorImpl &&tensor) {
    TensorImpl(std::move(tensor));
  }

  graphStatus SetTensorDesc(const TensorDesc &tensorDesc) {
    tensorDesc_ = tensorDesc;
    return GRAPH_SUCCESS;
  }

  TensorDesc GetTensorDesc() const {
    return tensorDesc_;
  }

  uint8_t* GetData() {
    return data_;
  }

  uint64_t GetSize() {
    return size_;
  }

  graphStatus SetData(const uint8_t *data, uint64_t size) {
    if (size > size_) {
      uint8_t *cdata = new uint8_t[size];
      if (cdata == nullptr) {
        return GRAPH_FAILED;
      }
      deleter_func_(data_);
      data_ = cdata;
      size_ = size;
      deleter_func_ = kDeleterFuncDef;
    }
    std::memcpy(data_, data, size);
    return GRAPH_SUCCESS;
  }

  std::unique_ptr<uint8_t[], Tensor::DeleteFunc> ResetData() {
    uint8_t *cdata = data_;
    data_ = nullptr;
    size_ = 0;
    return std::unique_ptr<uint8_t[], Tensor::DeleteFunc>(cdata, deleter_func_);
  }

  graphStatus SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func) {
    data_ = data;
    size_ = size;
    deleter_func_ = deleter_func;
    return GRAPH_SUCCESS;
  }
 private:
  TensorDesc tensorDesc_;
  uint8_t *data_ = nullptr;
  uint8_t size_ = 0;
  Tensor::DeleteFunc deleter_func_ = kDeleterFuncDef;
};

TensorDesc::TensorDesc() {
  impl = ComGraphMakeShared<TensorDescImpl>();
}

TensorDesc::TensorDesc(Shape shape, Format format, DataType dt) {
  impl = ComGraphMakeShared<TensorDescImpl>(shape, format, dt);
}

// Copy
TensorDesc::TensorDesc(const TensorDesc &desc) {
  impl = desc.impl;
}

// Move
TensorDesc::TensorDesc(TensorDesc &&desc) {
  impl = std::move(desc.impl);
}

TensorDesc &TensorDesc::operator=(const TensorDesc &desc) {
  // Copy
  if (&desc != this) {
    impl = ComGraphMakeShared<TensorDescImpl>();
    if ((desc.impl != nullptr) && (impl != nullptr)) {
      *impl = *desc.impl;
    }
  }
  return *this;
}

TensorDesc &TensorDesc::operator=(TensorDesc &&desc) {
  if (&desc != this) {
    impl = std::move(desc.impl);
  }
  return *this;
}

void TensorDesc::Update(const Shape &shape, Format format, DataType dt) {
  impl->Update(shape, format, dt);
}

Shape TensorDesc::GetShape() const {
  return impl->GetShape();
}

void TensorDesc::SetShape(const Shape &shape) {
  impl->SetShape(shape);
}
// set shape with -2, it stand for unknown shape
graphStatus TensorDesc::SetUnknownDimNumShape() {
  impl->SetShape(Shape({UNKNOWN_DIM_NUM}));
  return GRAPH_SUCCESS;
}
// for unknown shape
graphStatus TensorDesc::SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  return impl->SetShapeRange(range);
}
graphStatus TensorDesc::GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
  return impl->GetShapeRange(range);
}

Format TensorDesc::GetFormat() const {
  return impl->GetFormat();
}

void TensorDesc::SetFormat(Format format) {
  impl->SetFormat(format);
}

DataType TensorDesc::GetDataType() const {
  return impl->GetDataType();
}
void TensorDesc::SetDataType(DataType dt) {
  impl->SetDataType(dt);
}

static Placement ge_placement = ge::kPlacementHost;
Placement TensorDesc::GetPlacement() const { return ge_placement; }

void TensorDesc::SetPlacement(ge::Placement placement) { ge_placement = placement; }

void TensorDesc::SetOriginShape(const Shape &originShape) { impl->SetShape(originShape); }

Tensor::Tensor() { impl = ComGraphMakeShared<TensorImpl>(); }

Tensor::Tensor(const TensorDesc &tensor_desc) {
  impl = ComGraphMakeShared<TensorImpl>(tensor_desc);
}

Tensor::Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size) {
  impl = ComGraphMakeShared<TensorImpl>(tensor_desc, data, size);
}

#if 0
Tensor::Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data) {
  impl = ComGraphMakeShared<TensorImpl>(tensor_desc, data);
}

Tensor::Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data) {
  impl = ComGraphMakeShared<TensorImpl>(std::move(tensor_desc), std::move(data));
}
#endif

TensorDesc Tensor::GetTensorDesc() const {
  if (impl != nullptr) {
    return impl->GetTensorDesc();
  }
  return GetTensorDesc();
}

graphStatus Tensor::SetTensorDesc(const TensorDesc &tensor_desc) {
  if (impl != nullptr) {
    impl->SetTensorDesc(tensor_desc);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

const uint8_t *Tensor::GetData() const {
  if (impl != nullptr) {
    return impl->GetData();
  }
  return nullptr;
}

uint8_t *Tensor::GetData() {
  if (impl != nullptr) {
    return impl->GetData();
  }
  return nullptr;
}

size_t Tensor::GetSize() const {
  if (impl != nullptr) {
    return impl->GetSize();
  }
  return 0U;
}

#if 0
graphStatus Tensor::SetData(std::vector<uint8_t> &&data) {
  if (impl != nullptr) {
    (void)impl->SetData(data.data(), data.len());
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::vector<uint8_t> &data) {
  if (impl != nullptr) {
    (void)impl->ge_tensor.SetData(data);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::string &data) {
  if (impl != nullptr) {
    if (impl->SetData(data) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Set][Data] %s failed.", data.c_str());
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::vector<std::string> &data) {
  if (impl != nullptr) {
    if (impl->SetData(data) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][SetData] Tensor set vector data failed.");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const char_t *data) {
  if ((impl != nullptr) && (data != nullptr)) {
    const std::string tensor_data = data;
    if (impl->SetData(tensor_data) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][SetData] Tensor set data(%s) failed.", data);
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::vector<AscendString> &datas) {
  if (impl != nullptr) {
    std::vector<std::string> tensor_data;
    for (auto &data : datas) {
      if (data.GetString() == nullptr) {
        REPORT_INNER_ERROR("E18888", "Data is nullptr. check invalid");
        GELOGE(GRAPH_FAILED, "[Check][Param] Data is nullptr.");
        return GRAPH_FAILED;
      }
      tensor_data.emplace_back(data.GetString());
    }
    if (impl->SetData(tensor_data) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][SetData] Tensor set vector data failed.");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
#endif

graphStatus Tensor::SetData(const uint8_t *data, size_t size) {
  if (impl != nullptr) {
    return impl->SetData(data, size);
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func) {
  if (impl != nullptr) {
    return impl->SetData(data, size, deleter_func);
  }
  return GRAPH_FAILED;
}

std::unique_ptr<uint8_t[], Tensor::DeleteFunc> Tensor::ResetData() {
  if (impl != nullptr) {
    return impl->ResetData();
  }
  return std::unique_ptr<uint8_t[], Tensor::DeleteFunc>(nullptr, nullptr);
}

} // end ge
