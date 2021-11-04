/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef EXECUTE_GRAPH_ATTR_STORE_H
#define EXECUTE_GRAPH_ATTR_STORE_H
#include <string>
#include <unordered_map>
#include <map>
#include <set>

#include "any_value.h"

namespace ge {
using AttrId = uint64_t;
using AttrSubId = uint32_t;
enum AttrType {
  kAttrPredefinedInIr,  // IR预定义的属性
  kAttrGeneral,         // 通用属性
  kAttrTypeEnd
};
constexpr inline uint32_t GetAttrType(AttrId id) {
  return id >> 32;
}
constexpr inline uint32_t GetSubAttrId(AttrId id) {
  return id & 0xffffffff;
}
constexpr inline AttrId GetAttrId(uint32_t type, uint32_t sub_id) {
  return static_cast<uint64_t>(type) << 32 | static_cast<uint64_t>(sub_id);
}
constexpr AttrId kInvalidAttrId = GetAttrId(0xffffffff, 0);

class AttrStore {
public:
  static AttrStore Create(size_t pre_defined_attr_count);

  template<typename T>
  bool Set(AttrId attr_id, T &&value);
  template<typename T>
  bool Set(AttrId attr_id, const T &value);
  template<typename T>
  bool SetByName(const std::string &name, T &&value);
  template<typename T>
  bool SetByName(const std::string &name, const T &value);

  template<typename T>
  const T *Get(AttrId attr_id) const;
  template<typename T>
  T *MutableGet(AttrId attr_id);
  template<typename T>
  const T *GetByName(const std::string &name) const;
  template<typename T>
  T *MutableGetByName(const std::string &name);

  AttrId GetIdByName(const std::string &name) const noexcept;
  void SetNameAndId(std::string name, AttrId id);

  bool Exists(AttrId attr_id) const noexcept;
  bool Exists(const std::string &name) const noexcept;

  bool Delete(const std::string &name);

  void Swap(AttrStore &other);
  bool SetAnyValueByName(const std::string &name, const AnyValue &value);

  // unordered版本更好，为了兼容老版本接口，仍然用set和map，不论用哪种数据结构，这都是非常低效的接口
  std::set<std::string> GetAllAttrNames() const;
  std::map<std::string, AnyValue> GetAllAttrs() const;

  AnyValue *MutableAnyValue(const std::string &name) noexcept;
  AnyValue *GetOrCreateAnyValue(const std::string &name);
  const AnyValue *GetAnyValue(const std::string &name) const noexcept;

private:
  AnyValue *MutableAnyValue(AttrId attr_id) noexcept;
  AnyValue *GetOrCreateAnyValue(AttrId attr_id);
  const AnyValue *GetAnyValue(AttrId attr_id) const noexcept;

private:
  constexpr static int kDefaultMaxAttrCount = 8;

  class PreDefinedAttrStore {
  public:
    bool Exists(AttrSubId index) const noexcept;
    bool Delete(AttrSubId index);
    void Swap(PreDefinedAttrStore &other);

    AnyValue *GetOrCreateAnyValue(AttrSubId index);
    AnyValue *MutableAnyValue(AttrSubId index) noexcept;
    const AnyValue *GetAnyValue(AttrSubId index) const noexcept;

    void Resize(size_t s);

  private:
    std::vector<AnyValue> attrs_;
  };

  class CustomDefinedAttrStore {
  public:
    bool Exists(const std::string &name) const noexcept;
    bool Delete(const std::string &name);
    void Swap(CustomDefinedAttrStore &other);

    AnyValue *GetOrCreateAnyValue(const std::string &name);
    AnyValue *MutableAnyValue(const std::string &name) noexcept;
    const AnyValue *GetAnyValue(const std::string &name) const noexcept;

    void GetAllNames(std::set<std::string> &names) const;
    void GetAllAttrs(std::map<std::string, AnyValue> &names_to_attr) const;

  private:
    std::unordered_map<std::string, AnyValue> attrs_;
  };

private:
  std::unordered_map<std::string, AttrId> names_to_id_;
  // 更好的办法是定义一个虚基类、派生出两个子类，然后保存两个子类的指针：`std::array<std::unique_ptr<SubAttrStore>, kAttrTypeEnd>`
  // 然后根据不同的SubAttr类型，调用对应子类的函数。但是这么做会导致创建AttrStore时，总会带有两次子类实例堆申请的开销，
  // 为了减少堆内存申请，直接将子类平铺在成员变量上。
  PreDefinedAttrStore pre_defined_attrs_;
  CustomDefinedAttrStore general_attrs_;
};

#define SET_IMPL(key, value) \
    auto v = GetOrCreateAnyValue(key);  \
    if (v == nullptr) {  \
      return false;  \
    }  \
    v->SetValue(value);  \
    return true;

#define SET_IMPL_RVALUE(key, value) \
  auto v = GetOrCreateAnyValue(key);  \
  if (v == nullptr) {  \
  return false;  \
  }  \
  v->SetValue(std::forward<T>(value));  \
  return true;

template<typename T>
bool AttrStore::Set(AttrId attr_id, const T &value) {
  SET_IMPL(attr_id, value)
}
template<typename T>
bool AttrStore::Set(AttrId attr_id, T &&value) {
  SET_IMPL_RVALUE(attr_id, value)
}
template<typename T>
bool AttrStore::SetByName(const std::string &name, T &&value) {
  SET_IMPL_RVALUE(name, value)
}
template<typename T>
bool AttrStore::SetByName(const std::string &name, const T &value) {
  SET_IMPL(name, value)
}

#define GET_IMPL(key) \
    auto v = GetAnyValue(key);  \
    if (v == nullptr) {  \
      return nullptr;  \
    }  \
    return v->Get<T>();

template<typename T>
const T *AttrStore::Get(AttrId attr_id) const {
  GET_IMPL(attr_id)
}
template<typename T>
const T *AttrStore::GetByName(const std::string &name) const {
  GET_IMPL(name)
}

#define MUTABLE_IMPL(key) \
    auto v = MutableAnyValue(key);  \
    if (v == nullptr) {  \
      return nullptr;  \
    }  \
    return v->MutableGet<T>();
template<typename T>
T *AttrStore::MutableGet(AttrId attr_id) {
  MUTABLE_IMPL(attr_id)
}
template<typename T>
T *AttrStore::MutableGetByName(const std::string &name) {
  MUTABLE_IMPL(name)
}

}  // namespace ge

#endif  // EXECUTE_GRAPH_ATTR_STORE_H
