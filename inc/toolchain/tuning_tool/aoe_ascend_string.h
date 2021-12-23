/**
 * @file aoe_ascend_string.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 * 描述：mstune调优接口头文件
 */

#ifndef AOE_ASCEND_STRING_H
#define AOE_ASCEND_STRING_H
#include <memory>
#include <string>

namespace Aoe {
class AscendString {
public:
    AscendString() = default;
    ~AscendString() = default;
    inline explicit AscendString(const char *name);

    inline const char *GetString() const;
    inline bool operator<(const AscendString &d) const;
    inline bool operator>(const AscendString &d) const;
    inline bool operator<=(const AscendString &d) const;
    inline bool operator>=(const AscendString &d) const;
    inline bool operator==(const AscendString &d) const;
    inline bool operator!=(const AscendString &d) const;

private:
    std::shared_ptr<std::string> name_;
};

inline AscendString::AscendString(const char *name) {
  if (name != nullptr) {
    try {
      name_ = std::make_shared<std::string>(name);
    } catch (...) {
      name_ = nullptr;
    }
  }
}

inline const char *AscendString::GetString() const {
  if (name_ == nullptr) {
    return "";
  }
  return (*name_).c_str();
}

inline bool AscendString::operator<(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ < *(d.name_));
}

inline bool AscendString::operator>(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return true;
  }
  return (*name_ > *(d.name_));
}

inline bool AscendString::operator==(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return true;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ == *(d.name_));
}

inline bool AscendString::operator<=(const AscendString &d) const {
  if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ <= *(d.name_));
}

inline bool AscendString::operator>=(const AscendString &d) const {
  if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  }
  return (*name_ >= *(d.name_));
}

inline bool AscendString::operator!=(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return true;
  }
  return (*name_ != *(d.name_));
}
} // namespace Aoe
#endif
