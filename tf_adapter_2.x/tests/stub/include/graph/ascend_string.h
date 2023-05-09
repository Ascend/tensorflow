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

#ifndef INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
#define INC_EXTERNAL_GRAPH_ASCEND_STRING_H_

#include <string>
#include <memory>
#include <functional>

namespace ge {
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
}
#endif  // INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
