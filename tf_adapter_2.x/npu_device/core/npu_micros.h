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

#ifndef NPU_DEVICE_CORE_NPU_MICROS_H
#define NPU_DEVICE_CORE_NPU_MICROS_H

#define NPU_CTX_REQUIRES_OK(CTX, ...)          \
  do {                                         \
    (CTX)->status = (__VA_ARGS__);               \
    if (TF_PREDICT_FALSE(!CTX->status.ok())) { \
      LOG(ERROR) << (CTX)->status.ToString();    \
      return;                                  \
    }                                          \
  } while (0)

#define NPU_CTX_REQUIRES(CTX, EXP, STATUS)    \
  do {                                        \
    if (!TF_PREDICT_TRUE(EXP)) {              \
      (CTX)->status = (STATUS);               \
      LOG(ERROR) << (CTX)->status.ToString(); \
      return;                                 \
    }                                         \
  } while (0)

#define NPU_CTX_REQUIRES_OK_RETURN(CTX, EXP, RET) \
  do {                                            \
    (CTX)->status = (EXP);                        \
    if (TF_PREDICT_FALSE(!(CTX)->status.ok())) {  \
      LOG(ERROR) << (CTX)->status.ToString();     \
      return RET;                                 \
    }                                             \
  } while (0)

#define NPU_CTX_REQUIRES_RETURN(CTX, EXP, STATUS, RET) \
  do {                                                 \
    if (TF_PREDICT_FALSE(!(EXP))) {                    \
      (CTX)->status = (STATUS);                        \
      LOG(ERROR) << (CTX)->status.ToString();          \
      return RET;                                      \
    }                                                  \
  } while (0)

#define NPU_REQUIRES_OK(...)                    \
  do {                                          \
    tensorflow::Status _status = (__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_status.ok())) {      \
      LOG(ERROR) << _status.ToString();         \
      return _status;                           \
    }                                           \
  } while (0)

#define NPU_REQUIRES(EXP, STATUS)            \
  do {                                       \
    if (!TF_PREDICT_TRUE((EXP))) {           \
      tensorflow::Status _status = (STATUS); \
      LOG(ERROR) << _status.ToString();      \
      return _status;                        \
    }                                        \
  } while (0)

#define NPU_CTX_REQUIRES_GE_OK(CTX, PREFIX, ...)                        \
  do {                                                                  \
    ge::Status _status = (__VA_ARGS__);                                 \
    if (TF_PREDICT_FALSE(_status != ge::SUCCESS)) {                     \
      std::string err_msg = ge::GEGetErrorMsg();                        \
      if (err_msg.empty()) {                                            \
        err_msg = "<unknown error> code:" + std::to_string(_status);    \
      }                                                                 \
      CTX->status = tensorflow::errors::Internal(PREFIX, ":", err_msg); \
      LOG(ERROR) << CTX->status.ToString();                             \
      return;                                                           \
    }                                                                   \
  } while (0)

#define NPU_CTX_REQUIRES_GE_OK_RETURN(CTX, PREFIX, EXP, RET)              \
  do {                                                                    \
    ge::Status _status = (EXP);                                           \
    if (TF_PREDICT_FALSE(_status != ge::SUCCESS)) {                       \
      std::string err_msg = ge::GEGetErrorMsg();                          \
      if (err_msg.empty()) {                                              \
        err_msg = "<unknown error> code:" + std::to_string(_status);      \
      }                                                                   \
      (CTX)->status = tensorflow::errors::Internal(PREFIX, ":", err_msg); \
      LOG(ERROR) << (CTX)->status.ToString();                             \
      return RET;                                                         \
    }                                                                     \
  } while (0)

#define NPU_REQUIRES_ACL_OK(PREFIX, ...)                                              \
  do {                                                                                \
    auto _status = (__VA_ARGS__);                                                     \
    if (TF_PREDICT_FALSE(_status != ACL_ERROR_NONE)) {                                \
      return tensorflow::errors::Internal(PREFIX, ":<unknown error> code:", _status); \
    }                                                                                 \
  } while (0)

#define NPU_LOG_IF_ERROR(...)                                              \
  do {                                                                     \
    const ::tensorflow::Status _status = (__VA_ARGS__);                    \
    if (TF_PREDICT_FALSE(!_status.ok())) LOG(ERROR) << _status.ToString(); \
  } while (0)

#define HANDLE_ALL_FORMAT() \
  HANDLE_FORMAT(Nd)         \
  HANDLE_FORMAT(Nchw)       \
  HANDLE_FORMAT(Nc1hwc0)    \
  HANDLE_FORMAT(Fz)         \
  HANDLE_FORMAT(Hz)

#endif  // NPU_DEVICE_CORE_NPU_MICROS_H
