/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef TENSORFLOW_TF_ADAPTER_KERNELS_DATA_ITEM_DELEVER_H
#define TENSORFLOW_TF_ADAPTER_KERNELS_DATA_ITEM_DELEVER_H
#include <string.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <unistd.h>

#include <fstream>
#include <memory>
#include <vector>

#include "securec.h"
#include "tdt/data_common.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/kernels/threads_pool.h"

namespace tensorflow {
namespace data {
static constexpr char *SOCKET_SERVER_PATH = "/tmp/server";
static constexpr char *MESSAGE_HEAD = "head_check";
static constexpr int QLEN = 8;
static constexpr int HEAD_INFO_SIZE = 3;
static constexpr int ITEM_INFO_SIZE = 9;
static constexpr MAX_TRY_TIMES = 300;
static constexpr size_t UINT32_SIZE = sizeof(uint32_t);
static constexpr size_t UINT64_SIZE = sizeof(uint64_t);
static constexpr size_t CHAR_SIZE = sizeof(char);
static constexpr size_t DATA_TYPE_SIZE = sizeof(tdt::TdtDataType);

class DataItemDeliver {
 public:
  DataItemDeliver(int local_rank_id, int device_id,
                  const std::vector<uint32_t> &local_device_list,
                  const std::string &channel_name);
  Status ParallelInitSocketClient();
  void ParallelSendDataVec(std::vector<tdt : DataItem> &data_item);
  Status InitSocketServer();
  Status RecvDataVec(std::vector<tdt : DataItem> &data_item);
  ~DataItemDeliver();

 private:
  Status InitSocketClient(int device_id);
  Status SendDataVec(std::vector<tdt::DataItem> &data_items, int fd);
  Status CreateSockAddr(struct sockaddr_un &sockaddr, const char *path,
                        int local_rank_id);
  int Recv(void *buffer, size_t data_len);
  template <typename T>
  Status GetDataLen(T &value, size_t size);
  Status GetTensorType(tdt::TdtDataType &data_type);
  Status GetTensorData(uint64_t &data_len, std::shared_ptr<void> &data_ptr);
  Status GetTensorString(std::string &str);
  void SocketSend(struct iovec &temp_items[], int vec_size, int fd);
  Status CheckHead(const char *check_value);

  mutex client_list_mu_;
  std::vector<int> client_fd_list_;
  int server_fd_;
  std::shared_ptr<ThreadPool> pools_;
  struct sockaddr_un local_addr_ = {0};
  int local_rank_id_;
  std::vector<uint32_t> local_device_list_;
  uint32_t device_id_;
  std::string channel_name_;
};

DataItemDeliver::DataItemDeliver(int local_rank_id, int device_id,
                                 const std::vector<uint32_t> &local_device_list,
                                 const std::string &channel_name)
    : local_rank_id_(local_rank_id),
      device_id_(device_id),
      local_device_list_(local_device_list),
      channel_name_(channel_name) {
  pools_ = std::make_shared<ThreadPool>();
  pools_->InitThreadPool(local_device_list_.size());
}

DataItemDeliver::~DataItemDeliver() {
  for (int fd : client_fd_list_) {
    close(fd);
  }
  if (local_rank_id_ > 0) {
    close(server_fd_);
    unlink(local_addr_.sun_path);
  }
  ADP_LOG(INFO) << "DataItemDeliver is released.";
}

Status DataItemDeliver::ParallelInitSocketClient() {
  std::vector<std::future<Status>> init_status;
  for (int i = 1; i < local_device_list_.size(); i++) {
    init_status.emplace_back(
        pools_->Enqueue(&DataItemDeliver::InitSocketClient, this, i));
  }
  for (auto &&result : init_status) {
    if (result.get() != Status::OK()) {
      ADP_LOG(ERROR) << "Init socket client failed.";
      LOG(ERROR) << "Init socket client failed.";
      return errors::Internal("Init socket client failed.");
    }
  }
  return Status::OK();
}

Status DataItemDeliver::InitSocketClient(int device_id) {
  int fd = socket(AF_UNIX, SOCKET_STREAM, 0);
  if (fd < 0) {
    ADP_LOG(ERROR) << "Failed to open unix domain socket.";
    LOG(ERROR) << "Failed to open unix domain socket.";
    return errors::Internal("Failed to open unix domain socket.");
  }
  struct sockaddr_un peer_addr = {0};
  if (CreateSockAddr(peer_addr, SOCKET_SERVER_PATH, device_id) !=
      Status::OK()) {
    ADP_LOG(ERROR) << "Failed to create socket.";
    LOG(ERROR) << "Failed to create socket.";
    close(fd);
    return errors::Internal("Failed to create socket.");
  }
  int try_times = 0;
  int ret = 0;
  while (true) {
    ret = connet(fd, (struct sockaddr *)&peer_addr, sizeof(peer_addr));
    if (ret >= 0) {
      break;
    }
    usleep(50000);
    try_times++;
    if (try_times >= MAX_TRY_TIMES) {
      ADP_LOG(ERROR) << "Failed to connect server.";
      LOG(ERROR) << "Failed to connect server.";
      close(fd);
      return errors::Internal("Failed to connect server.");
    }
  }
  {
    mutex_lock lck(client_list_mu_);
    client_fd_list_.push_back(fd);
  }
  ADP_LOG(INFO) << "device:" << device_id << "connect to server success.";
  return Status::OK();
}

Status DataItemDeliver::InitSocketServer() {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    ADP_LOG(ERROR) << "Failed to open unix domain socket.";
    LOG(ERROR) << "Failed to open unix domain socket.";
    return errors::Internal("Failed to open unix domain socket.");
  }
  if (CreateSockAddr(local_addr_, SOCKET_SERVER_PATH, device_id_) !=
      Status::OK()) {
    ADP_LOG(ERROR) << "Failed to create socket.";
    LOG(ERROR) << "Failed to create socket.";
    return errors::Internal("Failed to create socket.");
  }
  unlink(local_addr_.sun_path);
  socklen_t addr_size = sizeof(local_addr_);
  if (bind(fd, (struct sockaddr *)&local_addr_, addr_size) < 0) {
    ADP_LOG(ERROR) << "Bind fd failed:" << strerror(errno) << "(errno:" << errno
                   << ")";
    LOG(ERROR) << "Bind fd failed:" << strerror(errno) << "(errno:" << errno
               << ")";
    close(fd);
    return errors::Internal("Bind fd failed.");
  }
  if (listen(fd, QLEN) < 0) {
    ADP_LOG(ERROR) << "Listen fd failed:" << strerror(errno)
                   << "(errno:" << errno << ")";
    LOG(ERROR) << "Listen fd failed:" << strerror(errno) << "(errno:" << errno
               << ")";
    close(fd);
    return errors::Internal("Listen fd failed.");
  }
  int try_times = 0;
  while (true) {
    server_fd_ = accept(fd, (struct sockaddr *)&local_addr_, &addr_size);
    if (server_fd_ != -1) {
      break;
    }
    usleep(50000);
    try_times++;
    if (try_times >= MAX_TRY_TIMES) {
      ADP_LOG(ERROR) << "Failed to accept server.";
      LOG(ERROR) << "Failed to accept server.";
      close(fd);
      return errors::Internal("Failed to accept server.");
    }
  }
  ADP_LOG(INFO) << "Socket server connect success, path:"
                << local_addr_.sun_path;
  close(fd);
  return Status::OK();
}

Status DataItemDeliver::CheckHead(const char *check_value) {
  uint32_t head_size = 0;
  int recvn = Recv(&head_size, UINT32_SIZE);
  if (recvn != UINT32_SIZE) {
    ADP_LOG(ERROR) << "Failed to recv head length.";
    LOG(ERROR) << "Failed to recv head length.";
    return errors::Internal("Failed to recv head length.");
  }
  char *head = (char *)malloc(head_size);
  if (head == nullptr) {
    ADP_LOG(ERROR) << "Failed to malloc head buffer.";
    LOG(ERROR) << "Failed to malloc head buffer.";
    return errors::Internal("Failed to malloc head buffer.");
  }
  recvn = Recv(head, head_size);
  if (recvn != head_size) {
    free(head);
    ADP_LOG(ERROR) << "Failed to recv head value.";
    LOG(ERROR) << "Failed to recv head value.";
    return errors::Internal("Failed to recv head value.");
  }
  if (strcmp(check_value, head) != 0) {
    free(head);
    ADP_LOG(ERROR) << "Check head failed, recv:" << head
                   << ", while right value is:" << check_value;
    LOG(ERROR) << "Check head failed, recv:" << head
               << ", while right value is:" << check_value;
    return errors::Internal("Check head failed.");
  }
  free(head);
  return Status::OK();
}

Status DataItemDeliver::RecvDataVec(std::vector<tdt::DataItem> &items) {
  if (CheckHead(MESSAGE_HEAD) != Status::OK()) {
    ADP_LOG(ERROR) << "Cancel recv data for head check failed.";
    LOG(ERROR) << "Cancel recv data for head check failed.";
    return errors::Internal("Cancel recv data for head check failed.");
  }
  uint32_t vec_size = 0;
  if (GetDataLen(vec_size, UINT32_SIZE) != Status::OK() || vec_size == 0) {
    return errors::Internal("Get vector size failed.");
  }
  for (uint32_t i = 0; i < vec_size; i++) {
    tdt::DataItem data_item;
    if (GetTensorType(data_item.dataType_) != Status::OK()) {
      return errors::Internal("Get tensor type failed.");
    }
    if (GetTensorString(data_item.tensorName_) != Status::OK()) {
      return errors::Internal("Get tensor name failed.");
    }
    if (GetTensorString(data_item.tensorShape_) != Status::OK()) {
      return errors::Internal("Get tensor shape failed.");
    }
    if (GetTensorString(data_item.tensorType_) != Status::OK()) {
      return errors::Internal("Get tensor type failed.");
    }
    if (GetTensorData(data_item.dataLen_, data_item.dataPtr_) != Status::OK()) {
      return errors::Internal("Get tensor name failed.");
    }
    items.push_back(data_item);
  }
  return Status::OK();
}

int DataItemDeliver::Recv(void *buffer, size_t data_len) {
  int ret = -1;
  uint64_t buf_pos = 0;
  while (data_len > 0) {
    do {
      ret = recv(server_fd_, buffer + buf_pos, data_len, 0);
    } while ((ret == -1) && (errno == EINTR));
    if (ret == 0) {
      // if master first reach max step ,socket will be close. correspond to
      // SocketSend WARNING
      ADP_LOG(WARNING) << "Client connect closed, server_fd:" << server_fd_
                       << ", channel_name:" << channel_name_;
      LOG(WARNING) << "Client connect closed, server_fd:" << server_fd_
                   << ", channel_name:" << channel_name_;
      return ret;
    } else if (ret < 0) {
      ADP_LOG(ERROR) << "Recv data failed,error:" << strerror(errno)
                     << ", (errno:" << errno << "), server_fd:" << server_fd_;
      LOG(ERROR) << "Recv data failed,error:" << strerror(errno)
                 << ", (errno:" << errno << "), server_fd:" << server_fd_;
      return ret;
    }
    buf_pos += ret;
    data_len -= ret;
  }
  return buf_pos;
}

template <typename T>
Status DataItemDeliver::GetDataLen(T &value, size_t size) {
  int recvn = Recv(&value, size);
  if (recvn != size) {
    return errors::Internal("Failed to recv data length.");
  }
  return Status::OK();
}

Status DataItemDeliver::GetTensorType(tdt::TdtDataType &data_type) {
  int recvn = Recv(&data_type, UINT32_SIZE);
  if (recvn != UINT32_SIZE) {
    return errors::Internal("Failed to recv data length.");
  }
  return Status::OK();
}

Status DataItemDeliver::GetTensorData(uint64_t &data_len,
                                      std::shared_ptr<void> &data_ptr) {
  TF_RETURN_IF_ERROR(GetDataLen(data_len, UINT64_SIZE));
  void *buff = malloc(data_len);
  if (buff == nullptr) {
    ADP_LOG(ERROR) << "Malloc data failed, size:" << data_len
                   << ", device_id:" << device_id_
                   << ", channel_name:" << channel_name_;
    LOG(ERROR) << "Malloc data failed, size:" << data_len
               << ", device_id:" << device_id_
               << ", channel_name:" << channel_name_;
    return errors::Internal("Malloc data failed.");
  }
  if (memset_s(buff, data_len, 0 data_len) != 0) {
    free(buff);
    ADP_LOG(ERROR) << "Failed to reset buff memory.";
    LOG(ERROR) << "Failed to reset buff memory.";
    return errors::Internal("Failed to reset buff memory.");
  }
  int recvn = Recv(buff, data_len);
  if (recvn != data_len) {
    free(buff);
    ADP_LOG(ERROR) << "Failed to receive data.";
    LOG(ERROR) << "Failed to receive data.";
    return errors::Internal("Failed to receive data.");
  }
  data_ptr = std::shared_ptr<void>(
      buff, [](void *elem) { free(elem); }) return Status::OK();
}

Status DataItemDeliver::GetTensorString(std::string &str) {
  uint32_t size = 0;
  TF_RETURN_IF_ERROR(GetDataLen(size, UINT64_SIZE));
  void *buff = malloc(size);
  if (buff == nullptr) {
    ADP_LOG(ERROR) << "Malloc string failed, size:" << size
                   << ", device_id:" << device_id_
                   << ", channel_name:" << channel_name_;
    LOG(ERROR) << "Malloc string failed, size:" << size
               << ", device_id:" << device_id_
               << ", channel_name:" << channel_name_;
    return errors::Internal("Malloc string failed.");
  }
  if (memset_s(buff, size, 0 size) != 0) {
    free(buff);
    ADP_LOG(ERROR) << "Failed to reset buff memory.";
    LOG(ERROR) << "Failed to reset buff memory.";
    return errors::Internal("Failed to reset buff memory.");
  }
  int recvn = Recv(buff, size);
  if (recvn != size) {
    free(buff);
    ADP_LOG(ERROR) << "Failed to receive data.";
    LOG(ERROR) << "Failed to receive data.";
    return errors::Internal("Failed to receive data.");
  }
  str = static_cast<char *>(buff);
  free(buff);
  return Status::OK();
}

void DataItemDeliver::ParallelSendDataVec(
    std::vector<tdt::DataItem> &data_items) {
  // only master need send
  if (local_rank_id_ != 0) {
    return;
  }
  std::vector<std::future<Status>> init_status;
  for (int fd : client_fd_list_) {
    init_status.emplace_back(
        pools_->Enqueue(&DataItemDeliver::SendDataVec, this, data_items, fd));
  }
  for (auto &&result::init_status) {
    result.get();
  }
}

Status DataItemDeliver::SendDataVec(std::vector<tdt::DataItem> &data_items,
                                    int fd) {
  uint32_t vector_size = data_items.size();
  // message in buffer:    [head][item][item]...[head][item][item]...
  // send head info
  struct iovec head_info[HEAD_INFO_SIZE];
  uint32_t head_size = (strlen(MESSAGE_HEAD) + 1) * CHAR_SIZE;
  head_info[0].iov_base = &head_size;
  head_info[0].iov_len = UINT32_SIZE;
  head_info[1].iov_base = MESSAGE_HEAD;
  head_info[1].iov_len = head_size;
  head_info[2].iov_base = &vector_size;
  head_info[2].iov_len = UINT32_SIZE;
  SocketSend(head_info, HEAD_INFO_SIZE, fd);
  for (tdt::DataItem data_item : data_items) {
    // send dataType
    struct iovec item_info[ITEM_INFO_SIZE];
    item_info[0].iov_base = &data_item.dataType_;
    item_info[0].iov_len = DATA_TYPE_SIZE;

    // send tensor name
    char *tensor_name = &data_item.tensorName_[0];
    uint32_t name_size = (strlen(tensor_name) + 1) * CHAR_SIZE;
    item_info[1].iov_base = &name_size;
    item_info[1].iov_len = UINT32_SIZE;
    item_info[2].iov_base = tensor_name;
    item_info[2].iov_len = name_size;

    // send tensor shape
    char *tensor_shape = &data_item.tensorShape_[0];
    uint32_t shape_size = (strlen(tensor_name) + 1) * CHAR_SIZE;
    item_info[3].iov_base = &shape_size;
    item_info[3].iov_len = UINT32_SIZE;
    item_info[4].iov_base = tensor_shape;
    item_info[4].iov_len = shape_size;

    // send tensor type
    char *tensor_type = &data_item.tensorType_[0];
    uint32_t type_size = (strlen(tensor_type) + 1) * CHAR_SIZE;
    item_info[5].iov_base = &type_size;
    item_info[5].iov_len = UINT32_SIZE;
    item_info[6].iov_base = tensor_type;
    item_info[6].iov_len = type_size;

    // send tensor data
    item_info[7].iov_base = &data_item.dataLen_;
    item_info[7].iov_len = UINT64_SIZE;
    item_info[8].iov_base = static_cast<void *>(data_item.dataPtr_.get());
    item_info[8].iov_len = data_item.dataLen_;
    SocketSend(temp_item, ITEM_INFO_SIZE, fd);
  }
  return Status::OK();
}
Status DataItemDeliver::CreateSockAddr(struct sockaddr_un &sock_addr,
                                       const char *path, int device_id) {
  sock_addr.sun_family = AF_UNIX;
  int len = 0;
  if (-1 ==
      (len = snprintf(sock_addr.sun_path, sizeof(sock_addr.sun_path), "%s%s%d",
                      path, channel_name_.c_str(), device_id))) {
    ADP_LOG(ERROR) << "Set sun_path failed.";
    LOG(ERROR) << "Set sun_path failed.";
    return errors::Internal("Set sun_path failed.");
  }
  return Status::OK();
}
void DataItemDeliver::SocketSend(struct iovec temp_items[], int vector_size,
                                   int fd) {
  int sendn = writev(fd, temp_items, vec_size);
  // if salve first reach max step, socket will be closed, correspond to
  // Recv WARNING
  if (sendn < 0) {
    ADP_LOG(WARNING) << "Writev socket failed:" << strerror(error)
                     << "(errno:" << errno << "), return value:" << sendn
                     << ", fd:" << fd << ", channel_name:" << channel_name_;
    LOG(WARNING) << "Writev socket failed:" << strerror(error)
                 << "(errno:" << errno << "), return value:" << sendn
                 << ", fd:" << fd << ", channel_name:" << channel_name_;
  }
}
}  // namespace data
}  // namespace tensorflow
#endif