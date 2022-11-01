/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#include <thread>
#include <mutex>
#include <functional>
#include <deque>

#include "securec.h"
#include "acl/acl_rt.h"
#include "tf_adapter/common/adapter_logger.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"

#ifndef TENSORFLOW_CORE_KERNELS_NPU_STREAM_H_
#define TENSORFLOW_CORE_KERNELS_NPU_STREAM_H_
namespace tensorflow {
namespace data {
enum class StreamNum {
  DEFAULT_STREAM_NUM = 28,
  MAX_STREAM_NUM = 1024
};

class Stream;
class StreamPool;

class StreamEvent {
public:
  ~StreamEvent() {
    ADP_LOG(EVENT) << "~StreamEvent";
    if (event_ != nullptr) {
      aclError rt = aclrtDestroyEvent(event_);
      if (rt != ACL_SUCCESS) {
        ADP_LOG(ERROR) << "[StreamPool] Destroy event faild!";
        event_ = nullptr;
      }
    }
    stream_ = nullptr;
  }

private:
  static std::shared_ptr<StreamEvent> CreateEvent(Stream *stream, const std::function<void(StreamEvent *)> del) {
    aclrtEvent event;
    aclError rt = aclrtCreateEvent(&event);
    if (rt != ACL_SUCCESS) {
      ADP_LOG(ERROR) << "[StreamPool] Create stream faild!";
      return nullptr;
    }
    return std::shared_ptr<StreamEvent>(new StreamEvent(stream, event), del);
  }

  inline Status RecordEvent(const std::function<void(Status status)> hook);
  inline Status Wait();

  Status TryWait() {
    aclrtEventStatus status;
    aclError rt = aclrtQueryEvent(event_, &status);
    if (rt != ACL_SUCCESS) {
      ADP_LOG(ERROR) << "[StreamPool] Query event status faild.";
      return errors::InvalidArgument("[StreamPool] Query event status faild: ", rt);
    }

    if (status != ACL_EVENT_STATUS_COMPLETE) {
      return errors::InvalidArgument("[StreamPool] Event not ready");
    }

    if (hook_ != nullptr) {
      hook_(Status::OK());
      hook_ = nullptr;
    }
    return Status::OK();
  }

  StreamEvent(Stream *stream, aclrtEvent event)
    : stream_(stream),
      event_(event) {};

  Stream *stream_ = nullptr;
  aclrtEvent event_ = nullptr;
  std::function<void(Status status)> hook_ = nullptr;

  friend class Stream;
  friend class StreamPool;
};

class Stream {
public:
  ~Stream() {
    ADP_LOG(EVENT) << "~Stream: stream_id = " << stream_id_;
    waiting_event_queue_.clear();
    for (auto event : event_queue_) {
      delete event;
    }
    event_queue_.clear();
    aclError rt = aclrtDestroyStream(stream_);
    if (rt != ACL_SUCCESS) {
      ADP_LOG(ERROR) << "[StreamPool] Destroy stream faild!";
    }
  }

  aclrtStream GetStream() const {
    return stream_;
  }

private:
  inline uint64_t GetStreamId() const { return stream_id_; }

  static std::shared_ptr<Stream> CreateStream(StreamPool *owner, uint64_t stream_id) {
    aclrtStream stream;
    aclError rt = aclrtCreateStream(&stream);
    if (rt != ACL_SUCCESS) {
        ADP_LOG(ERROR) << "[StreamPool] Create stream faild!";
        return nullptr;
    }

    std::function<void(Stream *)> del =  [](Stream *stream) {
      delete stream;
    };

    return std::shared_ptr<Stream>(new (std::nothrow)Stream(owner, stream_id, stream), del);
  }

  Status RecordEvent(const std::function<void(Status status)> func_, const std::function<void(StreamEvent *)> del) {
    std::unique_lock<std::mutex> lck(mtx_);
    std::shared_ptr<StreamEvent> stream_event;
    std::function<void(StreamEvent *)> equeue_del = [this, del](StreamEvent *event) {
      del(event);
      this->event_queue_.push_back(event);
    };
    if (!event_queue_.empty()) {
      stream_event.reset(event_queue_.front(), equeue_del);
      event_queue_.pop_front();
    } else {
      stream_event = StreamEvent::CreateEvent(this, equeue_del);
    }

    if (stream_event != nullptr) {
      Status status = stream_event->RecordEvent(func_);
      if (status.ok()) {
        waiting_event_queue_.emplace_back(stream_event);
      }
      return status;
    } else {
      return errors::InvalidArgument("Create event failed.");
    }
  }

  uint64_t ProcessAllReadyEvent() {
    uint64_t count = 0;
    for (auto it = waiting_event_queue_.begin(); it != waiting_event_queue_.end();) {
      if ((*it)->TryWait().ok()) {
        count++;
        it = waiting_event_queue_.erase(it);
      } else {
        ++it;
      }
    }

    return count;
  }

  Status WaitOneEvent() {
    uint64_t count = 0;
    std::shared_ptr<StreamEvent> event;
    {
      std::unique_lock<std::mutex> lck(mtx_);
      count = ProcessAllReadyEvent();
      if (count > 0) {
        return Status::OK();
      }

      if (waiting_event_queue_.empty()) {
        return errors::InvalidArgument("No event wait to be process.");
      }
      event = std::move(waiting_event_queue_.front());
      waiting_event_queue_.pop_front();
    }

    (void)event->Wait();
    return Status::OK();
  }

  explicit Stream(StreamPool *owner, uint64_t stream_id, aclrtStream stream)
    : stream_(stream),
      stream_id_(stream_id),
      owner_(owner) {
    ADP_LOG(EVENT) << "[StreamPool] Create stream, id = " << stream_id;
  };

  std::mutex mtx_;
  aclrtStream stream_;
  uint64_t stream_id_ = UINT64_MAX;
  std::deque<StreamEvent*> event_queue_;
  std::deque<std::shared_ptr<StreamEvent>> waiting_event_queue_;
  StreamPool *owner_;
  friend class StreamPool;
};

Status StreamEvent::RecordEvent(const std::function<void(Status status)> hook) {
  if (event_ != nullptr) {
    aclError rt = aclrtRecordEvent(event_, stream_->GetStream());
    if (rt != ACL_SUCCESS) {
      ADP_LOG(ERROR) << "[StreamPool] Record event faild, rt : " << rt;
      return errors::InvalidArgument("[StreamPool] Record event faild: ", rt);
    }
    hook_ = hook;
  } else {
    ADP_LOG(INFO) << "[StreamPool] Record event faild: event is null. ";
  }

  return Status::OK();
}

Status StreamEvent::Wait() {
  aclError rt;
  if (event_ != nullptr) {
    rt = aclrtSynchronizeEvent(event_);
  } else {
    rt = aclrtSynchronizeStream(stream_->GetStream());
  }

  Status status;
  if (rt != ACL_SUCCESS) {
    ADP_LOG(ERROR) << "[StreamPool] Syn event faild, rt = " << rt;
    status = errors::InvalidArgument("[StreamPool] Syn event faild, rt: ", rt);
  }
  if (hook_ != nullptr) {
    hook_(status);
    hook_ = nullptr;
  }
  return status;
}

class StreamPool {
public:
  explicit StreamPool(uint64_t stream_num, uint64_t max_task)
    : max_stream_(stream_num),
      max_task_(max_task) {
    uint64_t pos_stream_num = StreamPool::CheckStreamNum(stream_num);
    cur_event_num_ = new uint64_t[pos_stream_num];
    uint64_t size_count = sizeof(uint64_t) * pos_stream_num;
    if (memset_s(cur_event_num_, size_count, 0, size_count) != 0) {
      ADP_LOG(ERROR) << "[StreamPool] Failed to reset cur_event_num_ memory. size_count=" << size_count;
    }
    streams_.resize(pos_stream_num, nullptr);
  }

  ~StreamPool() {
    delete[] cur_event_num_;
    streams_.clear();
  }

  std::shared_ptr<Stream> GetStream(uint64_t streamid) {
    if (streamid >= max_stream_) {
      return nullptr;
    }
    std::unique_lock<std::mutex> lck(mtx_);
    if (streams_[streamid] == nullptr) {
      streams_[streamid] = Stream::CreateStream(this, streamid);
    }

    return streams_[streamid];
  }

  Status RecordEvent(const std::shared_ptr<Stream> stream, const std::function<void(Status status)> func_) {
    uint64_t stream_id = stream->GetStreamId();
    {
      std::unique_lock<std::mutex> lck(mtx_);
      if ((stream_id >= max_stream_) || (cur_event_num_[stream_id] >= max_task_)) {
        return errors::InvalidArgument("Cur stream is overload. reocrd event = ", cur_event_num_[stream_id]);
      }
      cur_event_num_[stream_id]++;
    }

    return stream->RecordEvent(func_, [this, stream_id](StreamEvent*) {
        std::unique_lock<std::mutex> lck(mtx_);
        this->cur_event_num_[stream_id]--;
    });
  }

  uint64_t GetIdleEventCount(const uint64_t stream_id) const {
    if ((stream_id >= max_stream_) || (cur_event_num_[stream_id] >= max_task_)) {
      return 0;
    }
    return max_task_ - cur_event_num_[stream_id];
  }

  uint64_t GetWaitingEventCount(const uint64_t stream_id) const {
    if (stream_id >= max_stream_) {
      return 0;
    }
    return cur_event_num_[stream_id];
  }

  Status WaitOneEvent(uint64_t stream_id) {
    std::shared_ptr<Stream> pstream = GetStream(stream_id);
    return pstream == nullptr ? errors::InvalidArgument("Stream is invalid, id=", stream_id) : pstream->WaitOneEvent();
  }

  static uint64_t CheckStreamNum(uint64_t stream_num) {
    if (stream_num == 0) {
      ADP_LOG(ERROR) << "[StreamPool] Check stream number error with stream_num=" << stream_num;
      return static_cast<uint64_t>(StreamNum::DEFAULT_STREAM_NUM);
    } else if (stream_num > static_cast<uint64_t>(StreamNum::MAX_STREAM_NUM)) {
      ADP_LOG(ERROR) << "[StreamPool] Check stream number error with stream_num=" << stream_num;
      return static_cast<uint64_t>(StreamNum::MAX_STREAM_NUM);
    }
    return stream_num;
  }

private:
  std::mutex mtx_;
  const uint64_t max_stream_;
  const uint64_t max_task_;
  uint64_t cur_stream_num_ = 0;
  uint64_t *cur_event_num_ = nullptr;
  std::vector<std::shared_ptr<Stream>> streams_;
  friend class Stream;
  friend class StreamEvent;
};
}  // namespace data
}  // namespace tensorflow
#endif // TENSORFLOW_CORE_KERNELS_NPU_STREAM_H_