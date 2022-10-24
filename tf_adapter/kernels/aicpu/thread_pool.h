/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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

#ifndef TENSORFLOW_TF_ADAPTER_KERNELS_THREAD_POOL_H
#define TENSORFLOW_TF_ADAPTER_KERNELS_THREAD_POOL_H

#include <atomic>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace data {
class ThreadPool {
public:
  template<class F, class... Args>
  auto Enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>;
  // initialize thread pool
  void InitThreadPool(size_t threads);
  // ThreadPool construct
  ThreadPool() : stop_(false), init_flag_(false) {}
  // ThreadPool destruct
  ~ThreadPool();
private:
  void AddWorkers(size_t threads);
  // need to keep track of threads so we can join them
  std::vector< std::thread > workers_;
  // the task queue
  std::queue< std::function<void()> > tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
  std::atomic<bool> init_flag_;
};

/**
 * @brief: add workers
 * @param threads: number of threads
 */
void ThreadPool::AddWorkers(size_t threads)
{
  for (size_t i = 0; i < threads; ++i) {
    workers_.emplace_back([this] {
      for (;;) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queue_mutex_);
          this->condition_.wait(lock,
              [this] { return this->stop_ || !this->tasks_.empty(); });
          if (this->stop_ || this->tasks_.empty()) { return; }
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }
        task();
      }
    });
  }
}

/**
 * @brief: launch some amount of workers_
 * @param threads: number of threads in thread pool
 */
void ThreadPool::InitThreadPool(size_t threads)
{
  if (!init_flag_) {
    AddWorkers(threads);
  }
  init_flag_ = true;
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::Enqueue(F&& f, Args&&... args)
  -> std::future<typename std::result_of<F(Args...)>::type>
{
  if (!init_flag_) { LOG(ERROR) << "thread pool is not initialized."; }
  using return_type = typename std::result_of<F(Args...)>::type;
  auto task = std::make_shared< std::packaged_task<return_type()> >(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));
  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (stop_) { LOG(ERROR) << "Enqueue on stopped ThreadPool."; }
    tasks_.emplace([task]() { (*task)(); });
  }
  condition_.notify_one();
  return res;
}

ThreadPool::~ThreadPool()
{
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  init_flag_ = false;
  condition_.notify_all();
  for (std::thread &worker : workers_) {
    try {
      worker.join();
    } catch (const std::system_error &) {
      LOG(FATAL) << "ThreadPool join failed because of system_error.";
    } catch (...) {
      LOG(FATAL) << "ThreadPool join failed because of unkown error.";
    }
  }
}
} // namespace data
} // namespace tensorflow
#endif