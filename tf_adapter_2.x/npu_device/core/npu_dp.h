/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: Common depends and micro defines for and only for data preprocess module
 */

#ifndef TENSORFLOW_NPU_DP_H
#define TENSORFLOW_NPU_DP_H

#include "tensorflow/c/c_api.h"
#include <utility>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"

#include "absl/algorithm/container.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"

#include "npu_micros.h"
#include "npu_types.h"

class IteratorResourceProvider {
  using ConsumeFunc = std::function<tensorflow::Status(tensorflow::Tensor)>;
  using DestroyFunc = std::function<tensorflow::Status()>;
  using DoneCallback = std::function<void(tensorflow::Status)>;

  struct Request {
    Request(tensorflow::Tensor t, int64_t n, DoneCallback d) : resource(std::move(t)), nums(n), done(std::move(d)) {}
    tensorflow::Tensor resource;
    int64_t nums;
    DoneCallback done;
  };

 public:
  tensorflow::Status ConsumeAsync(tensorflow::Tensor resource, int64_t nums, const DoneCallback &done) {
    {
      if (stopped_) {
        return tensorflow::errors::Internal("Iterator resource provider ", name_, " has stopped");
      }
      std::unique_lock<std::mutex> lk(mu_);
      if (request_stop_) {
        return tensorflow::errors::Internal("Iterator resource provider ", name_, " is stopping");
      }
      requests_.emplace(Request(std::move(resource), nums, done));
    }
    cv_.notify_one();
    return tensorflow::Status::OK();
  }
  tensorflow::Status Destroy() {
    {
      std::unique_lock<std::mutex> lk(mu_);
      request_stop_ = true;
    }
    cv_.notify_one();
    while (!stopped_) {
    }
    return destroy_func_();
  }

  std::string Name() const { return name_; }

  IteratorResourceProvider(std::string name, ConsumeFunc cf, DestroyFunc df)
      : name_(std::move(name)),
        consume_func_(std::move(cf)),
        destroy_func_(std::move(df)),
        request_stop_(false),
        stopped_(false) {
    worker_.reset(
      tensorflow::Env::Default()->StartThread(tensorflow::ThreadOptions{}, name_ + "_hdc_provider", [this]() {
        while (true) {
          std::unique_lock<std::mutex> lk(mu_);
          cv_.wait(lk, [this]() -> bool { return !requests_.empty() || request_stop_; });
          if (request_stop_) {
            stopped_.store(true);
            return;
          }
          auto task = requests_.front();
          requests_.pop();
          lk.unlock();
          int64_t nums = task.nums;
          tensorflow::Status status = tensorflow::Status::OK();
          while (nums-- > 0 && status.ok()) {
            status = consume_func_(task.resource);
          }
          task.done(status);
        }
      }));
  }
  ~IteratorResourceProvider() {
    {
      std::unique_lock<std::mutex> lk(mu_);
      stopped_ = true;
    }
    cv_.notify_one();
  }
  static tensorflow::FunctionDef GetFunctionDef(std::string channel_name, std::vector<int> device_ids,
                                                const TensorPartialShapes &shapes, const TensorDataTypes &types,
                                                TF_Status *status) {
    tensorflow::FunctionDef fdef;
    std::unique_ptr<tensorflow::Graph> graph = std::make_unique<tensorflow::Graph>(tensorflow::OpRegistry::Global());

    tensorflow::Node *arg_iterator = nullptr;
    tensorflow::Node *iterator_h2d = nullptr;

    NPU_CTX_REQUIRES_OK_RETURN(status,
                               tensorflow::NodeBuilder("arg_iterator", "_Arg")
                                 .Attr("index", 0)
                                 .Attr("T", tensorflow::DT_RESOURCE)
                                 .Finalize(graph.get(), &arg_iterator),
                               fdef);

    NPU_CTX_REQUIRES_OK_RETURN(status,
                               tensorflow::NodeBuilder("iterator_h2d", "IteratorH2D")
                                 .Input(arg_iterator, 0)
                                 .Attr("device_ids", device_ids)
                                 .Attr("channel_name", channel_name)
                                 .Finalize(graph.get(), &iterator_h2d),
                               fdef);

    NPU_CTX_REQUIRES_OK_RETURN(status, tensorflow::GraphToFunctionDef(*graph, "dp_provider_" + channel_name, &fdef),
                               fdef);
    return fdef;
  }

 private:
  std::string name_;
  ConsumeFunc consume_func_;
  DestroyFunc destroy_func_;
  bool request_stop_;
  std::atomic_bool stopped_{false};
  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<Request> requests_;
  std::unique_ptr<tensorflow::Thread> worker_;
};

#endif  // TENSORFLOW_NPU_DP_H
