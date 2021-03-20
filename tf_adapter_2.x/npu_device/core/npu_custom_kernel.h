/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: Common depends and micro defines for and only for data preprocess module
 */

#ifndef TENSORFLOW_NPU_CUSTOM_KERNEL_H
#define TENSORFLOW_NPU_CUSTOM_KERNEL_H

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

#include "npu_device.h"
#include "npu_logger.h"
#include "npu_micros.h"
#include "npu_parser.h"
#include "npu_unwrap.h"
#include "npu_utils.h"

using NpuCustomKernelFunc =
  std::function<void(TFE_Context *, NpuDevice *, const npu::OpSpec *, const TensorShapes &, const tensorflow::NodeDef &,
                     int, TFE_TensorHandle **, int, TFE_TensorHandle **, TF_Status *)>;

using NpuFallbackHookFunc = std::function<void(TFE_Context *, NpuDevice *, const char *, const TFE_OpAttrs *, int,
                                               TFE_TensorHandle **, int, TFE_TensorHandle **, TF_Status *)>;

class CustomKernelRegistry {
 public:
  static CustomKernelRegistry &Instance() {
    static CustomKernelRegistry inst;
    return inst;
  }
  void Register(const std::string &op_name, const NpuCustomKernelFunc &func) {
    std::lock_guard<std::mutex> lk(mu_);
    DCHECK(specific_kernels_.find(op_name) == specific_kernels_.end());
    specific_kernels_.emplace(std::make_pair(op_name, func));
  }
  void RegisterHook(const std::string &op_name, const NpuFallbackHookFunc &func) {
    std::lock_guard<std::mutex> lk(mu_);
    DCHECK(specific_kernels_.find(op_name) == specific_kernels_.end());
    specific_hooks_.emplace(std::make_pair(op_name, func));
  }

  bool GetCustomKernelFunc(const std::string &op_name, NpuCustomKernelFunc **func) {
    DLOG() << "NPU Looking up custom kernel for " << op_name;
    std::lock_guard<std::mutex> lk(mu_);
    if (specific_kernels_.find(op_name) == specific_kernels_.end()) {
      DLOG() << "NPU Looking up kernel not found for op " << op_name;
      return false;
    }
    *func = &specific_kernels_[op_name];
    return true;
  }

  bool GetFallbackHookFunc(const std::string &op_name, NpuFallbackHookFunc **func) {
    DLOG() << "NPU Looking up callback hook for " << op_name;
    std::lock_guard<std::mutex> lk(mu_);
    if (specific_hooks_.find(op_name) == specific_hooks_.end()) {
      DLOG() << "NPU Callback hook not found for op " << op_name;
      return false;
    }
    *func = &specific_hooks_[op_name];
    return true;
  }

 private:
  CustomKernelRegistry() = default;
  std::mutex mu_;
  std::map<std::string, NpuCustomKernelFunc> specific_kernels_;
  std::map<std::string, NpuFallbackHookFunc> specific_hooks_;
};

class CustomKernelSpec {
 public:
  CustomKernelSpec(std::string name, NpuCustomKernelFunc custom_func)
      : op(std::move(name)), func(std::move(custom_func)) {}
  std::string op;
  NpuCustomKernelFunc func;
};

class FallbackHookSpec {
 public:
  FallbackHookSpec(std::string name, NpuFallbackHookFunc custom_func)
      : op(std::move(name)), func(std::move(custom_func)) {}
  std::string op;
  NpuFallbackHookFunc func;
};

class CustomKernelReceiver {
 public:
  CustomKernelReceiver(const CustomKernelSpec &spec) {  // NOLINT(google-explicit-constructor)
    DLOG() << "NPU Register custom kernel for " << spec.op;
    CustomKernelRegistry::Instance().Register(spec.op, spec.func);
  }

  CustomKernelReceiver(const FallbackHookSpec &spec) {  // NOLINT(google-explicit-constructor)
    DLOG() << "NPU Register fallback hook for " << spec.op;
    CustomKernelRegistry::Instance().RegisterHook(spec.op, spec.func);
  }
};

#define NPU_REGISTER_CUSTOM_KERNEL(name, func) NPU_REGISTER_CUSTOM_KERNEL_1(__COUNTER__, name, func)
#define NPU_REGISTER_CUSTOM_KERNEL_1(ctr, name, func) NPU_REGISTER_CUSTOM_KERNEL_2(ctr, name, func)
#define NPU_REGISTER_CUSTOM_KERNEL_2(ctr, name, func) \
  static CustomKernelReceiver __preserved_op##ctr = CustomKernelSpec(name, func)

#define NPU_REGISTER_FALLBACK_HOOK(name, func) NPU_REGISTER_FALLBACK_HOOK_1(__COUNTER__, name, func)
#define NPU_REGISTER_FALLBACK_HOOK_1(ctr, name, func) NPU_REGISTER_FALLBACK_HOOK_2(ctr, name, func)
#define NPU_REGISTER_FALLBACK_HOOK_2(ctr, name, func) \
  static CustomKernelReceiver __preserved_op##ctr = FallbackHookSpec(name, func)

#endif  // TENSORFLOW_NPU_CUSTOM_KERNEL_H
