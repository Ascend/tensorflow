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

#include <memory>

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

#include "op_executors/npu_kernel_registry.h"

namespace npu {
static auto kernel = [](TFE_Context *context, NpuDevice *dev, const tensorflow::NodeDef &ndef, int num_inputs,
                        TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs, TF_Status *status) {
  TF_UNUSED_VARIABLE(context);
  TF_UNUSED_VARIABLE(num_inputs);
  TF_UNUSED_VARIABLE(inputs);
  for (int i = 0; i < num_outputs; ++i) {
    TFE_TensorHandle *retval = outputs[i];
    if (tensorflow::unwrap(retval)->DataType() == tensorflow::DT_RESOURCE) {
      const tensorflow::Tensor *tensor;
      NPU_CTX_REQUIRES_OK(status, npu::GetTensorHandleTensor(retval, &tensor));
      std::vector<tensorflow::PartialTensorShape> vec_shapes;
      TensorPartialShapes shapes;
      TensorDataTypes types;
      NPU_CTX_REQUIRES_OK(status, tensorflow::GetNodeAttr(ndef, "output_shapes", &vec_shapes));
      NPU_CTX_REQUIRES_OK(status, tensorflow::GetNodeAttr(ndef, "output_types", &types));
      for (const auto &shape : vec_shapes) {
        shapes.push_back(shape);
      }
      auto resource = tensor->scalar<tensorflow::ResourceHandle>()();
      DLOG() << "Start record mirrored host resource " << resource.DebugString();

      auto generator_ndef = std::make_shared<tensorflow::NodeDef>();
      tensorflow::NodeDefBuilder(npu::WrapResourceName(resource.name()), "IteratorV2")
        .Attr("container", resource.container())
        .Attr("shared_name", npu::WrapResourceName(resource.name()))
        .Attr("output_types", types)
        .Attr("output_shapes", vec_shapes)
        .Finalize(generator_ndef.get());

      std::stringstream ss;
      for (size_t j = 0; j < types.size(); j++) {
        auto &type = types[j];
        DLOG() << "Output " << j << " " << tensorflow::DataTypeString(type) << " " << vec_shapes[j].DebugString();
        if (tensorflow::DataTypeCanUseMemcpy(type)) {
          continue;
        } else if (type == tensorflow::DT_STRING) {
          tensorflow::TensorShape shape;
          if (vec_shapes[j].AsTensorShape(&shape) && tensorflow::TensorShapeUtils::IsScalar(shape)) {
            continue;
          } else {
            ss << "Output " << j << " unsupported non scalar string " << vec_shapes[j].DebugString();
          }
        } else {
          ss << "Output " << j << " unsupported data type " << tensorflow::DataTypeString(type);
        }
      }
      if (ss.str().empty()) {
        DLOG() << "Mirrored host resource " << resource.DebugString() << " recorded";
        dev->RecordResourceGeneratorDef(resource, std::make_shared<ResourceGenerator>(generator_ndef, 0));
        dev->RecordIteratorMirror(resource, shapes, types);
      } else {
        DLOG() << "Skip mirror host resource " << resource.DebugString() << " as " << ss.str();
      }
    }
  }
};

NPU_REGISTER_FALLBACK_HOOK("AnonymousIteratorV2", kernel);
NPU_REGISTER_FALLBACK_HOOK("AnonymousIterator", kernel);
NPU_REGISTER_FALLBACK_HOOK("AnonymousMultiDeviceIterator", kernel);
NPU_REGISTER_FALLBACK_HOOK("IteratorV2", kernel);
NPU_REGISTER_FALLBACK_HOOK("Iterator", kernel);
NPU_REGISTER_FALLBACK_HOOK("MultiDeviceIterator", kernel);
}  // namespace npu