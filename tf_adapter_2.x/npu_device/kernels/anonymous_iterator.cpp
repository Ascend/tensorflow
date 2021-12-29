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

#include "npu_custom_kernel.h"

static auto kernel = [](TFE_Context *context, NpuDevice *dev, const char *op_name, const TFE_OpAttrs *attributes,
                        int num_inputs, TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs,
                        TF_Status *status) {
  for (int i = 0; i < num_outputs; ++i) {
    TFE_TensorHandle *retval = outputs[i];
    if (npu::UnwrapHandle(retval)->DataType() == tensorflow::DT_RESOURCE) {
      const tensorflow::Tensor *tensor;
      NPU_CTX_REQUIRES_OK(status, npu::UnwrapTensor(retval, &tensor));
      std::vector<tensorflow::PartialTensorShape> vec_shapes;
      TensorPartialShapes shapes;
      TensorDataTypes types;
      tensorflow::NodeDef ndef;
      tensorflow::unwrap(attributes)->FillAttrValueMap(ndef.mutable_attr());
      NPU_CTX_REQUIRES_OK(status, tensorflow::GetNodeAttr(ndef, "output_shapes", &vec_shapes));
      NPU_CTX_REQUIRES_OK(status, tensorflow::GetNodeAttr(ndef, "output_types", &types));
      for (const auto &shape : vec_shapes) {
        shapes.push_back(shape);
      }
      auto resource = tensor->scalar<tensorflow::ResourceHandle>()();
      DLOG() << "Record mirrored host resource " << resource.DebugString();
      dev->RecordIteratorMirror(resource, shapes, types);
    }
  }
};

namespace npu {
NPU_REGISTER_FALLBACK_HOOK("AnonymousIteratorV2", kernel);
NPU_REGISTER_FALLBACK_HOOK("AnonymousIterator", kernel);
NPU_REGISTER_FALLBACK_HOOK("AnonymousMultiDeviceIterator", kernel);
NPU_REGISTER_FALLBACK_HOOK("IteratorV2", kernel);
NPU_REGISTER_FALLBACK_HOOK("Iterator", kernel);
NPU_REGISTER_FALLBACK_HOOK("MultiDeviceIterator", kernel);
}