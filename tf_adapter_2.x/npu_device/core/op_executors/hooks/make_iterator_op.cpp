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

#include "tensorflow/core/graph/algorithm.h"

#include "op_executors/npu_kernel_registry.h"
#include "npu_utils.h"

namespace npu {
namespace {
class MakeIteratorGraphBuilder {
 public:
  static tensorflow::GraphDef GetGraph(const std::string &container_name, const std::string &shared_name,
                                       const TensorPartialShapes &shapes, const TensorDataTypes &types,
                                       TF_Status *status) {
    tensorflow::GraphDef gdef;

    tensorflow::Graph graph(tensorflow::OpRegistry::Global());
    tensorflow::Node *device_queue;
    tensorflow::Node *make_iterator;
    tensorflow::Node *iterator_v2;
    NPU_CTX_REQUIRES_OK_RETURN(status,
                               tensorflow::NodeBuilder("DeviceQueue_" + shared_name, "DeviceQueueDataset")
                                 .Attr("channel_name", shared_name)
                                 .Attr("output_types", types)
                                 .Attr("output_shapes", shapes)
                                 .Attr("_iterator_name", shared_name)
                                 .Attr("_tf_version", "tf2")
                                 .Finalize(&graph, &device_queue),
                               gdef);
    NPU_CTX_REQUIRES_OK_RETURN(status,
                               tensorflow::NodeBuilder("IteratorV2_" + shared_name, "IteratorV2")
                                 .Attr("container", container_name)
                                 .Attr("shared_name", shared_name)
                                 .Attr("output_types", types)
                                 .Attr("output_shapes", shapes)
                                 .Finalize(&graph, &iterator_v2),
                               gdef);
    NPU_CTX_REQUIRES_OK_RETURN(status,
                               tensorflow::NodeBuilder("InitMakeIterator_" + shared_name, "MakeIterator")
                                 .Attr("_kernel", "dp")
                                 .Attr("_iterator_name", shared_name)
                                 .Input(device_queue, 0)
                                 .Input(iterator_v2, 0)
                                 .Finalize(&graph, &make_iterator),
                               gdef);

    if (kDumpExecutionDetail || kDumpGraph) {
      std::string file_name = "dp_init_" + shared_name + ".inner.pbtxt";
      DLOG() << "NPU Dump mirrored resource init graph inner graph to: " << file_name;
      (void)WriteTextProto(tensorflow::Env::Default(), file_name, graph.ToGraphDefDebug());
    }

    // Tensorflow model parser bug，如果名字不是dpop开头的，则会被remove掉
    std::string func_name = "dpop_init_func_" + shared_name;
    tensorflow::FunctionDefLibrary fdef_lib;
    tensorflow::FunctionDef *fdef = fdef_lib.add_function();
    (void)tensorflow::GraphToFunctionDef(graph, func_name, fdef);

    tensorflow::Graph dpop_graph(tensorflow::OpRegistry::Global());

    tensorflow::AttrValue function_attr;
    function_attr.mutable_func()->set_name(func_name);

    tensorflow::Node *dpop_node;
    NPU_CTX_REQUIRES_OK_RETURN(status,
                               tensorflow::NodeBuilder(func_name, "DPOP")
                                 .Input(std::vector<tensorflow::NodeBuilder::NodeOut>{})
                                 .Attr("Tin", tensorflow::DataTypeVector{})
                                 .Attr("Tout", tensorflow::DataTypeVector{})
                                 .Attr("function", function_attr)
                                 .Finalize(&dpop_graph, &dpop_node),
                               gdef);
    AssembleOpDef(dpop_node);
    dpop_node->AddAttr("func_def", fdef_lib.SerializeAsString());
    (void)tensorflow::FixupSourceAndSinkEdges(&dpop_graph);
    dpop_graph.ToGraphDef(&gdef);
    return gdef;
  }
};
}  // namespace

static const auto kernel = [](TFE_Context *context, NpuDevice *dev, const tensorflow::NodeDef &ndef, int num_inputs,
                              TFE_TensorHandle **inputs, int num_outputs, TFE_TensorHandle **outputs,
                              TF_Status *status) {
  (void)ndef;
  (void)num_outputs;
  (void)outputs;
  for (int j = 0; j < num_inputs; ++j) {
    TFE_TensorHandle *input = inputs[j];
    if (tensorflow::unwrap(input)->DataType() == tensorflow::DT_RESOURCE) {
      const tensorflow::Tensor *tensor;
      NPU_CTX_REQUIRES_OK(status, GetTensorHandleTensor(input, &tensor));
      auto handle = tensor->scalar<tensorflow::ResourceHandle>()();
      if (!dev->MirroredIterator(handle)) {
        DLOG() << "Skip create provider as iterator resource not mirrored " << handle.DebugString();
        return;
      }
      TensorPartialShapes shapes;
      TensorDataTypes types;
      NPU_CTX_REQUIRES_OK(status, dev->GetMirroredIteratorShapesAndTypes(handle, shapes, types));
      auto dp_init_graph = MakeIteratorGraphBuilder::GetGraph(handle.container(), npu::WrapResourceName(handle.name()),
                                                              shapes, types, status);
      NPU_REQUIRES_TFE_OK(status);
      if (kDumpExecutionDetail || kDumpGraph) {
        std::string file_name = "dp_init_" + handle.name() + ".pbtxt";
        DLOG() << "NPU Dump mirrored resource init graph to: " << file_name;
        (void)WriteTextProto(tensorflow::Env::Default(), file_name, dp_init_graph);
      }
      // 针对推荐网络，Provider需要支持1对N的传输，默认只向资源所处的Device发送
      dev->CreateIteratorProvider(context, tensor, {dev->device_id}, status);
      NPU_REQUIRES_TFE_OK(status);

      dev->RunGeGraphPin2CpuAnonymous(context, "dp_init_" + handle.name(), dp_init_graph, num_inputs, inputs, 0,
                                      nullptr, status);
      NPU_REQUIRES_TFE_OK(status);
    }
  }
};

NPU_REGISTER_FALLBACK_HOOK("MakeIterator", kernel);
NPU_REGISTER_FALLBACK_HOOK("MultiDeviceIteratorInit", kernel);
}  // namespace npu
