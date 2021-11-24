#include <memory>
#include "tf_adapter/kernels/aicpu/npu_cpu_ops.cc"
#include "gtest/gtest.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

PartialTensorShape TShape(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}

FakeInputFunctor FakeInputStub(DataType dt) {
  return [dt](const OpDef& op_def, int in_index, const NodeDef& node_def,
              NodeDefBuilder* builder) {
    char c = 'a' + (in_index % 26);
    string in_node =  string(&c, 1);
    builder->Input(in_node, 0, dt);
    return Status::OK();
  };
}

TEST(OCROpsTest, TestBatchEnqueue) {
    DataTypeSlice input_types({DT_INT32, DT_INT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_INT32});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    BatchEnqueueOp cache(context);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(OCROpsTest, TestOCRRecognitionPreHandle) {
    DataTypeSlice input_types({DT_UINT8, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_UINT8, DT_INT32, DT_INT32});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    OCRRecognitionPreHandleOp cache(context);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(OCROpsTest, TestOCRDetectionPreHandle) {
    DataTypeSlice input_types({DT_UINT8});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_UINT8, DT_FLOAT, DT_FLOAT});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    OCRDetectionPreHandleOp cache(context);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(OCROpsTest, TestOCRIdentifyPreHandle) {
    DataTypeSlice input_types({DT_UINT8, DT_INT32, DT_INT32});
    MemoryTypeSlice input_memory_types;
    DataTypeSlice output_types({DT_UINT8});
    MemoryTypeSlice output_memory_types;
    DeviceBase *device = new DeviceBase(Env::Default());
    NodeDef *node_def = new NodeDef();
    OpDef *op_def = new OpDef();
    OpKernelConstruction *context = new OpKernelConstruction(DEVICE_CPU, device, nullptr, node_def, op_def, nullptr,
                                                             input_types, input_memory_types, output_types, output_memory_types,
                                                             1, nullptr);
    OCRIdentifyPreHandleOp cache(context);
    delete device;
    delete node_def;
    delete op_def;
    delete context;
}

TEST(OCROpsTest, TestBatchEnqueueShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("BatchEnqueue", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_INT32)
                  .Attr("batch_size", 8)
                  .Attr("queue_name", "TEST")
                  .Attr("pad_mode", "REPLICATE")
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_UINT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({5}), TShape({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(OCROpsTest, TestOCRRecognitionPreHandleShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRRecognitionPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("T", DT_FLOAT)
                  .Attr("batch_size", 8)
                  .Attr("data_format", "NHWC")
                  .Attr("pad_mode", "REPLICATE")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({3}), TShape({3}), TShape({3}), TShape({3}), TShape({3})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(OCROpsTest, TestOCRDetectionPreHandleShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRDetectionPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("data_format", "NHWC")
                  .Input(FakeInputStub(DT_UINT8))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({5,5,3})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(OCROpsTest, TestOCRDetectionPreHandleShapeInference1) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRDetectionPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("data_format", "NCHW")
                  .Input(FakeInputStub(DT_UINT8))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({3,5,5})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(OCROpsTest, TestOCRDetectionPreHandleShapeInferenceFail1) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRDetectionPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("data_format", "NHWC")
                  .Input(FakeInputStub(DT_UINT8))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({5,5,4})}, {}, {}, {});
  ASSERT_TRUE(!reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRDetectionPreHandleShapeInferenceFail2) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRDetectionPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("data_format", "NCHW")
                  .Input(FakeInputStub(DT_UINT8))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({4,5,5})}, {}, {}, {});
  ASSERT_TRUE(!reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRDetectionPreHandleShapeInferenceFail3) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRDetectionPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("data_format", "NCHW")
                  .Input(FakeInputStub(DT_UINT8))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({3,5,5, 1})}, {}, {}, {});
  ASSERT_TRUE(!reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRIdentifyPreHandleShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRIdentifyPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("size", {1,2})
                  .Attr("data_format", "NHWC")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({-1}), TShape({-1}), TShape({-1, 3})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(OCROpsTest, TestOCRIdentifyPreHandleShapeInference1) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRIdentifyPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("size", {1,2})
                  .Attr("data_format", "NCHW")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({-1}), TShape({-1}), TShape({-1, 3})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(OCROpsTest, TestOCRIdentifyPreHandleShapeInferencefail1) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRIdentifyPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("size", {1,2})
                  .Attr("data_format", "NCHW")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({-1, 1}), TShape({-1}), TShape({-1, 3})}, {}, {}, {});
  ASSERT_TRUE(!reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRIdentifyPreHandleShapeInferencefail2) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRIdentifyPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("size", {1,2})
                  .Attr("data_format", "NCHW")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({-1}), TShape({-1, -1}), TShape({-1, 3})}, {}, {}, {});
  ASSERT_TRUE(!reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestOCRIdentifyPreHandleShapeInferencefail3) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRIdentifyPreHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("size", {1,2})
                  .Attr("data_format", "NCHW")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({-1}), TShape({-1}), TShape({-1, 4})}, {}, {}, {});
  ASSERT_TRUE(!reg->shape_inference_fn(&c).ok());
}

TEST(OCROpsTest, TestBatchDilatePolysShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("BatchDilatePolys", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)                  
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({1}), TShape({1}), TShape({1}), TShape({1}),TShape({1}),TShape({1}),TShape({1})},{}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(OCROpsTest, TestOCRFindContoursShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRFindContours", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)                  
                  .Attr("value_mode", 0)
                  .Input(FakeInputStub(DT_UINT8))                  
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,
    {TShape({2})},{}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(OCROpsTest, TestDequeueShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("Dequeue", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("queue_name", "TEST")
                  .Attr("output_type", DT_UINT8)
                  .Attr("output_shape", {2})
                  .Input(FakeInputStub(DT_UINT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(OCROpsTest, TestOCRDetectionPostHandleShapeInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("OCRDetectionPostHandle", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Attr("data_format", "NHWC")
                  .Input(FakeInputStub(DT_UINT8))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({3}), TShape({3}), TShape({3}), TShape({3})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}

TEST(OCROpsTest, TestResizeAndClipPolysInference) {
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("ResizeAndClipPolys", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_FLOAT))
                  .Input(FakeInputStub(DT_INT32))
                  .Input(FakeInputStub(DT_INT32))
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def,{TShape({})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
}
}  // namespace
}  // namespace tensorflow