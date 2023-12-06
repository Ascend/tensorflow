#include "tf_adapter/util/npu_attrs.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/version.h"
#include <stdlib.h>
#include "gtest/gtest.h"

namespace tensorflow {
namespace {

#define TF_ASSERT_OK(statement) \
  ASSERT_EQ(::tensorflow::Status::OK(), (statement))

#define TF_EXPECT_OK(statement) \
  EXPECT_EQ(::tensorflow::Status::OK(), (statement))

class DummyDevice : public DeviceBase {
 public:
  DummyDevice(Env* env, bool save) : DeviceBase(env), save_(save) {}
  bool RequiresRecordingAccessedTensors() const override { return save_; }
 private:
  bool save_;
};
}
class HostFeatureMappingTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

FakeInputFunctor FakeHostInputStub(DataType dt) {
  return [dt](const OpDef &op_def, int in_index, const NodeDef &node_def,
              NodeDefBuilder *builder) {
    char c = 'a' + (in_index % 26);
    string in_node = string(&c, 1);
    builder->Input(in_node, 0, dt);
    return Status::OK();
  };
}

PartialTensorShape THostShape(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}

TEST(HostFeatureMappingTest, HostFeatureMappingTestShapeInference) {
  const OpRegistrationData *reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("FeatureMapping", &reg));
  OpDef op_def = reg->op_def;
  NodeDef def;
  int threshold = 1;
  std::string table_name = "table_name1";
  TF_CHECK_OK(NodeDefBuilder("dummy", &op_def)
                  .Input(FakeHostInputStub(DT_INT32))
                  .Attr("threshold", threshold)
                  .Attr("table_name", table_name)
                  .Finalize(&def));
  shape_inference::InferenceContext c(0, &def, op_def, {THostShape({1})}, {}, {}, {});
  TF_CHECK_OK(reg->shape_inference_fn(&c));
  ASSERT_EQ("[1]", c.DebugString(c.output(0)));
}
} // end tensorflow