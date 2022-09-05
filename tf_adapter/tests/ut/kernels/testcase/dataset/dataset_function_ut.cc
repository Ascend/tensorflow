#include <stdlib.h>
#include <vector>

#include "securec.h"
#include "gtest/gtest.h"
#include "tf_adapter/common/adp_logger.h"
#define private public
#include "tf_adapter/kernels/aicpu/dataset_function.h"

namespace tensorflow {
namespace data {
namespace {

TEST(DatasetFunctionTest, MakeCompatShapeTest) {
  ADP_LOG(INFO) << "====== DatasetFunctionTest MakeCompatShapeTest ======";

  std::map<std::string, std::string> init_options;
  init_options["key"] = "value";
  DataTypeVector input_types;
  DataTypeVector output_types;
  std::vector<PartialTensorShape> input_shape;
  std::vector<PartialTensorShape> output_shape;
  DatasetFunction df = DatasetFunction(init_options, "func_name", input_types, output_types,
                                           input_shape, output_shape);
  PartialTensorShape shape_a = PartialTensorShape({3, 3});
  PartialTensorShape shape_b = PartialTensorShape({3, 3});
  PartialTensorShape ret = df.MakeCompatShape(shape_a, shape_b);
  int expected_dims = shape_a.dims();
  ADP_LOG(INFO) << "Result dims is " << ret.dims();
  EXPECT_EQ(expected_dims, ret.dims());

  PartialTensorShape shape_c = PartialTensorShape({3});
  PartialTensorShape kUnknownRankShape = PartialTensorShape();
  ret = df.MakeCompatShape(shape_a, shape_c);
  ADP_LOG(INFO) << "Result dims is " << ret.dims();
  EXPECT_EQ(kUnknownRankShape.dims(), ret.dims());
}
}  // namespace
}  // namespace data
}  // namespace tensorflow
