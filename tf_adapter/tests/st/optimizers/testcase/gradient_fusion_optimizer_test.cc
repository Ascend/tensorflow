#include "gtest/gtest.h"
#include "tf_adapter/optimizers/grad_fusion_optimizer.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {
class GradientFusionOptimizerTest : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};
TEST_F(GradientFusionOptimizerTest, RunOptimizer) {
  GrapplerItem item;
  item.graph = GraphDef();
  GraphDef output;
  const Status status = GradFusionOptimizer().Optimize(nullptr, item, &output);
  EXPECT_EQ(status, Status::OK());
}
} // end grappler
} // end tensorflow