#include "tf_adapter/common/graphcycles.h"
#include "gtest/gtest.h"
#include <stdlib.h>

namespace tensorflow {
namespace {
class GraphCyclesTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};
TEST_F(GraphCyclesTest, DebugStringTest) {
  GraphCycles graph_cycles;
  EXPECT_EQ(graph_cycles.DebugString(), "digraph {\n}\n");
}
}
} // end tensorflow