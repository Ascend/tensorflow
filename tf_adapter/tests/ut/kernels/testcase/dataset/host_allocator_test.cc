#include "tf_adapter/util/host_allocator.h"
#include "gtest/gtest.h"

namespace tensorflow {
class HostAllocatorTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(HostAllocatorTest, allocate_test)  {
  int64_t a = 10;
  HostAllocator host_allocat(static_cast<void *>(&a));
  std::string name = host_allocat.Name();
  EXPECT_EQ(name, "host_allocator");
  void *ptr = host_allocat.AllocateRaw(0, 0);
}
} //end tensorflow