#include <dataset_function.h>
#include "tf_adapter/util/npu_attrs.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include <stdlib.h>
#include "gtest/gtest.h"

namespace tensorflow {
namespace {
class MbufAllocatorTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
  const int64_t kRuntimeTensorDescSize = 1024UL;
  const size_t alignment = 64;
};

TEST_F(MbufAllocatorTest, EnableMbufAllocatorTest) {
  tensorflow::int64 enable_mbuf_allocator = 0;
  (void)tensorflow::ReadInt64FromEnvVar("ENABLE_MBUF_ALLOCATOR", 0, &enable_mbuf_allocator);
  if (enable_mbuf_allocator == 1) {
    Allocator* a = cpu_allocator();
    EXPECT_EQ(a->Name(), "MbufAllocator");

    void* raw_ptr = a->AllocateRaw(alignment, kRuntimeTensorDescSize);
    auto *alloc_ptr = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(raw_ptr) -
                                                  kRuntimeTensorDescSize);
    for (int i = 0; i < kRuntimeTensorDescSize / sizeof (int32_t); i++) {
      int32_t alloc_ptr_i = alloc_ptr[i];
      ADP_LOG(INFO) << i << " " << alloc_ptr_i; // no dump
    }
    a->DeallocateRaw(raw_ptr);
    unsetenv("ENABLE_MBUF_ALLOCATOR");
  }
}
}
} //end tensorflow
