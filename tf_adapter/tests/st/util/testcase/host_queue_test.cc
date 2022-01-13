#include "tensorflow/core/framework/tensor.h"
#include "tf_adapter/util/npu_plugin.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/host_queue.h"
#include "gtest/gtest.h"
#include <stdlib.h>

namespace tensorflow {
namespace {
class HostQueueTest : public testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(HostQueueTest, HostQueueSendData) {
  std::string name = "host_queue_001";
  uint32_t depth = 128U;
  uint32_t queue_id = 0U;
  TF_CHECK_OK(HostQueueInit(name, depth, queue_id));
  Tensor a(DT_UINT32, TensorShape({2, 2}));
  a.flat<uint32_t>()(0) = 1;
  a.flat<uint32_t>()(1) = 1;
  a.flat<uint32_t>()(2) = 1;
  a.flat<uint32_t>()(3) = 1;
  std::vector<Tensor> tensors{a};
  void *buff = nullptr;
  TF_CHECK_OK(MappingTensor2Buff(ACL_TENSOR_DATA_TENSOR, tensors, buff));
  bool need_resend = false;
  TF_CHECK_OK(HostQueueSendData(queue_id, buff, need_resend));
  HostQueueDestroy(queue_id);
}

TEST_F(HostQueueTest, HostQueueEndOfSequence) {
  std::string name = "host_queue_001";
  uint32_t depth = 128U;
  uint32_t queue_id = 0U;
  TF_CHECK_OK(HostQueueInit(name, depth, queue_id));
  void *buff = nullptr;
  TF_CHECK_OK(MappingTensor2Buff(ACL_TENSOR_DATA_TENSOR, {}, buff));
  bool need_resend = false;
  TF_CHECK_OK(HostQueueSendData(queue_id, buff, need_resend));
  HostQueueDestroy(queue_id);
}
}
} // end tensorflow