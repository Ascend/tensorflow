from tensorflow.contrib.util import loader
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
from npu_bridge.helper import helper

gen_npu_cpu_ops = helper.get_gen_ops();


## 提供embeddingrankid功能
#  @param addr_tensor tensorflow的tensor类型，embeddingrankid操作的输入；
#  @param index tensorflow的tensor类型，embeddingrankid操作的输入；
#  @param row_memory int类型，一行数据存储的大小 默认为320。
#  @param mode string类型，embeddingrankid的操作类型，可以为”mod”,”order”;数据存储的方式。
#  @return 对输入addr_tensor，index_tensor执行完embeddingrankid操作之后的结果tensor
def embeddingrankid(addr_tensor, index, row_memory=320, mode='mod'):
    result = gen_npu_cpu_ops.embedding_rank_id(
        addr_table=addr_tensor,
        index=index,
        row_memory=row_memory,
        mode=mode)
    return result

## 提供RandomChoiceWithMask功能
#  @param x bool 类型
#  @param count int 类型
#  @param seed int类型
#  @param seed2 int类型
#  @return y int32类型 mask bool 类型
def randomchoicewithmask(x, count, seed=0, seed2=0):
    result = gen_npu_cpu_ops.random_choice_with_mask(
        x=x,
        count=count,
        seed=seed,
        seed2=seed2)
    return result

## 提供DenseImageWarp功能
#  @param image tensor类型
#  @param flow tensor类型
#  @return y tensor类型
def dense_image_warp(image, flow, name=None):
    result = gen_npu_cpu_ops.dense_image_warp(
        image=image,
        flow=flow,
        name=name
    )
    return result

## DenseImageWarp的梯度函数
@ops.RegisterGradient("DenseImageWarp")
def dense_image_warp_grad(op, grad):
    image = op.inputs[0]
    flow = op.inputs[1]
    grad_image, grad_flow = gen_npu_cpu_ops.dense_image_warp_grad(
        grad, image, flow)
    return [grad_image, grad_flow]
