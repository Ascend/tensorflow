# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

## @file hccl_ops.py
# HCCL 算子API

from tensorflow.contrib.util import loader
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
from npu_bridge.helper import helper

gen_hccl_ops = helper.get_gen_ops();


## 提供group内的集合通信allreduce功能
#  @param tensor tensorflow的tensor类型，allreduce操作的输入；
#  @param reduction string类型，reduce的操作类型，可以为”max”,”min”,”prod”和”sum”;
#  @param fusion int类型，算子融合标识。0: 不融合；1: 按照梯度切分设置融合，默认融； 2: 按照相同fusion_id融合。
#  @param fusion_id int类型，算子融合索引标识，相同fusion_id的算子将会融合。
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
#  @return 对输入tensor执行完allreduce操作之后的结果tensor
def allreduce(tensor, reduction, fusion=1, fusion_id=-1, group="hccl_world_group"):
    result = gen_hccl_ops.hcom_all_reduce(
        input=tensor,
        reduction=reduction,
        fusion=fusion,
        fusion_id=fusion_id,
        group=group)
    return result


@ops.RegisterGradient('HcomAllReduce')
def _allreduce_grad(op, grad):
    return allreduce(grad, "sum", fusion=0)


## 提供group内的集合通信allgather功能
#  @param tensor tensorflow的tensor类型，allgather操作的输入；
#  @param rank_size int类型，group内device的数量;
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
#  @return 对输入tensor执行完allgather操作之后的结果tensor
def allgather(tensor, rank_size, group="hccl_world_group"):
    result = gen_hccl_ops.hcom_all_gather(
        input=tensor,
        group=group,
        rank_size=rank_size)
    return result


## 提供group内的集合通信broadcast功能
#  @param tensor tensorflow的tensor类型，broadcast操作的输入；
#  @param root_rank int类型，作为root节点的rank_id，该id是group内的rank id;
#  @param fusion int类型，算子融合标识。0: 不融合；2:按照相同fusion_id融合;其他值非法。
#  @param fusion_id int类型，算子融合索引标识，相同fusion_id的算子将会融合。
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
#  @return 对输入tensor执行完broadcast操作之后的结果tensor
def broadcast(tensor, root_rank, fusion=2, fusion_id=0, group="hccl_world_group"):
    result = gen_hccl_ops.hcom_broadcast(
        input=tensor,
        fusion=fusion,
        fusion_id=fusion_id,
        group=group,
        root_rank=root_rank)
    return result

## 提供group内的集合通信reduce功能
#  @param tensor tensorflow的tensor类型，reduce操作的输入；
#  @param reduction string类型，reduce的操作类型，可以为”max”,”min”,”prod”和”sum”;
#  @param fusion int类型，算子融合标识。0: 不融合； 2: 按照相同fusion_id融合。
#  @param fusion_id int类型，算子融合索引标识，相同fusion_id的算子将会融合。
#  @param root_rank int类型，作为root节点的rank_id，该id是group内的rank id;
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
#  @return 对输入tensor执行完reduce操作之后的结果tensor
def reduce(tensor, reduction, root_rank, fusion=0, fusion_id=-1, group="hccl_world_group"):
    result = gen_hccl_ops.hcom_reduce(
        input=tensor,
        reduction=reduction,
        fusion=fusion,
        fusion_id=fusion_id,
        group=group,
        root_rank=root_rank)
    return result


## 提供group内的集合通信reduce_scatter功能
#  @param tensor tensorflow的tensor类型，reduce_scatter操作的输入；
#  @param reduction string类型，reduce的操作类型，可以为”max”,”min”,”prod”和”sum”;
#  @param rank_size int类型，group内device的数量;
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
#  @return 对输入tensor执行完reduce_scatter操作之后的结果tensor
def reduce_scatter(tensor, reduction, rank_size, group="hccl_world_group"):
    result = gen_hccl_ops.hcom_reduce_scatter(
        input=tensor,
        reduction=reduction,
        group=group,
        rank_size=rank_size)
    return result


## 提供group内的集合通信send功能
#  @param tensor tensorflow的tensor类型，send操作的输入；
#  @param sr_tag int类型，消息标签，相同sr_tag的send/recv对可以收发数据;
#  @param dest_rank int类型，数据的目标节点，该rank是group中的rank id;
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
def send(tensor, sr_tag, dest_rank, group="hccl_world_group"):
    result = gen_hccl_ops.hcom_send(
        input=tensor,
        group=group,
        sr_tag=sr_tag,
        dest_rank=dest_rank)
    return result


## 提供group内的集合通信receive功能
#  @param shape 接收tensor的shape；
#  @param data_type 接收tensor的数据类型；
#  @param sr_tag int类型，消息标签，相同sr_tag的send/recv对可以收发数据;
#  @param dest_rank int类型，数据的目标节点，该rank是group中的rank id;
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
def receive(shape, data_type, sr_tag, src_rank, group="hccl_world_group"):
    result = gen_hccl_ops.hcom_receive(
        shape=shape,
        T=data_type,
        group=group,
        sr_tag=sr_tag,
        src_rank=src_rank)
    return result

## 提供remote read功能
#  @param remote 远端内存信息，shape(index_num, 3)：[u64 remoteId, u64 remoteAddr, u64 dataLength]
#  @param data_type 接收tensor的数据类型
#  @return 本端接收内存 shape(index_num, dataLength/sizeof(data_type))
def remote_read(tensorRemote, data_type):
    result = gen_hccl_ops.hcom_remote_read(
        remote=tensorRemote,
        dtype=data_type)
    return result

##提供remote ref read功能
#  @param tensorRemote 远端内存信息，shape(index_num, 3)：[u64 remoteId, u64 remoteAddr, u64 dataLength]
#  @param cache 本端接收内存基地址
#  @param offset 进行跳读的步长
def remote_ref_read(tensorRemote, cache, offset):
    result=gen_hccl_ops.hcom_remote_ref_read(
        remote=tensorRemote,
        cache_var=cache,
        local_offset=offset)
    return result

## 提供remote write功能
#  @param remote 写入远端内存信息，shape(index_num, 3)：[u64 remoteId, u64 remoteAddr, u64 dataLength]
#  @param local 本端发送内存
def remote_write(tensorRemote, tensorLocal, data_type):
    result = gen_hccl_ops.hcom_remote_write(
        remote=tensorRemote,
        local=tensorLocal)
    return result

##提供remote scatter write功能
#  @param tensorRemote 写入远端内存信息，shape(index_num, 3)：[u64 remoteId, u64 remoteAddr, u64 dataLength]
#  @param tensorLocal 本端发送内存基地址
#  @param offset 进行跳写的步长
def remote_scatter_write(tensorRemote, tensorLocal, offset):
    result = gen_hccl_ops.hcom_remote_scatter_write(
        remote=tensorRemote,
        local=tensorLocal,
        local_offset=offset)
    return result

##提供gather-alltoallv功能
#  @param addrinfo 需要读取的地址信息，shape(index_num, 2)：[u64 remoteAddr, u64 dataLength];
#  @param addrinfo_count_per_rank 向每个rank上读取的数据地址信息数，shape(rank_size, );
#  @param recv_counts 从每个rank上接收的数据size，shape(rank_size, 1);
#  @param recv_displacements 从每个rank接收数据起始位置相对于output起始的偏移，shape(rank_size, );
#  @param dtype 读取对象的数据类型;
#  @param addr_length u64 类型，从每个地址信息读取的字节数，如不相等设为-1，如相等且实际长度未知则设为-2，如相等且实际长度已知则设为实际长度;
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
def gather_all_to_all_v(addrinfo, addrinfo_count_per_rank, recv_counts, recv_displacements, dtype, addr_length, group="hccl_world_group"):
    result = gen_hccl_ops.hcom_gather_all_to_all_v(
        addrinfo=addrinfo,
        addrinfo_count_per_rank=addrinfo_count_per_rank,
        recv_counts=recv_counts,
        recv_displacements=recv_displacements,
        dtype=dtype,
        addr_length=addr_length,
        group=group)
    return result

##提供alltoallv功能
#  @param send_data 需要发送的数据;
#  @param send_counts 给每个rank发送的数据size，shape(rank_size, );
#  @param send_displacements 给每个rank发送数据起始位置相对于send_data起始的偏移，shape(rank_size, );
#  @param recv_counts 从每个rank上接收的数据size，shape(rank_size, 1);
#  @param recv_displacements 从每个rank接收数据起始位置相对于output起始的偏移，shape(rank_size, );
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
def all_to_all_v_dynamic(send_data, send_counts, send_displacements, recv_counts, recv_displacements, group="hccl_world_group"):
    result = gen_hccl_ops.hcom_all_to_all_v_dynamic(
        send_data=send_data,
        send_counts=send_counts,
        send_displacements=send_displacements,
        recv_counts=recv_counts,
        recv_displacements=recv_displacements,
        group=group)
    return result