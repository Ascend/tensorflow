#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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

import json
from tensorflow.python.framework import ops
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.util import compat
import tensorflow.compat.v1 as tf
from npu_bridge.helper import helper

gen_npu_cpu_ops = helper.get_gen_ops()


## 提供embeddingrankid功能
#  @param addr_tensor tensorflow的tensor类型，embeddingrankid操作的输入；
#  @param index tensorflow的tensor类型，embeddingrankid操作的输入；
#  @param row_memory int类型，一行数据存储的大小 默认为320。
#  @param mode string类型，embeddingrankid的操作类型，可以为”mod”,”order”;数据存储的方式。
#  @return 对输入addr_tensor，index_tensor执行完embeddingrankid操作之后的结果tensor
def embeddingrankid(addr_tensor, index, row_memory=320, mode='mod'):
    """ Embed rank index. """
    result = gen_npu_cpu_ops.embedding_rank_id(
        addr_table=addr_tensor,
        index=index,
        row_memory=row_memory,
        mode=mode)
    return result


## 提供embeddinglocalindex功能
#  @param addr_tensor tensorflow的tensor类型，embeddinglocalindex操作的输入；
#  @param index tensorflow的tensor类型，embeddinglocalindex操作的输入；
#  @param row_memory int类型，一行数据存储的大小 默认为320。
#  @param mode string类型，embeddinglocalindex的操作类型，可以为”mod”,”order”;数据存储的方式。
#  @return 对输入addr_tensor，index_tensor执行完embeddinglocalindex操作之后的结果tensor
def embedding_local_index(addr_tensor, index, row_memory=320, mode='mod'):
    """ Embed local index. """
    result = gen_npu_cpu_ops.embedding_local_index(
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
    """ Random choice with mask. """
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
    """ Dense image warp. """
    result = gen_npu_cpu_ops.dense_image_warp(
        image=image,
        flow=flow,
        name=name
    )
    return result


## DenseImageWarp的梯度函数
@ops.RegisterGradient("DenseImageWarp")
def dense_image_warp_grad(op, grad):
    """ Dense image warp grad. """
    image = op.inputs[0]
    flow = op.inputs[1]
    grad_image, grad_flow = gen_npu_cpu_ops.dense_image_warp_grad(
        grad, image, flow)
    return [grad_image, grad_flow]


## 提供BatchEnqueue功能
#  @param x uint8 类型
#  @param queue_id uint32 类型
#  @param batch_size int 类型
#  @param queue_name string 类型
#  @param queue_depth int64 类型
#  @param pad_mode string 类型
#  @return enqueue_count int64类型
def batch_enqueue(x, queue_id=0, batch_size=8, queue_name="", queue_depth=100, pad_mode="REPLICATE"):
    """ Batch enqueue. """
    result = gen_npu_cpu_ops.batch_enqueue(
        x=x,
        queue_id=queue_id,
        batch_size=batch_size,
        queue_name=queue_name,
        queue_depth=queue_depth,
        pad_mode=pad_mode)
    return result


## 提供OCRRecognitionPreHandle功能
#  @param imgs_data uint8 类型
#  @param imgs_offset int32 类型
#  @param imgs_size int32 类型
#  @param langs int32 类型
#  @param langs_score int32 类型
#  @param batch_size int 类型
#  @param data_format string 类型
#  @param pad_mode string 类型
#  @return imgs,imgs_relation,imgs_lang,imgs_piece_fillers uint8,int32,int32,int32 类型
def ocr_recognition_pre_handle(imgs_data, imgs_offset, imgs_size, langs, langs_score, \
                               batch_size=8, data_format="NHWC", pad_mode="REPLICATE"):
    """ Recognize ocr pre-handle. """
    result = gen_npu_cpu_ops.ocr_recognition_pre_handle(
        imgs_data=imgs_data,
        imgs_offset=imgs_offset,
        imgs_size=imgs_size,
        langs=langs,
        langs_score=langs_score,
        batch_size=batch_size,
        data_format=data_format,
        pad_mode=pad_mode)
    return result


## 提供OCRDetectionPreHandle功能
#  @param img uint8 类型
#  @param data_format string 类型
#  @return resized_img,h_scale,w_scale uint8,float32,float32 类型
def ocr_detection_pre_handle(img, data_format="NHWC"):
    """
    ocr detection pre-handle
    """
    result = gen_npu_cpu_ops.ocr_detection_pre_handle(
        img=img,
        data_format=data_format)
    return result


## 提供OCRIdentifyPreHandle功能
#  @param imgs_data uint8 类型
#  @param imgs_offset int32 类型
#  @param imgs_size int32 类型
#  @param size list(int) 类型
#  @param data_format string 类型
#  @return resized_imgs, uint8 类型
def ocr_identify_pre_handle(imgs_data, imgs_offset, imgs_size, size, data_format="NHWC"):
    """ Ocr identification pre-handle. """
    result = gen_npu_cpu_ops.ocr_identify_pre_handle(
        imgs_data=imgs_data,
        imgs_offset=imgs_offset,
        imgs_size=imgs_size,
        size=size,
        data_format=data_format)
    return result


## 提供BatchDilatePolys功能
#  @param polys_data int32 类型
#  @param polys_offset int32 类型
#  @param polys_size int32 类型
#  @param score float 类型
#  @param min_border int32 类型
#  @param min_area_thr int32 类型
#  @param score_thr float 类型
#  @param expand_scale float 类型
#  @return dilated_polys_data int32 类型
#  @return dilated_polys_offset int32 类型
#  @return dilated_polys_size int32 类型
def batch_dilate_polys(polys_data, polys_offset, polys_size, score, \
                       min_border, min_area_thr, score_thr, expand_scale):
    """ Batch dilate poly. """
    result = gen_npu_cpu_ops.batch_dilate_polys(
        polys_data=polys_data,
        polys_offset=polys_offset,
        polys_size=polys_size,
        score=score,
        min_border=min_border,
        min_area_thr=min_area_thr,
        score_thr=score_thr,
        expand_scale=expand_scale)
    return result


## 提供OCRFindContours功能
#  @param img uint8 类型
#  @param value_mode int 类型
#  @return polys_data int32 类型
#  @return polys_offset int32 类型
#  @return polys_size int32 类型
def ocr_find_contours(img, value_mode=0):
    """ Ocr find contours. """
    result = gen_npu_cpu_ops.ocr_find_contours(img=img, value_mode=value_mode)
    return result


## 提供Dequeue功能
#  @param queue_id uint32 类型
#  @param output_type RealNumberType 类型
#  @param output_shape list(int) 类型
#  @param queue_name string 类型
#  @return data 根据output_type确定类型
def dequeue(queue_id, output_type, output_shape, queue_name=""):
    """ Dequeue. """
    result = gen_npu_cpu_ops.dequeue(
        queue_id=queue_id,
        output_type=output_type,
        output_shape=output_shape,
        queue_name=queue_name)
    return result


## 提供OCRDetectionPostHandle功能
#  @param img uint8 类型
#  @param polys_data int32 类型
#  @param polys_offset int32 类型
#  @param polys_size int32 类型
#  @param data_format string 类型
#  @return imgs_data,imgs_offset,imgs_size,rect_points uint8,int32,int32,int32 类型
def ocr_detection_post_handle(img, polys_data, polys_offset, polys_size, data_format="NHWC"):
    """ Orc detection post-handle. """
    result = gen_npu_cpu_ops.ocr_detection_post_handle(
        img=img,
        polys_data=polys_data,
        polys_offset=polys_offset,
        polys_size=polys_size,
        data_format=data_format)
    return result


## 提供WarpAffineV2功能
#  @param x uint8, float32 类型
#  @param matrix float32 类型
#  @param dst_size int32, int64 类型
#  @param interploation string 类型
#  @param border_type string 类型
#  @param border_value int 类型
#  @return y uint8, float32 类型
def warp_affine_v2(x, matrix, dst_size, interploation="INTEL_BILINEAR", border_type="BORDER_CONSTANT", border_value=0):
    """ Warp Affine V2. """
    result = gen_npu_cpu_ops.warp_affine_v2(
        x=x,
        matrix=matrix,
        dst_size=dst_size,
        interploation=interploation,
        border_type=border_type,
        border_value=border_value)
    return result


## 提供ResizeV2功能
#  @param x uint8, float32 类型
#  @param dst_size int32, int64 类型
#  @param interploation string 类型
#  @return y uint8, float32 类型
def resize_v2(x, dst_size, interploation="INTEL_BILINEAR"):
    """ Resize V2. """
    result = gen_npu_cpu_ops.resize_v2(
        x=x,
        dst_size=dst_size,
        interploation=interploation)
    return result


## 提供ResizeAndClipPolys功能
#  @param polys_data int32 类型
#  @param polys_offset int32 类型
#  @param polys_size int32 类型
#  @param h_scale float32 类型
#  @param w_scale float32 类型
#  @param img_h int32 类型
#  @param img_w int32 类型
#  @return clipped_polys_data,clipped_polys_offset,clipped_polys_size int32,int32,int32 类型
def resize_and_clip_polys(polys_data, polys_offset, polys_size, h_scale, w_scale, img_h, img_w):
    """ Resize and clip polys. """
    result = gen_npu_cpu_ops.resize_and_clip_polys(
        polys_data=polys_data,
        polys_offset=polys_offset,
        polys_size=polys_size,
        h_scale=h_scale,
        w_scale=w_scale,
        img_h=img_h,
        img_w=img_w)
    return result


## 提供NonZeroWithValueShape功能
#  @param value double, float, float16, int8, unit8, int16, unit16, int32, unit32, int64, unit64, bool 类型
#  @param index int32 类型
#  @param count int32 类型
#  @return out_value,out_index double, float, float16, int8, unit8, int16, unit16, int32, unit32, int64,
#                              unit64, bool,int32,int32 类型
def non_zero_with_value_shape(value, index, count):
    """ Non zero with value shape. """
    result = gen_npu_cpu_ops.non_zero_with_value_shape(
        value=value,
        index=index,
        count=count)
    return result


## 提供host侧FeatureMapping功能
#  @param feature_id int64 类型
#  @param threshold int 类型
#  @param table_name string 类型
#  @return offset_id int64 类型
def host_feature_mapping(feature_id, threshold=1, table_name="default_table_name"):
    """ host feature mapping. """
    result = gen_npu_cpu_ops.HostFeatureMapping(
        feature_id=feature_id,
        threshold=threshold,
        table_name=table_name)
    return result


## 提供device侧FeatureMapping功能
#  @param feature_id int64 类型
#  @return offset_id int32 类型
def device_feature_mapping(feature_id):
    """ device feature mapping. """
    result = gen_npu_cpu_ops.EmbeddingFeatureMapping(
        feature_id=feature_id)
    return result


## 提供host侧FeatureMapping Import功能
#  @param path string 类型
#  @param table_name string 类型
#  @return fake int32 类型
def host_feature_mapping_export(path, table_name_list):
    """ host feature mapping export. """
    result = gen_npu_cpu_ops.FeatureMappingExport(path=path, table_name_list=table_name_list)
    return result


## 提供host侧FeatureMapping Export功能
#  @param path string 类型
#  @param table_name string 类型
#  @return fake int32 类型
def host_feature_mapping_import(path):
    """ host feature mapping export. """
    result = gen_npu_cpu_ops.FeatureMappingImport(path=path)
    return result
