
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef QUANTIZE_COMMON_H
#define QUANTIZE_COMMON_H
#include <map>
#include <float.h>
#include <vector>
#include <semaphore.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <pthread.h>
#include <mutex>
#include <iostream>
#include <fcntl.h>
#include <stdio.h>

// Define common constants in quantization
const int BASE = 2;
const float EPSILON = 1e-6;
const int SHIFT_POW = 15;
const int DEQ_SCALE_BINS = 32;
const int N_LFET_BINS = 24;
const int N_RIGHT_BINS = 56;
const int CIN_DIM = 2;
const int COUT_DIM = 3;
const int NCHW_H_DIM = 2;
const int NCHW_W_DIM = 3;
const int NHWC_H_DIM = 1;
const int NHWC_W_DIM = 2;


// Define the structure of data quantification
template <typename T>
struct QuantInputParam {
  int size;
  const T* in;
  T* out;
  float scale;
  float offset;
  int quant_bits;
};

// Define the structure of weight quantification
template <typename T>
struct WeightQuantInputParam {
  int size;
  const signed char* weight;
  const signed char* offset;
  T* out;
  int channel_in_num;
  int channel_out_num;
  bool channel_wise;
  bool transpose;
};

// Define the structure of data anti quantification
template <typename T>
struct AntiQuantInputParam {
  int size;
  const T* in;
  T* out;
  float scale;
  float offset;
};

// Define the structure of data dequantification
template <typename T>
struct DequantInputParam {
  int area_factor;
  int size;
  const T* input;
  T* out;
  const long long unsigned int* deqscale;
  int channel_num;
  int hw_size;
  bool channel_wise;
  bool transpose;
  std::string data_format;
};

#endif
