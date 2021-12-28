/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
REGISTER_OP("QueueDataset")
    .Input("input_dataset: variant")
    .Attr("sourcedata: string")
    .Output("handle: variant")
    .SetIsStateful()
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("HostQueueDataset")
    .Input("geop_dataset: variant")
    .Input("input_dataset: variant")
    .Attr("channel_name: string")
    .Output("handle: variant")
    .SetIsStateful()
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("DeviceQueueDataset")
    .Attr("channel_name: string")
    .Output("handle: variant")
    .SetIsStateful()
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("GEOPDataset")
    .Output("handle: variant")
    .Attr("f: func")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("DPGroupDataset")
    .Input("input_datasets: N * variant")
    .Output("handle: variant")
    .Attr("N: int >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("AdpGetNext")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("queue_name: string")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("GetNext")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("channel_name: string")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("DynamicGetNextV2")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("channel_name: string")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("GetNextFromQueue")
    .Input("data: uint8")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

REGISTER_OP("QueueData")
    .Output("output: T")
    .Attr("index: int >= 0")
    .Attr("T: type")
    .Attr("queue_name: string")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);
}  // namespace tensorflow
