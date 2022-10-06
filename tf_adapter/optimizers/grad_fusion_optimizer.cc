/*
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

#include "tf_adapter/optimizers/grad_fusion_optimizer.h"
#include "tf_adapter/common/adp_logger.h"
#include "tf_adapter/common/common.h"

#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/util/overflow.h"

namespace tensorflow {
namespace grappler {
Status GradFusionOptimizer::Optimize(Cluster *cluster, const GrapplerItem &item, GraphDef *optimizedGraph) {
  (void) cluster;
  ADP_LOG(INFO) << "INFO: GradFusionOptimizer::Optimize begin";
  REQUIRES_NOT_NULL(optimizedGraph);
  *optimizedGraph = item.graph;
  ADP_LOG(INFO) << "INFO: GradFusionOptimizer::Optimize end";
  return Status::OK();
}
REGISTER_GRAPH_OPTIMIZER(GradFusionOptimizer);
}  // end namespace grappler
}  // end namespace tensorflow
