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

#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_GRADIENT_FUSION_OPTIMIZER_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_GRADIENT_FUSION_OPTIMIZER_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {
class GradFusionOptimizer : public CustomGraphOptimizer {
 public:
  GradFusionOptimizer() {}

  ~GradFusionOptimizer() override = default;

  string name() const override { return "GradFusionOptimizer"; }

  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer *config) override { return Status::OK(); }

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster *cluster, const GrapplerItem &item, GraphDef *optimizedGraph) override;

#ifndef TF_VERSION_TF2
  void Feedback(Cluster *cluster, const GrapplerItem &item, const GraphDef &optimizedGraph, double result) override {}
#endif
};
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_GRADIENT_FUSION_OPTIMIZER_H_