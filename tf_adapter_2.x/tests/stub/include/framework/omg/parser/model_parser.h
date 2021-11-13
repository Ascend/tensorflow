#ifndef INC_FRAMEWORK_OMG_PARSER_MODEL_PARSER_H_
#define INC_FRAMEWORK_OMG_PARSER_MODEL_PARSER_H_

#include "stub/defines.h"

namespace domi {
class GE_FUNC_VISIBILITY ModelParser {
 public:
  ModelParser() = default;
  ~ModelParser() = default;
  ge::DataType ConvertToGeDataType(const uint32_t type) { return ge::DT_FLOAT; }
  Status ParseProtoWithSubgraph(const std::string &serialized_proto, GetGraphCallbackV2 callback,
                                ge::ComputeGraphPtr &graph) {
    return ge::SUCCESS;
  }
};
}  // namespace domi

#endif