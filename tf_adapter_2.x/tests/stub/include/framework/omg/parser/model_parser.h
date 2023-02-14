#ifndef INC_FRAMEWORK_OMG_PARSER_MODEL_PARSER_H_
#define INC_FRAMEWORK_OMG_PARSER_MODEL_PARSER_H_

#include "stub/defines.h"

namespace domi {
class GE_FUNC_VISIBILITY ModelParser {
 public:
  ModelParser() = default;
  ~ModelParser() = default;
  ge::DataType ConvertToGeDataType(const uint32_t type);
  Status ParseProtoWithSubgraph(const std::string &serialized_proto, GetGraphCallbackV2 callback,
                                ge::ComputeGraphPtr &graph);

  Status ParseProtoWithSubgraph(const std::vector<std::string> &partitioned_serialized,
                                const std::map<std::string, std::string> &const_value_map,
                                GetGraphCallbackV2 callback,
                                ge::ComputeGraphPtr &graph);
};
}  // namespace domi

#endif