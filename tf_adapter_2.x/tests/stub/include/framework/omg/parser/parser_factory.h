#ifndef INC_FRAMEWORK_OMG_PARSER_PARSER_FACTORY_H_
#define INC_FRAMEWORK_OMG_PARSER_PARSER_FACTORY_H_

#include "stub/defines.h"

#include "model_parser.h"

namespace domi {
class GE_FUNC_VISIBILITY ModelParserFactory {
 public:
  static ModelParserFactory *Instance() {
    static ModelParserFactory instance;
    return &instance;
  }
  std::shared_ptr<ModelParser> CreateModelParser(const domi::FrameworkType type) {
    return std::make_shared<ModelParser>();
  }

 private:
  ModelParserFactory() = default;
};
}  // namespace domi

#endif