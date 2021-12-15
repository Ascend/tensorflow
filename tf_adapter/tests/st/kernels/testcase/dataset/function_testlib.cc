#include "tensorflow/core/framework/function_testlib.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace test {
namespace function {

typedef FunctionDefHelper FDH;

// Helper to construct a NodeDef.
NodeDef NDef(StringPiece names, StringPiece opereator, gtl::ArraySlice<string> input,
             gtl::ArraySlice<std::pair<string, FDH::AttrValueWrapper>> attr,
             const string& device_id) {
  NodeDef n;
  n.set_name(string(names));
  n.set_op(string(opereator));
  for (const auto& in : input) n.add_input(in);
  n.set_device(device_id);
  for (auto& na : attr) n.mutable_attr()->insert({na.first, na.second.proto});
  return n;
}

FunctionDef MakeRangeDataset() {
  return FDH::Define(
      "MakeRangeDataset", {"start: int64", "stop: int64", "step: int64"}, {"y:variant"},
      {"output_types: list(type) >= 1", "output_shapes: list(shape) >= 1"}, {{{"y"},"RangeDataset", 
      {"start", "stop", "step"}, {{"output_types", "$output_types"}, {"output_shapes", "$output_shapes"}}}});
}

FunctionDef MakeTakeDataset() {
  return FDH::Define(
      "TakeDataset", {"input_dataset: variant", "count: int64"}, {"y:variant"},
      {"output_types: list(type) >= 1", "output_shapes: list(shape) >= 1"},
      {{{"y"}, "TakeDataset", {"input_dataset", "count"}, {{"output_types", "$output_types"},
      {"output_shapes", "$output_shapes"}}}});
}

FunctionDef MakeTensorSliceDataset() {
  return FDH::Define(
      "MakeTensorSliceDataset", {"x: Toutput_types"}, {"y: variant"} , {"Toutput_types: list(type) >= 1", "output_shapes: list(shape) >= 1"},
      {{{"y"}, "TensorSliceDataset", {"x"}, {{"Toutput_types", "$Toutput_types"}, {"output_shapes", "$output_shapes"}}}});
}
}  // end namespace function
}  // end namespace test
}  // end namespace tensorflow
