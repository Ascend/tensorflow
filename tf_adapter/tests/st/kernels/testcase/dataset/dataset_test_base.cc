

#include "tensorflow/core/kernels/data/dataset_test_base.h"

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "iostream"
using namespace std;
namespace tensorflow {
namespace data {

template <typename T>
Status IsEqual2(const Tensor& tensor1, const Tensor& tensor2) {
  if (tensor1.dtype() != tensor2.dtype()) {
    return tensorflow::errors::Internal("Tensor1 and tensor2 is different in dtypes: ", DataTypeString(tensor1.dtype())," and. ", DataTypeString(tensor2.dtype()));
  }
  if (!tensor1.IsSameSize(tensor2)) {
    return tensorflow::errors::Internal("Tensor1 and tensor2 is different in sha: ", tensor1.shape().DebugString()," and. ", tensor2.shape().DebugString());
  }

  auto flat_t1 = tensor1.flat<T>();
  auto flat_t2 = tensor2.flat<T>();
  auto length = flat_t1.size();

  for (int i = 0; i < length; ++i) {
    if (flat_t1(i) != flat_t2(i)) {
      return tensorflow::errors::Internal("values is different ""at [", i, "]: ", flat_t1(i), " and. ", flat_t2(i));
    }
  }
  return Status::OK();
}

Status DatasetOpsTestBase::ExpectEqual(const Tensor& c, const Tensor& d) {
  switch (c.dtype()) {
#define case1(DT)                           \
  case DataTypeToEnum<DT>::value:          \
    TF_RETURN_IF_ERROR(IsEqual2<DT>(c, d)); \
    break;
    TF_CALL_NUMBER_TYPES(case1);
    TF_CALL_tstring(case1);
    TF_CALL_uint32(case1);
    TF_CALL_uint64(case1);
#undef case
    default:
      return errors::Internal("The different dtype: ", c.dtype());
  }
  return Status::OK();
}

template <typename T>
bool compare(const Tensor& a, const Tensor& b) {
  auto flat_a = a.flat<T>();
  auto flat_b = b.flat<T>();
  auto length = std::min(flat_a.size(), flat_b.size());
  for (int i = 0; i < length; ++i) {
    if (flat_a(i) < flat_b(i)) {
		return true;
	}
    if (flat_a(i) > flat_b(i))
	{return false;}

  }
  return flat_a.size() < length;
}

Status DatasetOpsTestBase::ExpectEqual(std::vector<Tensor> tensors1, std::vector<Tensor> tensors2, bool order) {
  if (tensors1.size() != tensors2.size()) {
    return Status(tensorflow::errors::Internal( "The two tensor vectors have different size (", tensors1.size()," and. ", tensors2.size(), ")"));
  }

  if (tensors1.empty()) return Status::OK();
  if (tensors1[0].dtype() != tensors2[0].dtype()) {
    return Status(tensorflow::errors::Internal("The two tensor vectors have different dtypes (",tensors1[0].dtype(), " and . ", tensors2[0].dtype(),")"));
  }

  if (!order) {
    const DataType& dtype = tensors1[0].dtype();
    switch (dtype) {
#define case01(DT)                                                \
  case DT:                                                      \
    std::sort(tensors1.begin(), tensors1.end(), \
              compare<EnumToDataType<DT>::Type>);               \
    std::sort(tensors2.begin(), tensors2.end(), \
              compare<EnumToDataType<DT>::Type>);               \
    break;
      case01(DT_FLOAT);
      case01(DT_DOUBLE);
      case01(DT_INT32);
      case01(DT_UINT8);
      case01(DT_INT16);
      case01(DT_INT8);
      case01(DT_STRING);
      case01(DT_INT64);
      case01(DT_BOOL);
      case01(DT_QINT8);
      case01(DT_QUINT8);
      case01(DT_QINT32);
      case01(DT_QINT16);
      case01(DT_QUINT16);
      case01(DT_UINT16);
      case01(DT_HALF);
      case01(DT_UINT32);
      case01(DT_UINT64);
#undef case
      default:
        return errors::Internal("This dtype is now unsupport: ", dtype);
    }
  }

  for (int i = 0; i < tensors1.size(); ++i) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::ExpectEqual(tensors1[i],
                                                       tensors2[i]));
  }
  return Status::OK();
}

Status DatasetOpsTestBase::CreateTensorSliceDatasetKernel(StringPiece node_name, const DataTypeVector& dtypes, const std::vector<PartialTensorShape>& shape,
                                                          std::unique_ptr<OpKernel>* dataset_kernel) {
  std::vector<string> com;
  com.reserve(dtypes.size());
  for (int i = 0; i < dtypes.size(); ++i) {
    com.emplace_back(strings::StrCat("component_", i));
  }
  NodeDef node_def = test::function::NDef(node_name, "TensorSliceDataset", com, {{"Toutput_types", dtypes}, {"output_shapes", shape}});
  TF_RETURN_IF_ERROR(CreateOpKernel(node_def, dataset_kernel));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateTensorSliceDataset(StringPiece node_name, std::vector<Tensor>* const com, DatasetBase** slice_data) {
  std::unique_ptr<OpKernel> dataset_tensors;
  DataTypeVector dtypes;
  dtypes.reserve(com->size());
  std::vector<PartialTensorShape> shape;
  shape.reserve(com->size());
  for (const auto& t : *com) {
    dtypes.push_back(t.dtype());
    gtl::InlinedVector<int64, 4> partial_dim_sizes;
    for (int i = 1; i < t.dims(); ++i) {
      partial_dim_sizes.push_back(t.dim_size(i));
    }
    shape.emplace_back(std::move(partial_dim_sizes));
  }
  TF_RETURN_IF_ERROR(CreateTensorSliceDatasetKernel(node_name, dtypes, shape, &dataset_tensors));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto& tensor : *com) {
    inputs.emplace_back(&tensor);
  }
  TF_RETURN_IF_ERROR(CheckOpKernelInput(*dataset_tensors, inputs));
  std::unique_ptr<OpKernelContext> context;
  TF_RETURN_IF_ERROR(CreateOpKernelContext(dataset_tensors.get(),&inputs, &context));
  TF_RETURN_IF_ERROR(RunOpKernel(dataset_tensors.get(), context.get()));
  TF_RETURN_IF_ERROR(GetDatasetFromContext(context.get(), 0, slice_data));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeRangeDataset(const Tensor& rt, const Tensor& sp, const Tensor& st, const DataTypeVector& output_types,
                                            const std::vector<PartialTensorShape>& output_shapes, Tensor* range_dataset) {
  GraphConstructorOptions gops;
  gops.allow_internal_ops = true;
  gops.expect_device_spec = false;
  TF_RETURN_IF_ERROR(RunFunction(test::function::MakeRangeDataset(),{{RangeDatasetOp::kOutputTypes, output_types},{RangeDatasetOp::kOutputShapes, output_shapes}},
                    {rt, sp, st}, gops, {range_dataset}));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeRangeDataset(const RangeDatasetParams& params, Tensor* range_dataset) {
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = false;
  TF_RETURN_IF_ERROR(RunFunction(
      test::function::MakeRangeDataset(),{{RangeDatasetOp::kOutputTypes, params.output_dtypes},{RangeDatasetOp::kOutputShapes, params.output_shapes}},
      {params.start, params.stop, params.step}, opts,{range_dataset}));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeTakeDataset( const Tensor& input, int64 count, const DataTypeVector& types, const std::vector<PartialTensorShape>& shapes,
    Tensor* take_dataset) {
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = false;

  Tensor count_tensor = CreateTensor<int64>(TensorShape({}), {count});
  TF_RETURN_IF_ERROR(RunFunction(test::function::MakeTakeDataset(),{{TakeDatasetOp::kOutputTypes, types},
                    {TakeDatasetOp::kOutputShapes, shapes}}, {input, count_tensor}, opts,{take_dataset}));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateOpKernel(const NodeDef& node_def, std::unique_ptr<OpKernel>* op_kernels) {
  OpKernel* kernels;
  TF_RETURN_IF_ERROR(tensorflow::CreateOpKernel(device_type_, device_.get(), allocator_, flr_, node_def, TF_GRAPH_DEF_VERSION, &kernels));
  op_kernels->reset(kernels);
  return Status::OK();
}

Status DatasetOpsTestBase::CreateDatasetContext(OpKernel* const data_set_kernel, gtl::InlinedVector<TensorValue, 4>* const nums, std::unique_ptr<OpKernelContext>* context) {
  TF_RETURN_IF_ERROR(CheckOpKernelInput(*data_set_kernel, *nums));
  TF_RETURN_IF_ERROR(CreateOpKernelContext(data_set_kernel, nums, context));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateDataset(OpKernel* data_set_kernel, OpKernelContext* context_data, DatasetBase** const dataset_test_base) {
  TF_RETURN_IF_ERROR(RunOpKernel(data_set_kernel, context_data));
  DCHECK_EQ(context_data->num_outputs(), 1);
  TF_RETURN_IF_ERROR(GetDatasetFromContext(context_data, 0, dataset_test_base));
  return Status::OK();
}

Status DatasetOpsTestBase::RestoreIterator(IteratorContext* context, IteratorStateReader* readers, const string& output_prefix, const DatasetBase& dataset,
                                           std::unique_ptr<IteratorBase>* iterator) {
  TF_RETURN_IF_ERROR(dataset.MakeIterator(context, output_prefix, iterator));
  TF_RETURN_IF_ERROR((*iterator)->Restore(context, readers));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateIteratorContext(OpKernelContext* const context, std::unique_ptr<IteratorContext>* iterator) {
  IteratorContext::Params params(context);
  params.resource_mgr = context->resource_manager();
  function_handle_cache_ = absl::make_unique<FunctionHandleCache>(flr_);
  params.function_handle_cache = function_handle_cache_.get();
  params.cancellation_manager = cancellation_manager_.get();
  *iterator = absl::make_unique<IteratorContext>(params);
  return Status::OK();
}

Status DatasetOpsTestBase::GetDatasetFromContext(OpKernelContext* context, int index, DatasetBase** const dataset_base) {
  Tensor* out_put = context->mutable_output(index);
  Status status = GetDatasetFromVariantTensor(*out_put, dataset_base);
  (*dataset_base)->Ref();
  return status;
}

Status DatasetOpsTestBase::InitThreadPool(int num) {
  if (num < 1) {
    return errors::InvalidArgument("current num is : ", num);
  }
  thread_pool_ = absl::make_unique<thread::ThreadPool>(Env::Default(), ThreadOptions(), "test_thread_pool", num);
  return Status::OK();
}

Status DatasetOpsTestBase::InitFunctionLibraryRuntime( const std::vector<FunctionDef>& flib, int num) {
  if (num < 1) {
    return errors::InvalidArgument("The cpu nums is : ", num);
  }
  SessionOptions option;
  auto* device_count = option.config.mutable_device_count();
  device_count->insert({"CPU", num});
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(option, "/job:localhost/replica:0/task:0", &devices));
  device_mgr_ = absl::make_unique<DeviceMgr>(std::move(devices));
  resource_mgr_ = absl::make_unique<ResourceMgr>("default_container");

  FunctionDefLibrary protobuf;
  for (const auto& fdef : flib) *(protobuf.add_function()) = fdef;
  lib_def_ = absl::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(), protobuf);
  OptimizerOptions options;
  pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(device_mgr_.get(), Env::Default(), TF_GRAPH_DEF_VERSION, lib_def_.get(),
      options, thread_pool_.get(), nullptr );
  flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  if (thread_pool_ == nullptr) {
    runner_ = [](std::function<void()> fn) { fn(); };
  } else {
    runner_ = [this](std::function<void()> fn) {thread_pool_->Schedule(std::move(fn));
    };
  }
  return Status::OK();
}

Status DatasetOpsTestBase::RunOpKernel(OpKernel* op_kernels, OpKernelContext* ctx) {
  device_->Compute(op_kernels, ctx);
  return ctx->status();
}

Status DatasetOpsTestBase::RunFunction(const FunctionDef& fdef1, test::function::Attrs attr, const std::vector<Tensor>& arg, const GraphConstructorOptions& gps, std::vector<Tensor*> rets) {
  std::unique_ptr<Executor> exec;
  InstantiationResult ret;
  auto GetOpSig = [](const string& op, const OpDef** sig) {
    return OpRegistry::Global()->LookUpOpDef(op, sig);
  };
  TF_RETURN_IF_ERROR(InstantiateFunction(fdef1, attr, GetOpSig, &ret));

  DataTypeVector arg_type = ret.arg_types;
  DataTypeVector ret_type = ret.ret_types;

  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_RETURN_IF_ERROR(ConvertNodeDefsToGraph(gps, ret.nodes, g.get()));

  const int version = g->versions().producer();
  LocalExecutorParams pas;
  pas.function_library = flr_;
  pas.device = device_.get();
  pas.create_kernel = [this, version](const NodeDef& ndef, OpKernel** kernel) {
    return CreateNonCachedKernel(device_.get(), this->flr_, ndef, version,  kernel);
  };
  pas.delete_kernel = [](OpKernel* kernel) {
    DeleteNonCachedKernel(kernel);
  };
  pas.rendezvous_factory = [](const int64, const DeviceMgr* device_mgr,Rendezvous** r) {
    *r = new IntraProcessRendezvous(device_mgr);
    return Status::OK();
  };

  Executor* cur_execs;
  TF_RETURN_IF_ERROR(NewLocalExecutor(pas, std::move(g), &cur_execs));
  exec.reset(cur_execs);
  FunctionCallFrame frame(arg_type, ret_type);
  TF_RETURN_IF_ERROR(frame.SetArgs(arg));
  Executor::Args args;
  args.call_frame = &frame;
  args.runner = runner_;
  TF_RETURN_IF_ERROR(exec->Run(args));
  std::vector<Tensor> computed;
  TF_RETURN_IF_ERROR(frame.GetRetvals(&computed));
  if (computed.size() != rets.size()) {
    return errors::InvalidArgument("The result ",". expected: ", rets.size(), "and actual: ", computed.size());
  }
  for (int i = 0; i < rets.size(); ++i) {
    *(rets[i]) = computed[i];
  }
  return Status::OK();
}

Status DatasetOpsTestBase::CreateOpKernelContext( OpKernel* kernels, gtl::InlinedVector<TensorValue, 4>* inputs, std::unique_ptr<OpKernelContext>* ctx) {
  params_ = absl::make_unique<OpKernelContext::Params>();
  unique_ptr<OpKernelContext::Params> pas;
  pas = std::move(params_);
  cancellation_manager_ = absl::make_unique<CancellationManager>();
  unique_ptr<CancellationManager> cm = std::move(cancellation_manager_);
  pas->cancellation_manager = cm.release();
  pas->device = device_.get();
  pas->frame_iter = FrameAndIter(0, 0);
  pas->function_library = flr_;
  pas->inputs = inputs;
  pas->op_kernel = kernels;
  pas->resource_manager = resource_mgr_.get();
  pas->runner = &runner_;
  slice_reader_cache_ = absl::make_unique<checkpoint::TensorSliceReaderCacheWrapper>();
  pas->slice_reader_cache = slice_reader_cache_.get();
  step_container_ = absl::make_unique<ScopedStepContainer>(0, [](const string&) {});
  unique_ptr<ScopedStepContainer> stc = std::move(step_container_);
  pas->step_container = stc.release();

  allocator_attrs_.clear();
  for (int i = 0; i < pas->op_kernel->num_outputs(); i++) {
    AllocatorAttributes attr;
    const bool on_host = (pas->op_kernel->output_memory_types()[i] == HOST_MEMORY);
    attr.set_on_host(on_host);
    allocator_attrs_.emplace_back(attr);
  }
  pas->output_attr_array = gtl::vector_as_array(&allocator_attrs_);

  *ctx = absl::make_unique<OpKernelContext>(pas.release());
  return Status::OK();
}

Status DatasetOpsTestBase::CreateSerializationContext(std::unique_ptr<SerializationContext>* context) {
  *context = absl::make_unique<SerializationContext>(SerializationContext::Params{});
  return Status::OK();
}

Status DatasetOpsTestBase::CheckOpKernelInput(
    const OpKernel& kernel, const gtl::InlinedVector<TensorValue, 4>& inputs) {
  if (kernel.input_types().size() != inputs.size()) {
    cout<<"++"<<kernel.input_types().size()<<" : "<<inputs.size()<<endl;
    return errors::Internal("Elements should be ", kernel.input_types().size(), ", but got: ", inputs.size());
  }
  return Status::OK();
}

Status DatasetOpsTestBase::AddDatasetInput(
    gtl::InlinedVector<TensorValue, 4>* inputs, DataTypeVector input_types,
    DataType dtype, const TensorShape& shape) {
  if (input_types.size() < inputs->size()) {
    return errors::InvalidArgument("Adding  types: ", inputs->size(), " and. ", input_types.size());
  }
  bool is_ref = IsRefType(input_types[inputs->size()]);
  std::unique_ptr<Tensor> input = absl::make_unique<Tensor>(allocator_, dtype, shape);

  if (is_ref) {
    DataType expected_dtype = RemoveRefType(input_types[inputs->size()]);
    if (expected_dtype != dtype) {
      return errors::InvalidArgument("Data type is ", dtype, " , but expected: ", expected_dtype);
    }
    inputs->push_back({&lock_for_refs_, input.get()});
  } else {
    if (input_types[inputs->size()] != dtype) {
      return errors::InvalidArgument( "Data type is ", dtype, " , but expected: ", input_types[inputs->size()]);
    }
    inputs->push_back({nullptr, input.get()});
  }
  tensors_.push_back(std::move(input));

  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorGetNext( const std::vector<Tensor>& expected_outputs, bool orders) {
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensor;
  while (!end_of_sequence) {
    // sleep(1);
    std::vector<Tensor> next;
    TF_RETURN_IF_ERROR(iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    out_tensor.insert(out_tensor.end(), next.begin(), next.end());
  }

  TF_EXPECT_OK(ExpectEqual(out_tensor, expected_outputs, orders));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetNodeName( const string& node_names) {
  EXPECT_EQ(dataset_->node_name(), node_names);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetTypeString( const string& str) {
  EXPECT_EQ(dataset_->type_string(), str);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetOutputDtypes( const DataTypeVector& dtypes) {
  TF_EXPECT_OK(VerifyTypesMatch(dataset_->output_dtypes(), dtypes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetOutputShapes(const std::vector<PartialTensorShape>& expected_output_shapes) {
  TF_EXPECT_OK(VerifyShapesCompatible(dataset_->output_shapes(),expected_output_shapes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetCardinality(int expected_cardinality) {
  EXPECT_EQ(dataset_->Cardinality(), expected_cardinality);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorOutputDtypes(const DataTypeVector& dtypes) {
  TF_EXPECT_OK(
      VerifyTypesMatch(iterator_->output_dtypes(), dtypes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorOutputShapes(const std::vector<PartialTensorShape>& shapes) {
  TF_EXPECT_OK(VerifyShapesCompatible(iterator_->output_shapes(), shapes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorPrefix( const string& expected_prefix) {
  EXPECT_EQ(iterator_->prefix(), expected_prefix);
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
