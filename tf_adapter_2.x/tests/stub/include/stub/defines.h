#ifndef STUB_DEFINES_H_
#define STUB_DEFINES_H_

#include <functional>
#include "graph/types.h"
#include "ge_api_error_codes.h"

namespace ge {
// Option key: graph run mode
const char *const OPTION_GRAPH_RUN_MODE = "ge.graphRunMode";
const char *const OPTION_DEVICE_TYPE = "ge.deviceType";

// Option key: topo sorting mode
const char *const OPTION_TOPO_SORTING_MODE = "ge.topoSortingMode";

// Option key: ome init
const char *const OPTION_EXEC_SESSION_ID = "ge.exec.sessionId";
const char *const OPTION_EXEC_DEVICE_ID = "ge.exec.deviceId";
const char *const OPTION_EXEC_JOB_ID = "ge.exec.jobId";
const char *const OPTION_EXEC_IS_USEHCOM = "ge.exec.isUseHcom";
const char *const OPTION_EXEC_IS_USEHVD = "ge.exec.isUseHvd";
const char *const OPTION_EXEC_RANK_ID = "ge.exec.rankId";
const char *const OPTION_EXEC_POD_NAME = "ge.exec.podName";
const char *const OPTION_EXEC_DEPLOY_MODE = "ge.exec.deployMode";
const char *const OPTION_EXEC_RANK_TABLE_FILE = "ge.exec.rankTableFile";
const char *const GE_AICPU_FLAG = "ge.aicpuFlag";
const char *const OPTION_EXEC_EXTERN_PLUGIN_PATH = "ge.soLoadPath";

const std::string OPTION_EXEC_CM_CHIEF_IP = "ge.cmChiefIp";
const std::string OPTION_EXEC_CM_CHIEF_PORT = "ge.cmChiefPort";
const std::string OPTION_EXEC_CM_CHIEF_DEVICE = "ge.cmChiefWorkerDevice";
const std::string OPTION_EXEC_CM_WORKER_IP = "ge.cmWorkerIp";
const std::string OPTION_EXEC_CM_WORKER_SIZE = "ge.cmWorkerSize";

// Dump flag and para
const char *const OPTION_EXEC_ENABLE_DUMP = "ge.exec.enableDump";
const char *const OPTION_EXEC_DUMP_PATH = "ge.exec.dumpPath";
const char *const OPTION_EXEC_DUMP_STEP = "ge.exec.dumpStep";
const char *const OPTION_EXEC_DUMP_MODE = "ge.exec.dumpMode";
const char *const OPTION_EXEC_ENABLE_DUMP_DEBUG = "ge.exec.enableDumpDebug";
const char *const OPTION_EXEC_DUMP_DEBUG_MODE = "ge.exec.dumpDebugMode";
const char *const OPTION_EXEC_ENABLE_INCRE_BUILD = "ge.exec.enableIncreBuild";
const char *const OPTION_EXEC_INCRE_BUILD_CACHE_PATH = "ge.exec.increBuildCachePath";
const char *const OPTION_EXEC_ENABLE_EXCEPTION_DUMP = "ge.exec.enable_exception_dump";
const char *const OPTION_EXEC_ENABLE_SCOPE_FUSION_PASSES = "ge.exec.enableScopeFusionPasses";
const char *const OPTION_EXEC_PROFILING_FPPONIT_OPTIONS = "ge.exec.profilingFpPointOptions";
const char *const OPTION_EXEC_PROFILING_BPPONIT_OPTIONS = "ge.exec.profilingBpPointOptions";
// profiling flag
const char *const OPTION_EXEC_PROFILING_MODE = "ge.exec.profilingMode";
const char *const OPTION_EXEC_PROFILING_OPTIONS = "ge.exec.profilingOptions";
// Hccl flag, if ge.exec.hcclFlag =1, it means load plugin for opskernel, else:ge.exec.hcclFlag =0
const char *const OPTION_EXEC_HCCL_FLAG = "ge.exec.hcclFlag";
const char *const OPTION_EXEC_ATOMIC_FLAG = "ge.exec.enable_atomic";
const char *const OPTION_EXEC_DISABLE_REUSED_MEMORY = "ge.exec.disableReuseMemory";
const char *const OPTION_EXEC_ENABLE_TAILING_OPTIMIZATION = "ge.exec.isTailingOptimization";
// Dynamic input flag. ge.exec.dynamicInput=1, means enable dynaimc input,
// ge.exec.dynamicGraphExecuteMode, dynamic_execute[default]
const char *const OPTION_EXEC_DYNAMIC_INPUT = "ge.exec.dynamicInput";
const char *const OPTION_EXEC_DYNAMIC_EXECUTE_MODE = "ge.exec.dynamicGraphExecuteMode";
const char *const OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE = "ge.exec.dataInputsShapeRange";
const char *const OPTION_EXEC_ENABLE_COPY_OUTPUT_ADDR = "ge.exec.enableCopyOutputAddr";
const char *const OPTION_EXEC_GRAPH_EXEC_TIMEOUT = "ge.exec.graphExecTimeout";
const char *const OPTION_EXEC_LOGICAL_DEVICE_CLUSTER_DEPLOY_MODE = "ge.exec.logicalDeviceClusterDeployMode";
const char *const OPTION_EXEC_LOGICAL_DEVICE_ID = "ge.exec.logicalDeviceId";
const char *const OPTION_EXEC_MODEL_DEPLOY_MODE = "ge.exec.modelDeployMode";
const char *const OPTION_EXEC_MODEL_DEPLOY_DEVICELIST = "ge.exec.modelDeployDevicelist";

// Option key: memory init
const char *const GRAPH_MEMORY_MAX_SIZE = "ge.graphMemoryMaxSize";
const char *const VARIABLE_MEMORY_MAX_SIZE = "ge.variableMemoryMaxSize";
const char *const MEMORY_OPTIMIZATION_POLICY = "ge.exec.memoryOptimizationPolicy";
const char *const OPTION_INPUT_REUSE_MEM_INDEXES = "ge.exec.inputReuseMemIndexes";
const char *const OPTION_OUTPUT_REUSE_MEM_INDEXES = "ge.exec.outputReuseMemIndexes";

// Configure stream num by Session constructor options param,
// its value should be int32_t type, default value is "1"
const std::string STREAM_NUM = "ge.streamNum";

// Configure add head stream to model.
// its value should be "0" or "1", default value is "0"
const std::string HEAD_STREAM = "ge.headStream";

// Configure perf level by Session constructor options param,
// its value please see enum PerfLevel, default value is "4"
const std::string PERF_LEVEL = "ge.perfLevel";

// Configure encrypt mode by Session constructor options param,
// its value should be int32_t type, default value is "-1"
const std::string ENCRYPT_MODE = "ge.encryptMode";

// configure ek file by Session constructor options param,
// its value should be file path, default value is ""
const std::string EK_FILE = "ge.ekFile";

// Configure cert file by Session constructor options param,
// its value should be file path, default value is ""
const std::string CERT_FILE = "ge.certFile";

// Configure hw key file by Session constructor options param,
// its value should be file path, default value is ""
const std::string HW_KEY_FILE = "ge.hwKeyFile";

// Configure private file by Session constructor options param,
// its value should be file path, default value is ""
const std::string PRIVATE_KEY_FILE = "ge.privateKeyFile";

// Configure framework type by Session constructor options param,
// its value please see enum FrameworkType, default value is "3"
const std::string FRAMEWORK_TYPE = "ge.frameworkType";

// Configure calibration info file by Session constructor options param,
// its value should be file path, default value is ""
const std::string CALIBRATION_CONF_FILE = "ge.calibrationConfFile";

// Configure insert op info file by Session constructor options param,
// its value should be file path, default value is ""
const std::string INSERT_OP_FILE = "ge.insertOpFile";

// Configure output node name by Session constructor options param,
// its value should be std::string type, default value is ""
const std::string OUTPUT_NODE_NAME = "ge.outputNodeName";

// Configure weight compress flag by Session constructor options param,
// its value should be "0" or "1", default value is "0"
const std::string COMPRESS_FLAG = "ge.compressFlag";

const std::string PRECISION_MODE = "ge.exec.precision_mode";

const std::string TUNE_DEVICE_IDS = "ge.exec.tuneDeviceIds";

// Configure single op flag for FE
// its value should be "0" or "1", default value is "0"
const std::string SINGLE_OP_FLAG = "ge.exec.single_op";

// Configure train flag by Session constructor options param,
// its value should be "0" or "1", default value is "0"
const std::string TRAIN_FLAG = "ge.trainFlag";

// Configure run flag by Session constructor options param,
// its value should be "0" or "1", default value is "0"
const std::string RUN_FLAG = "ge.runFlag";

// Configure run flag by Session constructor options param,
// its value should be "0" or "1", default value is "0"
// this option is to enable local framework op feature
const std::string LOCAL_FMKOP_FLAG = "ge.enabledLocalFmkop";

// Configure run flag by Session constructor options param,
// its value should be a path
// this option is to obtain the TBE op plugin path
const std::string TBE_PLUGIN_PATH_FLAG = "ge.TBE_plugin_path";

// Configure run flag by Session constructor options param,
// its value should be a path
// this option is to obtain the DDK Version info
const std::string DDK_VERSION_FLAG = "ge.DDK_version";

// Configure run flag by Session constructor options param,
// its value should be a path
// this option is to obtain fe flag
const std::string GE_FE_FLAG = "ge.feFlag";

// Configure stream max parallel num only by Session constructor options param,
// its value should be stream:int, such as "DNN_V100:2,DNN_HCCL:3",
// default value is "1", such as "DNN_V100:1,DNN_HCCL:1"
// this option is to obtain stream max parallel num
const std::string STREAM_MAX_PARALLEL_NUM = "ge.streamMaxParallelNum";

// congigure outputDatatype to setting net output type
const std::string OUTPUT_DATATYPE = "ge.outputDatatype";

// congigure opSelectImplmode to setting op select implmode
const std::string OP_SELECT_IMPL_MODE = "ge.opSelectImplmode";

// congigure optypelist_for_implmode to setting which op use implmode
const std::string OPTYPELIST_FOR_IMPLMODE = "ge.optypelistForImplmode";

// configure whether to enable hcom parallel by session constructor options param,
// its value should be "0" or "1", default value is "0"
const std::string HCOM_PARALLEL = "ge.hcomParallel";

// configure whether to use dynamic batch size
const char *const kDynamicBatchSize = "ge.dynamicBatchSize";

// configure threshold of fusion data size for communication op
const std::string FUSION_TENSOR_SIZE = "ge.fusionTensorSize";

const std::string INPUT_SHAPE = "ge.inputShape";

const std::string DYNAMIC_NODE_TYPE = "ge.dynamicNodeType";
// configure whether to use dynamic image size
const char *const kDynamicImageSize = "ge.dynamicImageSize";

// Configure whether to use dynamic dims
const char *const kDynamicDims = "ge.dynamicDims";

// Configure auto tune mode, this option only take effect while AUTO_TUNE_FLAG is Y,
// example: GA|RL, support configure multiple, split by |
const std::string AUTO_TUNE_MODE = "ge.autoTuneMode";

// Configure soc version , example: "Ascend310"
const std::string SOC_VERSION = "ge.socVersion";

// Configure core type "VectorEngine", default value is "AIcoreEngine"
const std::string CORE_TYPE = "ge.engineType";

// Configure AICORE NUM
const std::string AICORE_NUM = "ge.aicoreNum";

// Configure L1FUSION
const std::string L1_FUSION = "ge.l1Fusion";

// Configure l1,l2,and others optimize option
const std::string BUFFER_OPTIMIZE = "ge.bufferOptimize";

// Configure Small Channel flag
const std::string ENABLE_SMALL_CHANNEL = "ge.enableSmallChannel";

// Configure Jit Compile
const std::string JIT_COMPILE = "ge.jit_compile";

// Configure External Weight
const std::string EXTERNAL_WEIGHT = "ge.externalWeight";

// Configure Compress Weight flag
const std::string ENABLE_COMPRESS_WEIGHT = "ge.enableCompressWeight";

// Configure fusion switch file path
const std::string FUSION_SWITCH_FILE = "ge.fusionSwitchFile";

// Save original model
const std::string SAVE_ORIGINAL_MODEL = "ge.saveOriginalModel";

// Save original model file name
const std::string ORIGINAL_MODEL_FILE = "ge.originalModelFile";

const char *const OPTION_GE_MAX_DUMP_FILE_NUM = "ge.maxDumpFileNum";
const char *const OPTION_GE_MAX_DUMP_FILE_SIZE = "ge.maxDumpFileSize";
const char *const OPTION_GE_MAX_DUMP_OP_NUM = "ge.maxDumpOpNum";

// Configure for print op pass
// Its value should be "0" or "1", default value is "1"
const char *const ENABLE_PRINT_OP_PASS = "ge.enablePrintOpPass";

// Configure operator compilation path
// Its value should be file path, default value is "./"
const char *const DEBUG_DIR = "ge.debugDir";

// Configure operator compiler cache path
// Its value should be file path, default value is "./"
const char *const OP_COMPILER_CACHE_DIR = "ge.op_compiler_cache_dir";

// Configure operator compiler cache mode
// Its value should be "disable", "enable" or "force", default value is "disable"
const char *const OP_COMPILER_CACHE_MODE = "ge.op_compiler_cache_mode";

// Configure whether to use single stream.
// Its value should be "true" or "false", default value is "false"
const char *const ENABLE_SINGLE_STREAM = "ge.enableSingleStream";

// Configure input fp16 nodes
const std::string INPUT_FP16_NODES = "ge.INPUT_NODES_SET_FP16";

// Configure debug level, its value should be 0(default), 1 or 2.
// 0: close debug; 1: open TBE compiler; 2: open ccec compiler
const std::string OP_DEBUG_LEVEL = "ge.opDebugLevel";

// Configure model bank path
const std::string MDL_BANK_PATH_FLAG = "ge.mdl_bank_path";

// Configure display_model_info flag
const std::string DISPLAY_MODEL_INFO = "ge.display_model_info";

// Configure op bank path
const std::string OP_BANK_PATH_FLAG = "ge.op_bank_path";
const std::string OP_BANK_UPDATE_FLAG = "ge.op_bank_update";

// Configure for fix hcombroadcast format.
// when config model multi, broadcast format should be fixed
// 0: data multi; 1: model multi;
const std::string HCOM_MULTI_MODE = "ge.hcomMultiMode";

// atc and ir option
const char *const INPUT_SHAPE_RANGE = "input_shape_range";

// Configure express high compile performance or high execute performance
// normal: no need to compile, used saved .o files directly
// high: need to recompile, high execute performance mode
const std::string PERFORMANCE_MODE = "ge.performance_mode";

// For selecting the mode of shape generalization when build graph.
// shape_generalized: Shape will be generalized during graph build.
// shape_precise: Shape will not be generalized, use precise shape.
const std::string SHAPE_GENERALIZED_BUILD_MODE = "ge.shape_generalized_build_mode";

const std::string MODIFY_MIXLIST = "ge.exec.modify_mixlist";

const std::string OP_PRECISION_MODE = "ge.exec.op_precision_mode";

// Graph run mode
enum GraphRunMode { PREDICTION = 0, TRAIN };
// Topo sorting mode
enum class TopoSortingMode { BFS = 0, DFS = 1 };

using Status = uint32_t;
class ComputeGraph;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;

using graphStatus = uint32_t;
const graphStatus GRAPH_FAILED = 0xFFFFFFFF;
const graphStatus GRAPH_SUCCESS = 0;
const graphStatus GRAPH_NOT_CHANGED = 1343242304;
const graphStatus GRAPH_PARAM_INVALID = 50331649;
const graphStatus GRAPH_NODE_WITHOUT_CONST_INPUT = 50331648;
const graphStatus GRAPH_NODE_NEED_REPASS = 50331647;

}  // namespace ge

namespace domi {
using Status = uint32_t;
using GetGraphCallbackV2 = std::function<std::string(const std::string &subgraph_name)>;
enum FrameworkType {
  CAFFE = 0,
  MINDSPORE = 1,
  TENSORFLOW = 3,
  ANDROID_NN,
  ONNX,
  FRAMEWORK_RESERVED,
};
}  // namespace domi

#endif
