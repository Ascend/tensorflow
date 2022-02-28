cmake_minimum_required(VERSION 3.14)
project(TFAdapter)

execute_process(COMMAND ln -sf ${CMAKE_CURRENT_LIST_DIR}/../../tf_adapter ${CMAKE_BINARY_DIR})
execute_process(COMMAND ln -sf ${CMAKE_CURRENT_LIST_DIR}/../../inc ${CMAKE_BINARY_DIR})
execute_process(COMMAND ln -sf ${CMAKE_CURRENT_LIST_DIR}/../../cmake ${CMAKE_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_LIST_DIR}/../../tf_adapter/swig DESTINATION ${CMAKE_BINARY_DIR})

include(${CMAKE_BINARY_DIR}/cmake/secure_c.cmake)
include(${CMAKE_BINARY_DIR}/cmake/nlohmann_json.cmake)

include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/inc)
include_directories(${CMAKE_BINARY_DIR}/inc/external)
include_directories(${CMAKE_BINARY_DIR}/inc/external/acl)
include_directories(${CMAKE_BINARY_DIR}/inc/external/hccl)
include_directories(${CMAKE_BINARY_DIR}/inc/graphengine/inc/external)
include_directories(${CMAKE_BINARY_DIR}/inc/toolchain/tuning_tool)
include_directories(${TF_INCLUDE_DIR})

execute_process(COMMAND mkdir -p ${CMAKE_BINARY_DIR}/tensorflow/core/graph/)
execute_process(COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/core/common_runtime/graph_constructor.h 
    ${CMAKE_BINARY_DIR}/tensorflow/core/graph/graph_constructor.h)

execute_process(COMMAND mkdir -p ${CMAKE_BINARY_DIR}/tensorflow/core/kernels/data/)
execute_process(COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/core/data/captured_function.h
    ${CMAKE_BINARY_DIR}/tensorflow/core/kernels/data/captured_function.h)
execute_process(COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/core/framework/dataset.h
    ${CMAKE_BINARY_DIR}/tensorflow/core/kernels/data/dataset.h)

execute_process(COMMAND mkdir -p ${CMAKE_BINARY_DIR}/tensorflow/compiler/jit/graphcycles/)
execute_process(COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/compiler/xla/service/graphcycles/graphcycles.h
    ${CMAKE_BINARY_DIR}/tensorflow/compiler/jit/graphcycles/graphcycles.h)

execute_process(COMMAND mkdir -p ${CMAKE_BINARY_DIR}/tf_adapter_2.x/npu_device/core/)
execute_process(COMMAND ln -sf ${CMAKE_CURRENT_LIST_DIR}/../npu_device/core/npu_micros.h
    ${CMAKE_BINARY_DIR}/tf_adapter_2.x/npu_device/core/)

file(GLOB_RECURSE SOURCES
    ${CMAKE_BINARY_DIR}/tf_adapter/common/*.cc 
    ${CMAKE_BINARY_DIR}/tf_adapter/kernels/*.cc 
    ${CMAKE_BINARY_DIR}/tf_adapter/optimizers/*.cc 
    ${CMAKE_BINARY_DIR}/tf_adapter/util/*.cc 
)

execute_process(COMMAND swig -python -c++ -threads ${CMAKE_BINARY_DIR}/swig/ge_plugin.i)
file(COPY ${CMAKE_BINARY_DIR}/swig/tf_adapter.py DESTINATION ${CMAKE_BINARY_DIR}/dist/python/npu_device)
add_library(_tf_adapter SHARED ${SOURCES} ${CMAKE_BINARY_DIR}/swig/ge_plugin_wrap.cxx)
set_target_properties(_tf_adapter PROPERTIES PREFIX "")

target_compile_definitions(_tf_adapter PRIVATE LOG_CPP TF_VERSION_TF2)

target_link_libraries(_tf_adapter PRIVATE
    tensorflow_libs
    ge_libs
    acl_libs
    ${ASCEND_INSTALLED_PATH}/compiler/lib64/libfmk_onnx_parser.so
    ${ASCEND_INSTALLED_PATH}/compiler/lib64/libc_sec.so
    ${ASCEND_INSTALLED_PATH}/compiler/lib64/libdatatransfer.so
    ${ASCEND_INSTALLED_PATH}/compiler/lib64/libindextransform.so
)