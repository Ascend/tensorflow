cmake_minimum_required(VERSION 3.14)
project(NpuCompatV1)

set(ADAPTER_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)

if (DEFINED ASCEND_CI_BUILD_DIR)
    include_directories(${ASCEND_CI_BUILD_DIR}/abl/libc_sec/include)
    include_directories(${ADAPTER_ROOT}/inc/parser/inc)
    include_directories(${ADAPTER_ROOT}/inc/parser/inc/external)
else ()
    include(${ADAPTER_ROOT}/cmake/secure_c.cmake)
    include(${ADAPTER_ROOT}/cmake/nlohmann_json.cmake)
endif ()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ge_plugin_wrap.cxx
        COMMAND swig -python -c++ -threads -o ${CMAKE_CURRENT_BINARY_DIR}/ge_plugin_wrap.cxx -outdir ${COMPAT_V1_DST} ${ADAPTER_ROOT}/tf_adapter/swig/ge_plugin.i
        DEPENDS ${ADAPTER_ROOT}/tf_adapter/swig/ge_plugin.i generate_compat_python
        VERBATIM)

add_custom_target(generate_compat_python
        COMMAND cp -r ${ADAPTER_ROOT}/tf_adapter/python/npu_bridge/. ${COMPAT_V1_DST}
        COMMAND ${PYTHON_BIN_PATH} ${CMAKE_CURRENT_LIST_DIR}/upgrade_tf1_compat_tf2.py ${COMPAT_V1_DST}
        )

add_custom_target(generate_compat_headers
        COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/graph/
        COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/kernels/data/
        COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/compiler/jit/graphcycles/
        COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/core/common_runtime/graph_constructor.h ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/graph/graph_constructor.h
        COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/core/data/captured_function.h ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/kernels/data/captured_function.h
        COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/core/framework/dataset.h ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/kernels/data/dataset.h
        COMMAND ln -sf ${TF_INCLUDE_DIR}/tensorflow/compiler/xla/service/graphcycles/graphcycles.h ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/compiler/jit/graphcycles/graphcycles.h
        )

file(GLOB_RECURSE SOURCES
        ${ADAPTER_ROOT}/tf_adapter/common/*.cc
        ${ADAPTER_ROOT}/tf_adapter/kernels/*.cc
        ${ADAPTER_ROOT}/tf_adapter/optimizers/*.cc
        ${ADAPTER_ROOT}/tf_adapter/util/*.cc
        )
file(GLOB_RECURSE SEPARATED_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/kernels/aicpu/map_dataset_op.cc
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/kernels/aicpu/map_and_batch_dataset_op.cc
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/kernels/aicpu/dataset_function.cc
        )
list(REMOVE_ITEM SOURCES ${SEPARATED_SOURCES})
add_library(_tf_adapter SHARED ${SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/ge_plugin_wrap.cxx)
set_target_properties(_tf_adapter PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${COMPAT_V1_DST})

add_dependencies(_tf_adapter generate_compat_headers)

target_compile_definitions(_tf_adapter PRIVATE LOG_CPP TF_VERSION_TF2)

target_include_directories(_tf_adapter PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${ADAPTER_ROOT}/
        ${ADAPTER_ROOT}/inc/
        ${ADAPTER_ROOT}/inc/toolchain/tuning_tool
        )

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
)

set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

add_library(fmk_onnx_parser SHARED ${fake_sources})
add_library(c_sec SHARED ${fake_sources})
add_library(datatransfer SHARED ${fake_sources})
add_library(indextransform SHARED ${fake_sources})

add_library(compat_v1_addon_libs INTERFACE)
target_link_libraries(compat_v1_addon_libs INTERFACE
        fmk_onnx_parser
        c_sec
        datatransfer
        indextransform)

target_link_libraries(_tf_adapter PRIVATE
        tensorflow_libs
        ge_libs
        acl_libs
        compat_v1_addon_libs
        )
