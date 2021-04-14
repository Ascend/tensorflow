cmake_minimum_required(VERSION 3.14)
project(NpuOps)

file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../../tf_adapter/ops/*.cc)

foreach (CPP_SOURCE ${SOURCES})
    file(RELATIVE_PATH RELATIVE_CPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} ${CPP_SOURCE})
    set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_DEFINITIONS __FILE__=\"${RELATIVE_CPP_SOURCE}\")
endforeach (CPP_SOURCE)

add_library(_npu_ops SHARED ${SOURCES})
set_target_properties(_npu_ops PROPERTIES PREFIX "")

target_link_libraries(_npu_ops PRIVATE
        tensorflow_libs)
