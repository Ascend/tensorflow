cmake_minimum_required(VERSION 3.14)
project(NpuDevice)

include_directories(${CMAKE_CURRENT_LIST_DIR}/core)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
foreach (CPP_SOURCE ${SOURCES})
    file(RELATIVE_PATH RELATIVE_CPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} ${CPP_SOURCE})
    set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_DEFINITIONS __FILE__=\"${RELATIVE_CPP_SOURCE}\")
endforeach (CPP_SOURCE)

add_library(_npu_device_backends SHARED ${SOURCES})
set_target_properties(_npu_device_backends PROPERTIES PREFIX "")

target_link_libraries(_npu_device_backends PRIVATE
        tensorflow_libs
        ge_libs
        acl_libs)
