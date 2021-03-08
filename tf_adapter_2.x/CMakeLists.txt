cmake_minimum_required(VERSION 3.14)
project(NpuDevice)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "-O2 -DNDEBUG -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -s -pipe ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-O2 -DNDEBUG -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -s -pipe ${CMAKE_CXX_FLAGS}")


if (DEFINED ASCEND_CI_BUILD_DIR)
    set(CMAKE_C_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
    include_directories(${PYTHON_INCLUDE_DIR})
else()
    if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS OR NOT EXISTS
            ${CMAKE_CURRENT_LIST_DIR}/tools/TF_INSTALLED_PATH OR NOT EXISTS
            ${CMAKE_CURRENT_LIST_DIR}/tools/ASCEND_INSTALLED_PATH OR NOT EXISTS
            ${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH)
        message(FATAL_ERROR "No validate configuration found. Did you forget to configure first?")
    endif ()

    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/TF_INSTALLED_PATH" TF_INSTALLED_PATH)
    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/ASCEND_INSTALLED_PATH" ASCEND_INSTALLED_PATH)
    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH" PYTHON_BIN_PATH)

    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS" CUSTOM_COMPILE_FLAGS)
    foreach (COMPILE_FLAG ${CUSTOM_COMPILE_FLAGS})
        set(CMAKE_C_FLAGS "${COMPILE_FLAG} ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${COMPILE_FLAG} ${CMAKE_CXX_FLAGS}")
    endforeach (COMPILE_FLAG)
endif ()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/acl/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/tensorflow/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/graph_engine/module.cmake)

include_directories(${CMAKE_CURRENT_LIST_DIR}/npu_device/core)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/npu_device/python DESTINATION ${CMAKE_BINARY_DIR}/dist)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_LIST_DIR}/npu_device/*.cpp)

IF (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    add_definitions(-Wno-builtin-macro-redefined)
ENDIF ()

foreach (CPP_SOURCE ${SOURCES})
    file(RELATIVE_PATH RELATIVE_CPP_SOURCE ${CMAKE_CURRENT_LIST_DIR} ${CPP_SOURCE})
    set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_DEFINITIONS __FILE__=\"${RELATIVE_CPP_SOURCE}\")
endforeach (CPP_SOURCE)

add_library(_npu_device_backends SHARED ${SOURCES})
set_target_properties(_npu_device_backends PROPERTIES PREFIX "")
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/dist/python/npu_device)

target_link_libraries(_npu_device_backends PRIVATE
        tensorflow_libs
        ge_libs
        acl_libs)

add_custom_command(TARGET _npu_device_backends
        POST_BUILD
        COMMAND cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} setup.py bdist_wheel
        VERBATIM)

install(CODE "execute_process(COMMAND ${PYTHON_BIN_PATH} -m pip install ${CMAKE_BINARY_DIR}/dist/python/dist/npu_device-0.1-py3-none-any.whl --upgrade)")
