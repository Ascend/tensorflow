cmake_minimum_required(VERSION 3.14)
project(AscendAdapter2)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "-D_FORTIFY_SOURCE=2 -O2 -DNDEBUG -Wfloat-equal -Wextra -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -s -fno-common -pipe ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-D_FORTIFY_SOURCE=2 -O2 -DNDEBUG -Wfloat-equal -Wextra -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -s -fno-common -pipe ${CMAKE_CXX_FLAGS}")
set(CMAKE_SKIP_RPATH TRUE)

IF (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    add_definitions(-Wno-builtin-macro-redefined)
ENDIF ()

if (DEFINED ASCEND_CI_BUILD_DIR)
    set(CMAKE_C_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
    include_directories(${PYTHON_INCLUDE_DIR})
    add_definitions(-DASCEND_CI_LIMITED_PY37)
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

file(COPY ${CMAKE_CURRENT_LIST_DIR}/python DESTINATION ${CMAKE_BINARY_DIR}/dist)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/dist/python/npu_device)

add_subdirectory(npu_device)
add_subdirectory(npu_ops)

add_custom_target(ascend_adapter2 ALL
        COMMAND cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} setup.py bdist_wheel
        DEPENDS _npu_ops _npu_device_backends
        VERBATIM)

install(CODE "execute_process(COMMAND ${PYTHON_BIN_PATH} -m pip install ${CMAKE_BINARY_DIR}/dist/python/dist/npu_device-0.1-py3-none-any.whl --upgrade)")
