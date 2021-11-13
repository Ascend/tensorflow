cmake_minimum_required(VERSION 3.14)
project(AscendAdapter2Test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "-D_FORTIFY_SOURCE=2 -O0 -g -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -fno-common -pipe ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-D_FORTIFY_SOURCE=2 -O0 -g -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -fno-common -pipe ${CMAKE_CXX_FLAGS}")
set(CMAKE_SKIP_RPATH FALSE)

IF (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    add_definitions(-Wno-builtin-macro-redefined)
ENDIF ()

if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/TF_INSTALLED_PATH OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH)
    message(FATAL_ERROR "No validate configuration found. Did you forget to configure first?")
endif ()

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/TF_INSTALLED_PATH" TF_INSTALLED_PATH)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH" PYTHON_BIN_PATH)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS" CUSTOM_COMPILE_FLAGS)
foreach (COMPILE_FLAG ${CUSTOM_COMPILE_FLAGS})
    set(CMAKE_C_FLAGS "${COMPILE_FLAG} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${COMPILE_FLAG} ${CMAKE_CXX_FLAGS}")
endforeach (COMPILE_FLAG)

file(GLOB_RECURSE TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/../npu_device/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/testcases/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/stub/*.cpp
        )

include_directories(${CMAKE_CURRENT_LIST_DIR}/../npu_device/core)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../inc/external) # just for acl
include_directories(${CMAKE_CURRENT_LIST_DIR}/stub/include)

add_executable(adapter2_test_main ${TEST_SOURCES})

include(${CMAKE_CURRENT_LIST_DIR}/cmake/gtest.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/tensorflow/module.cmake)
execute_process(COMMAND ${PYTHON_BIN_PATH} -c "from distutils import sysconfig; import os.path as op; \
                v=sysconfig.get_config_vars(); print(op.join(v['LIBDIR'], v['LDLIBRARY']))"
        OUTPUT_VARIABLE PYTHON_LIBS)
string(STRIP ${PYTHON_LIBS} PYTHON_LIBS)

target_link_libraries(adapter2_test_main PRIVATE
        tensorflow_libs gtest_main ${PYTHON_LIBS} -lrt -ldl -lutil -lgcov)

add_custom_target(adapter2_ut ALL
        COMMAND ./adapter2_test_main
        DEPENDS adapter2_test_main
        VERBATIM)

add_custom_target(adapter2_st ALL
        COMMAND ./adapter2_test_main --gtest_filter=ST_*
        DEPENDS adapter2_test_main
        VERBATIM)
