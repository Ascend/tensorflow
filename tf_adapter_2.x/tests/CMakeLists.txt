cmake_minimum_required(VERSION 3.14)
project(AscendAdapter2Test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "-D_FORTIFY_SOURCE=2 -O0 -g --coverage -DNDEBUG -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -fno-common -pipe ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-D_FORTIFY_SOURCE=2 -O0 -g --coverage -DNDEBUG -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -fno-common -pipe ${CMAKE_CXX_FLAGS}")
set(CMAKE_SKIP_RPATH FALSE)

IF (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    add_definitions(-Wno-builtin-macro-redefined)
ENDIF ()

link_libraries(-lgcov)

if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/TF_INSTALLED_PATH OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_LD_LIBRARY)
    message(FATAL_ERROR "No validate configuration found. Did you forget to configure first?")
endif ()

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/TF_INSTALLED_PATH" TF_INSTALLED_PATH)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH" PYTHON_BIN_PATH)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS" CUSTOM_COMPILE_FLAGS)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_LD_LIBRARY" PYTHON_LD_LIBRARY)
foreach (COMPILE_FLAG ${CUSTOM_COMPILE_FLAGS})
    set(CMAKE_C_FLAGS "${COMPILE_FLAG} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${COMPILE_FLAG} ${CMAKE_CXX_FLAGS}")
endforeach (COMPILE_FLAG)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../../inc/external) # just for acl
include_directories(${CMAKE_CURRENT_LIST_DIR}/stub/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../npu_device/core)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../inc/metadef/inc)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/gtest.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/acl/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/tensorflow/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/graph_engine/module.cmake)

add_subdirectory(ut)
add_subdirectory(st)
