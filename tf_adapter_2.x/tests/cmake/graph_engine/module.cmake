add_library(ge_libs INTERFACE)

add_library(ge_stub STATIC
        ${CMAKE_CURRENT_LIST_DIR}/../../stub/ge_stub.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../../stub/parser_stub.cpp)

target_include_directories(ge_stub PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../../npu_device/core
        ${CMAKE_CURRENT_LIST_DIR}/../../tests/stub/include)
target_link_libraries(ge_libs INTERFACE ge_stub)
