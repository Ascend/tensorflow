add_library(aoe_libs INTERFACE)

add_library(aoe_stub STATIC ${CMAKE_CURRENT_LIST_DIR}/../../stub/aoe_stub.cpp)

target_include_directories(aoe_stub PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../../../../inc/toolchain/tuning_tool)

target_link_libraries(aoe_libs INTERFACE aoe_stub)
