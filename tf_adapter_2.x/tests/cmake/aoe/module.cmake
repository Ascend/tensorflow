add_library(aoe_tuning INTERFACE)

add_library(aoe_stub STATIC ${CMAKE_CURRENT_LIST_DIR}/../../stub/aoe_stub.cpp)

target_link_libraries(aoe_tuning INTERFACE aoe_stub)
