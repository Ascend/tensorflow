add_library(acl_libs INTERFACE)

add_library(acl_stub STATIC ${CMAKE_CURRENT_LIST_DIR}/../../stub/acl_stub.cpp)

target_include_directories(acl_stub PRIVATE
        ${ASCEND_INSTALLED_PATH}/opensdk/opensdk/include/ascendcl/external
        ${CMAKE_CURRENT_LIST_DIR}/../../stub/include)

target_link_libraries(acl_libs INTERFACE acl_stub)
