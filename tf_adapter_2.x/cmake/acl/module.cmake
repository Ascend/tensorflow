add_library(acl_libs INTERFACE)

if(DEFINED ASCEND_INSTALLED_PATH)
    if(DEFINED ENV{ALL_IN_ONE_ENABLE})
        include_directories(${ASCEND_INSTALLED_PATH}/compiler/include)
        include_directories(${ASCEND_INSTALLED_PATH}/compiler/include/acl/error_codes)
        include_directories(${ASCEND_INSTALLED_PATH}/runtime/include)
        target_link_libraries(acl_libs INTERFACE
                ${ASCEND_INSTALLED_PATH}/runtime/lib64/libascendcl.so
                ${ASCEND_INSTALLED_PATH}/compiler/lib64/libacl_tdt_channel.so
                ${ASCEND_INSTALLED_PATH}/compiler/lib64/libacl_op_compiler.so)
    else()
        include_directories(${ASCEND_INSTALLED_PATH}/fwkacllib/include)
        target_link_libraries(acl_libs INTERFACE
                ${ASCEND_INSTALLED_PATH}/fwkacllib/lib64/libascendcl.so
                ${ASCEND_INSTALLED_PATH}/fwkacllib/lib64/libacl_tdt_channel.so
                ${ASCEND_INSTALLED_PATH}/fwkacllib/lib64/libacl_op_compiler.so)
    endif()
else()
    include_directories(${ASCEND_CI_BUILD_DIR}/inc/external)
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
    )

    set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

    add_library(ascendcl SHARED ${fake_sources})
    add_library(acl_op_compiler SHARED ${fake_sources})
    add_library(acl_tdt_channel SHARED ${fake_sources})
    target_link_libraries(acl_libs INTERFACE
            ascendcl
            acl_op_compiler
            acl_tdt_channel)
endif()
