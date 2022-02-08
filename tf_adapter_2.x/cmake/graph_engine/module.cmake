add_library(ge_libs INTERFACE)

if(DEFINED ASCEND_INSTALLED_PATH)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../inc/graphengine/inc)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../inc/graphengine/inc/external)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../inc/metadef/inc)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../inc/metadef/inc/external)
    target_link_libraries(ge_libs INTERFACE
            ${ASCEND_INSTALLED_PATH}/compiler/lib64/libge_runner.so
            ${ASCEND_INSTALLED_PATH}/compiler/lib64/libfmk_parser.so)
else()
    include_directories(${ASCEND_CI_BUILD_DIR}/graphengine/inc)
    include_directories(${ASCEND_CI_BUILD_DIR}/graphengine/inc/external)
    include_directories(${ASCEND_CI_BUILD_DIR}/metadef/inc)
    include_directories(${ASCEND_CI_BUILD_DIR}/metadef/inc/external)
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
    )

    set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

    add_library(ge_runner SHARED ${fake_sources})
    add_library(fmk_parser SHARED ${fake_sources})
    target_link_libraries(ge_libs INTERFACE
            ge_runner
            fmk_parser)
endif()
