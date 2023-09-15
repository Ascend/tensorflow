add_library(aoe_libs INTERFACE)

if (DEFINED ASCEND_INSTALLED_PATH)
    include_directories(${ASCEND_INSTALLED_PATH}/opensdk/opensdk/include/aoe)
    target_link_libraries(aoe_libs INTERFACE
            ${ASCEND_INSTALLED_PATH}/tools/aoe/lib64/libaoe_tuning.so)
else ()
    include_directories(${ASCEND_CI_BUILD_DIR}/asl/aoetools/inc/aoe)
    include_directories(${ASCEND_CI_BUILD_DIR}/abl/slog/inc)
    include_directories(${ASCEND_CI_BUILD_DIR}/abl/msprof/inc)
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
    )

    set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

    add_library(aoe_tuning SHARED ${fake_sources})
    target_link_libraries(aoe_libs INTERFACE
            aoe_tuning)
endif ()