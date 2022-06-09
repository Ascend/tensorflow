file(GLOB_RECURSE TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/../../npu_device/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../../../tf_adapter/ops/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcases/*.cpp
        )

add_executable(adapter2_ut_main ${TEST_SOURCES})

target_compile_options(adapter2_ut_main PRIVATE
        -O0 -g --coverage -fprofile-arcs -ftest-coverage)

target_link_libraries(adapter2_ut_main PRIVATE
        tensorflow_libs gtest_main acl_stub aoe_stub ge_stub c_sec ${PYTHON_LD_LIBRARY} -lrt -ldl -lutil)

add_custom_target(adapter2_ut ALL
        COMMAND ./adapter2_ut_main
        COMMAND lcov -c -q -d . -o ut_addons.coverage
        COMMAND lcov -q -e ut_addons.coverage "*tf_adapter_2.x/npu_device*" -o ut_addons.coverage
        COMMAND lcov -o ut.coverage -a ${CMAKE_BINARY_DIR}/st/st.coverage -a ut_addons.coverage
        DEPENDS adapter2_ut_main adapter2_st
        VERBATIM)
