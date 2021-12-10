file(GLOB_RECURSE TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/../../npu_device/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../../../tf_adapter/ops/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/testcases/*.cpp
        )

add_executable(adapter2_ut_main ${TEST_SOURCES})

execute_process(COMMAND ${PYTHON_BIN_PATH} -c "from distutils import sysconfig; import os.path as op; \
                v=sysconfig.get_config_vars(); print(op.join(v['LIBPL'], v['LDLIBRARY']))"
        OUTPUT_VARIABLE PYTHON_LIBS)
string(STRIP ${PYTHON_LIBS} PYTHON_LIBS)

target_compile_options(adapter2_ut_main PRIVATE
        -O0 -g --coverage -fprofile-arcs -ftest-coverage)
target_link_libraries(adapter2_ut_main PRIVATE
        tensorflow_libs gtest_main acl_stub ge_stub ${PYTHON_LIBS} -lrt -ldl -lutil)

add_custom_target(adapter2_ut ALL
        COMMAND ./adapter2_ut_main
        COMMAND lcov -c -q -d . -o ut_coverage.info
        COMMAND lcov -e ut_coverage.info "*tf_adapter_2.x/npu_device*" -o ut_coverage.info
        DEPENDS adapter2_ut_main
        VERBATIM)
