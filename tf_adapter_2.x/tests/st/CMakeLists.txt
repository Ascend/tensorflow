file(COPY ${CMAKE_CURRENT_LIST_DIR}/../../python DESTINATION ${CMAKE_BINARY_DIR}/dist)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/adapter2_st.py DESTINATION ${CMAKE_BINARY_DIR}/dist/python)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/dist/python/npu_device)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../npu_device npu_device)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../npu_ops npu_ops)

add_custom_target(adapter2_st ALL
        COMMAND export NPU_DEBUG=true && export NPU_DUMP_GRAPH=true && cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest adapter2_st
        COMMAND lcov -c -q -d . -o st_p1.coverage && lcov -e st_p1.coverage "*npu_device*" -o st_p1.coverage
        COMMAND export GE_USE_STATIC_MEMORY=1 && cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest adapter2_st.Adapter2St.test_dropout_v3
        COMMAND lcov -c -q -d . -o st_p2.coverage && lcov -e st_p2.coverage "*npu_device*" -o st_p2.coverage
        COMMAND lcov -o st.coverage -a st_p1.coverage -a st_p2.coverage
        DEPENDS _npu_ops _npu_device_backends
        VERBATIM)
