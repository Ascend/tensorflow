option(ENABLE_OPEN_SRC "Enable graphengine compile in opensource." FALSE)

set(TFADAPTER_DIR ${CMAKE_CURRENT_LIST_DIR})

if (ENABLE_OPEN_SRC)
    cmake_minimum_required(VERSION 3.14)
    project(TFAdapter)

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_C_FLAGS "-O2 -DNDEBUG -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -s -pipe ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -DNDEBUG -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -s -pipe ${CMAKE_CXX_FLAGS}")

    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

    # build external prjects
    if(DEFINED ENV{D_PKG_SERVER})
        set(TF_PKG_SERVER $ENV{D_PKG_SERVER})
        message("Download packages from PKG server")
    endif()

    include(${CMAKE_CURRENT_LIST_DIR}/cmake/nlohmann_json.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/secure_c.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/tensorflow.cmake)
    include_directories(${CMAKE_CURRENT_LIST_DIR})
    include_directories(${CMAKE_CURRENT_LIST_DIR}/inc)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/inc/toolchain)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/inc/external)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/inc/soft_dp)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/inc/graphengine/inc)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/inc/graphengine/inc/external)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/inc/metadef/inc)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/inc/metadef/inc/external)

    if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS OR NOT EXISTS
            ${CMAKE_CURRENT_LIST_DIR}/tools/LINK_FLAGS OR NOT EXISTS
            ${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH OR NOT EXISTS
            ${CMAKE_CURRENT_LIST_DIR}/tools/SWIG_BIN_PATH)
        message(FATAL_ERROR "No validate configuration found. Did you forget to configure first?")
    endif ()

    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS" COMPILE_FLAGS)
    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/LINK_FLAGS" LINK_FLAGS)
    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH" PYTHON_BIN_PATH)
    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/SWIG_BIN_PATH" SWIG_BIN_PATH)
    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/UT_LINK_FLAGS" UT_LINK_FLAGS)

    if(ENABLE_TFADAPTER_UT)
        add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tf_adapter/tests)
    endif()

    file(COPY ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/python DESTINATION ${CMAKE_BINARY_DIR}/dist)
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/convert_tf2npu DESTINATION ${CMAKE_BINARY_DIR}/dist/python/npu_bridge)
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/swig DESTINATION ${CMAKE_BINARY_DIR}/dist)
    execute_process(COMMAND ${SWIG_BIN_PATH} -python -c++ -threads ${CMAKE_BINARY_DIR}/dist/swig/ge_plugin.i)
    file(COPY ${CMAKE_BINARY_DIR}/dist/swig/tf_adapter.py DESTINATION ${CMAKE_BINARY_DIR}/dist/python/npu_bridge)

    file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/common/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/kernels/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/ops/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/optimizers/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/util/*.cc
    )
    add_library(_tf_adapter SHARED ${SOURCES} ${CMAKE_BINARY_DIR}/dist/swig/ge_plugin_wrap.cxx)

    foreach (COMPILE_FLAG ${COMPILE_FLAGS})
        target_compile_options(_tf_adapter PUBLIC "${COMPILE_FLAG}")
    endforeach (COMPILE_FLAG)

    target_link_libraries(_tf_adapter PUBLIC "dl")
    foreach (LINK_FLAG ${LINK_FLAGS})
        target_link_libraries(_tf_adapter PUBLIC "${LINK_FLAG}")
    endforeach (LINK_FLAG)

    target_compile_definitions(_tf_adapter PUBLIC
        LOG_CPP
    )

    set_target_properties(_tf_adapter PROPERTIES PREFIX "")

    set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/dist/python/npu_bridge)

    add_custom_command(TARGET _tf_adapter
            POST_BUILD
            COMMAND cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} setup.py bdist_wheel
            VERBATIM
            )
else()
    set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(CMAKE_VERBOSE_MAKEFILE ON)

    # rewrite __FILE__ to avoid absolute path
    set(CMAKE_C_COMPILE_OBJECT "<CMAKE_C_COMPILER> <DEFINES> -D__FILE__='\"$(subst ${CMAKE_CURRENT_SOURCE_DIR}/,,$(abspath $<))\"' -Wno-builtin-macro-redefined  <INCLUDES> <FLAGS> -o <OBJECT>  -c <SOURCE>")
    set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES>  -D__FILE__='\"$(subst ${CMAKE_CURRENT_SOURCE_DIR}/,,$(abspath $<))\"' -Wno-builtin-macro-redefined  <INCLUDES> <FLAGS> -o <OBJECT>  -c <SOURCE>")

    add_custom_command(
        OUTPUT ${BASE_DIR}/tensorflow_framework.cc
        COMMAND touch ${BASE_DIR}/tensorflow_framework.cc
    )

    add_library(tensorflow_framework SHARED
        ${BASE_DIR}/tensorflow_framework.cc
    )

    #rename libtensorflow_framework.so to libtensorflow_framework.so.1
    set_target_properties(tensorflow_framework
        PROPERTIES
        VERSION 1
    )

    add_custom_command(
        OUTPUT ${BASE_DIR}/pywrap_tensorflow_internal.cc
        COMMAND touch ${BASE_DIR}/pywrap_tensorflow_internal.cc
    )

    add_library(pywrap_tensorflow_internal SHARED
        ${BASE_DIR}/pywrap_tensorflow_internal.cc
    )

    #rename libpywrap_tensorflow_internal.so to _pywrap_tensorflow_internal.so
    set_target_properties(pywrap_tensorflow_internal
        PROPERTIES
        PREFIX _
    )

    file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/common/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/kernels/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/ops/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/optimizers/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/tf_adapter/util/*.cc
    )

    add_custom_target(tensorflow_source ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tensorflow_source.timestamp)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tensorflow_source.timestamp
                       COMMAND echo "cp tensorflow1.15 source begin:"
                       COMMAND rm -rf ${BASE_DIR}/tensorflow_15 && mkdir -p ${BASE_DIR}/tensorflow_15
                       COMMAND cp -rfL ${TOP_DIR}/open_source/tensorflow ${BASE_DIR}/tensorflow_15 || echo skip
                       COMMAND cd ${BASE_DIR}/tensorflow_15/tensorflow && git checkout . && git fetch --all --tags && git checkout tags/v1.15.5
                       COMMAND echo "end cp tensorflow1.15 source"
                       DEPENDS ${TOP_DIR}/open_source/tensorflow)

    add_library(tf_adapter SHARED
        ${SOURCES}
        ${BASE_DIR}/tf_adapter/util/ge_plugin_wrap.cxx
    )

    add_dependencies(tf_adapter tensorflow_source)

    target_include_directories(tf_adapter PRIVATE
        ${BASE_DIR}/
        ${TOP_DIR}/inc/
        ${TOP_DIR}/inc/external/
        ${TOP_DIR}/inc/common/
        ${TOP_DIR}/soft_dp/
        ${TOP_DIR}/ace/execfwk/soft_dp/
        ${TOP_DIR}/graphengine/inc/
        ${TOP_DIR}/graphengine/inc/external/
        ${TOP_DIR}/metadef/inc/
        ${TOP_DIR}/metadef/inc/external/
        ${TOP_DIR}/libc_sec/include/
        ${TOP_DIR}/abl/libc_sec/include/
        ${TOP_DIR}/third_party/json/include/
        ${TOP_DIR}/open_source/json/include/
        ${BASE_DIR}/tensorflow_15/tensorflow
        /opt/buildtools/tensorflow-1.15.5/tensorflow_core/include/
        /opt/buildtools/tensorflow-1.15.5/tensorflow-1.15.5.data/purelib/tensorflow_core/include/
        ${HI_PYTHON_INC}/
    )

    target_compile_options(tf_adapter PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
        -O2
        -DNDEBUG
        -ftrapv
        -Wno-deprecated-declarations
    )
    target_compile_definitions(tf_adapter PUBLIC
        _FORTIFY_SOURCE=2
        LOG_CPP
    )

    target_link_libraries(tf_adapter PUBLIC
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:aoe_headers>
        $<BUILD_INTERFACE:soft_dp_headers>
        -Wl,--no-as-needed
        c_sec
        ge_runner
        datatransfer
        fmk_parser
        indextransform
        tensorflow_framework
        pywrap_tensorflow_internal
        -Wl,--as-needed
        -s
    )

    # rename libtf_adapter.so to _tf_adapter.so
    set_target_properties(tf_adapter
        PROPERTIES
        PREFIX _
    )
    ###################################################################################################
    add_custom_command(
        OUTPUT ${BASE_DIR}/tf_adapter/util/ge_plugin_wrap.cxx
        COMMAND echo "before generate swig files"
        && swig -c++ -threads -python ${BASE_DIR}/tf_adapter/swig/ge_plugin.i
        && mv ${BASE_DIR}/tf_adapter/swig/ge_plugin_wrap.cxx ${BASE_DIR}/tf_adapter/util/
        && mv ${BASE_DIR}/tf_adapter/swig/tf_adapter.py ${BASE_DIR}/tf_adapter/python/npu_bridge/
        && echo "end generate swig files"
    )

    ###################################################################################################
    add_custom_target(
        hw_npu_bridge_1.15.0 ALL
        DEPENDS tensorflow_framework pywrap_tensorflow_internal tf_adapter SoftDp ${CMAKE_CURRENT_BINARY_DIR}/npu_bridge-1.15.0-py3-none-any.whl ${CMAKE_CURRENT_BINARY_DIR}/npu_device-0.1-py3-none-any.whl
    )

    # package _tf_adapter.so and libSoftDp.so in npu_bridge-xxx.whl
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/npu_bridge-1.15.0-py3-none-any.whl
        COMMAND echo "package whl start"
        && mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/wheel
        && cp -r ${BASE_DIR}/tf_adapter/python/. ${CMAKE_CURRENT_BINARY_DIR}/wheel
        && cp -r ${BASE_DIR}/convert_tf2npu ${CMAKE_CURRENT_BINARY_DIR}/wheel/npu_bridge
        && cp -r ${CMAKE_CURRENT_BINARY_DIR}/_tf_adapter.so ${CMAKE_CURRENT_BINARY_DIR}/wheel/npu_bridge
        && cp -r $<TARGET_FILE:SoftDp> ${CMAKE_CURRENT_BINARY_DIR}/wheel/npu_bridge
        # && cp -r ${CMAKE_CURRENT_BINARY_DIR}/../../../../../soft_dp/libSoftDp.so ${CMAKE_CURRENT_BINARY_DIR}/wheel/npu_bridge
    	&& cd ${CMAKE_CURRENT_BINARY_DIR}/wheel
        && ${HI_PYTHON} setup.py bdist_wheel >/dev/null
        && cp -f dist/npu_bridge-1.15.0-py3-none-any.whl ${CMAKE_CURRENT_BINARY_DIR}/
        && rm -rf ${BASE_DIR}/pywrap_tensorflow_internal.cc
        && rm -rf ${BASE_DIR}/tensorflow_framework.cc
        && rm -rf ${BASE_DIR}/_pywrap_tensorflow_internal.so
        && rm -rf ${BASE_DIR}/libpywrap_tensorflow_internal.so
        && rm -rf ${BASE_DIR}/libtensorflow_framework.so.1
        && rm -rf ${BASE_DIR}/libtensorflow_framework.so
        && rm -rf ${BASE_DIR}/tensorflow_15/
        && echo "package whl end"
    )
    ###################################### Tensorflow 2.x ###########################################
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/npu_device-0.1-py3-none-any.whl
        COMMAND echo "package ascend adapter for tensorflow 2.x start"
        && cd ${CMAKE_CURRENT_LIST_DIR}/tf_adapter_2.x/
        && rm -rf build && mkdir build && cd build
        && cmake .. -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DPYTHON_BIN_PATH=${HI_PYTHON_PATH} -DPYTHON_INCLUDE_DIR=${HI_PYTHON_INC} -DASCEND_CI_BUILD_DIR=${TOP_DIR}
        && make -j32
        && cp -f dist/python/dist/npu_device-0.1-py3-none-any.whl ${CMAKE_CURRENT_BINARY_DIR}/
        && echo "package ascend adapter for tensorflow 2.x end"
    )
    ###################################################################################################
    set(INSTALL_BASE_DIR "")
    set(INSTALL_LIBRARY_DIR lib)

    include(CMakePackageConfigHelpers)

    install(TARGETS tf_adapter OPTIONAL
        EXPORT tfplugin-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    )

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/npu_bridge-1.15.0-py3-none-any.whl ${CMAKE_CURRENT_BINARY_DIR}/npu_device-0.1-py3-none-any.whl OPTIONAL
        DESTINATION ${INSTALL_LIBRARY_DIR}
    )
endif()
