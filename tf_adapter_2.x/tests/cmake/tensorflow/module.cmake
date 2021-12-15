add_library(tensorflow_libs INTERFACE)

SET(TF_INCLUDE_DIR ${TF_INSTALLED_PATH}/include)
target_link_libraries(tensorflow_libs INTERFACE
        ${TF_INSTALLED_PATH}/python/_pywrap_tensorflow_internal.so
        ${TF_INSTALLED_PATH}/libtensorflow_framework.so.2)

include_directories(${TF_INCLUDE_DIR})
include_directories(${TF_INCLUDE_DIR}/external/farmhash_archive/src)
include_directories(${TF_INCLUDE_DIR}/external/pybind11/_virtual_includes/pybind11)