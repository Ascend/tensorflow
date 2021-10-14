# Copyright 2019-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

project(tests CXX C)

include(${TFADAPTER_DIR}/cmake/tests/gtest.cmake)
include(${TFADAPTER_DIR}/cmake/tests/securec.cmake)
include(${TFADAPTER_DIR}/cmake/tests/intf_pub_linux.cmake)

execute_process(COMMAND ${PYTHON_BIN_PATH} -c "import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')"
                OUTPUT_VARIABLE TF_INSTALL_LIB_PATH)

execute_process(COMMAND ${PYTHON_BIN_PATH} -c "from distutils import sysconfig; import os.path as op; \
                v=sysconfig.get_config_vars(); print(op.join(v['LIBDIR'], v['LDLIBRARY']))"
                OUTPUT_VARIABLE PYTHON_LIB_PATH)
message("python lib path:${PYTHON_LIB_PATH}")

add_subdirectory(depends/mmpa)
add_subdirectory(depends/indextransform)
add_subdirectory(depends/alog)
add_subdirectory(depends/datatransfer)
add_subdirectory(depends/ge_runner)
add_subdirectory(depends/aoe)

if (ENABLE_TFADAPTER_UT)
    add_subdirectory(ut)
endif()

if (ENABLE_TFADAPTER_ST)
    add_subdirectory(st)
endif()
