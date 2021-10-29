# Copyright 2019-2020 Huawei Technologies Co., Ltd
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

project(st_optimizers)

set(CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE ST_SOURCES
    ${TFADAPTER_DIR}/tf_adapter/common/*.cc
    ${TFADAPTER_DIR}/tf_adapter/ops/*.cc
    ${TFADAPTER_DIR}/tf_adapter/optimizers/*.cc
    ${TFADAPTER_DIR}/tf_adapter/util/*.cc
    ${TFADAPTER_DIR}/tf_adapter/kernels/*.cc
    "optimizers/testcase/om_partition_subgraphs_pass_test.cc"
    "kernels/testcase/ocr_ops_test.cc"
)

add_executable(tfadapter_stest
               "main.cc"
               ${ST_SOURCES}
               )

target_include_directories(tfadapter_stest PRIVATE
    ${TF_INSTALL_LIB_PATH}/include/
)

target_compile_options(tfadapter_stest PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

target_compile_definitions(tfadapter_stest PRIVATE
    _FORTIFY_SOURCE=2
    LOG_CPP
)

foreach (ST_LINK_FLAG ${ST_LINK_FLAGS})
    target_link_libraries(tfadapter_stest PUBLIC "${ST_LINK_FLAG}")
endforeach (ST_LINK_FLAG)

string(STRIP ${PYTHON_LIB_PATH} PYTHON_LIB_PATH)
message("python lib path ${PYTHON_LIB_PATH}")

add_dependencies(tfadapter_stest aoe_tuning)

target_link_libraries(tfadapter_stest PUBLIC
    $<BUILD_INTERFACE:intf_pub>
    gtest gtest_main c_sec mmpa_stub indextransform_stub alog_stub datatransfer_stub ge_runner_stub
    ${PYTHON_LIB_PATH} -lrt -ldl -lgcov
)

