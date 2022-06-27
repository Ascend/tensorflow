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

project(ge_runner_stub)

file(GLOB_RECURSE SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "src/*.cc"
)

set(BASE_DIR ${TOP_DIR}/../)

add_library(ge_runner_stub SHARED ${SRC_FILES})

target_include_directories(ge_runner_stub PRIVATE
    ${TFADAPTER_DIR}/inc/
    ${TF_INSTALL_LIB_PATH}/include/
    ${TFADAPTER_DIR}/inc/external/
    ${TFADAPTER_DIR}/inc/common/
    ${TFADAPTER_DIR}/inc/graphengine/inc/
    ${TFADAPTER_DIR}/inc/graphengine/inc/external/
    ${TFADAPTER_DIR}/inc/graphengine/inc/framework/
    ${TFADAPTER_DIR}/metadef/inc/
    ${TFADAPTER_DIR}/metadef/inc/external/
    ${TFADAPTER_DIR}/tf_adapter/tests/depends/ascendcl/src
    ${TFADAPTER_DIR}/tf_adapter/common
)

target_link_libraries(ge_runner_stub PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)