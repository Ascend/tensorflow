#!/bin/bash
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

set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
RELEASE_PATH="${BASE_PATH}/output"
export BUILD_PATH="${BASE_PATH}/build"
INSTALL_PATH="${BUILD_PATH}/install"
CMAKE_PATH="${BUILD_PATH}/tfadapter"
RELEASE_TARGET="tfadapter.tar"

# print usage message
usage() {
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-v] [-g] [-u]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -j[n] Set the number of threads used to build CANN, default is 8"
  echo "    -v Verbose"
  echo "    -g GCC compiler prefix, used to specify the compiler toolchain"
  echo "    -u TF_adapter utest"
  echo "to be continued ..."
}

logging() {
  echo "[INFO] $@"
}

# parse and set optionss
checkopts() {
  VERBOSE=""
  THREAD_NUM=8
  GCC_PREFIX=""
  ENABLE_TFADAPTER_UT="off"
  ENABLE_TFADAPTER_ST="off"
  # Process the options
  while getopts 'hj:vusg:' opt
  do
    case "${opt}" in
      h) usage
         exit 0 ;;
      j) THREAD_NUM=$OPTARG ;;
      v) VERBOSE="VERBOSE=1" ;;
      g) GCC_PREFIX=$OPTARG ;;
      u) ENABLE_TFADAPTER_UT="on" ;;
      s) ENABLE_TFADAPTER_ST="on" ;;
      *) logging "Undefined option: ${opt}"
         usage
         exit 1 ;;
    esac
  done
}

# mkdir directory
mk_dir() {
  local dir_name="$1"
  mkdir -pv "${dir_name}"
  logging "Created dir ${dir_name}"
}

# create build path
build_tfadapter() {
  if [[ "X$ENABLE_TFADAPTER_UT" = "Xon" ]]; then
    export OPEN_UT=1
  fi
  logging "Create build directory and build tfadapter"
  cd "${BASE_PATH}" && ./configure
  CMAKE_ARGS="-DENABLE_OPEN_SRC=True -DBUILD_PATH=$BUILD_PATH -DCMAKE_INSTALL_PREFIX=${RELEASE_PATH}"
  if [[ "$GCC_PREFIX" != "" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DGCC_PREFIX=$GCC_PREFIX"
  fi
  if [[ "X$ENABLE_TFADAPTER_UT" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TFADAPTER_UT=ON"
  fi
  if [[ "X$ENABLE_TFADAPTER_ST" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TFADAPTER_ST=ON"
  fi
  logging "CMake Args: ${CMAKE_ARGS}"

  mk_dir "${CMAKE_PATH}"
  cd "${CMAKE_PATH}" && cmake ${CMAKE_ARGS} ../..
  if [ 0 -ne $? ]
  then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    return 1
  fi
  if [[ "X$ENABLE_TFADAPTER_UT" = "Xon" ]]; then
    make tfadapter_utest ${VERBOSE} -j${THREAD_NUM}
    logging "Build tfadapter utest success!"
  elif [[ "X$ENABLE_TFADAPTER_ST" = "Xon" ]]; then
    make tfadapter_stest ${VERBOSE} -j${THREAD_NUM}
    logging "Build tfadapter stest success!"
  else
    make ${VERBOSE} -j${THREAD_NUM}
    logging "tfadapter build success!"
    chmod +x "${BASE_PATH}/tf_adapter_2.x/CI_Build"
    sh "${BASE_PATH}/tf_adapter_2.x/CI_Build"
  fi
}

release_tfadapter() {
  logging "Create output directory"
  cd ${CMAKE_PATH}/dist/python/dist && mkdir -p tfplugin/bin && cp -r "${BASE_PATH}/script" tfplugin/ && mv npu_bridge-*.whl tfplugin/bin && mv "${BASE_PATH}/tf_adapter_2.x/build/dist/python/dist/npu_device-0.1-py3-none-any.whl" tfplugin/bin && tar cfz "${RELEASE_TARGET}" * && mv "${RELEASE_TARGET}" "${RELEASE_PATH}"
}

release_tfadapter_for_cann() {
  logging "Create output directory"
  cd ${CMAKE_PATH}/dist/python/dist && mkdir -p fwkplugin/bin && cp -r "${BASE_PATH}/script" fwkplugin/ && mv npu_bridge-*.whl fwkplugin/bin && mv "${BASE_PATH}/tf_adapter_2.x/build/dist/python/dist/npu_device-0.1-py3-none-any.whl" fwkplugin/bin && tar cfz "${RELEASE_TARGET}" * && mv "${RELEASE_TARGET}" "${RELEASE_PATH}"
}

main() {
  checkopts "$@"
  # tfadapter build start
  logging "---------------- tfadapter build start ----------------"
  ${GCC_PREFIX}g++ -v
  mk_dir "${RELEASE_PATH}"
  build_tfadapter
  if [[ "X$ENABLE_TFADAPTER_UT" = "Xoff" ]] && [[ "X$ENABLE_TFADAPTER_ST" = "Xoff" ]]; then
    if [[ "X$ALL_IN_ONE_ENABLE" = "X1" ]]; then
      release_tfadapter_for_cann
    else
      release_tfadapter
    fi
  fi
  if [[ "X$ENABLE_TFADAPTER_UT" = "Xon" ]]; then
    cd ${BASE_PATH}
    export ASCEND_OPP_PATH=${BASE_PATH}/tf_adapter/tests/depends/support_json
    export PRINT_MODEL=1
    export LD_LIBRARY_PATH=${CMAKE_PATH}/tf_adapter/tests/depends/aoe/:$LD_LIBRARY_PATH
    RUN_TEST_CASE=${CMAKE_PATH}/tf_adapter/tests/ut/tfadapter_utest && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
      echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
      echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
      exit 1;
    fi
    logging "Generating coverage statistics, please wait..."
    rm -rf ${BASE_PATH}/coverage
    mkdir ${BASE_PATH}/coverage
    lcov -c -d ${CMAKE_PATH}/tf_adapter/tests/ut/ -o coverage/tmp.info
    lcov -r coverage/tmp.info '*/tests/*' '*/nlohmann_json-src/*' '*/tensorflow-src/*' \
      '*/inc/*' '*/output/*' '*/usr/*' '*/Eigen/*' '*/absl/*' '*/google/*' '*/tensorflow/core/*' \
      -o coverage/coverage.info
  fi
  if [[ "X$ENABLE_TFADAPTER_ST" = "Xon" ]]; then
    cd ${BASE_PATH}
    export ASCEND_OPP_PATH=${BASE_PATH}/tf_adapter/tests/depends/support_json
    export PRINT_MODEL=1
    export LD_LIBRARY_PATH=${CMAKE_PATH}/tf_adapter/tests/depends/aoe/:$LD_LIBRARY_PATH
    RUN_TEST_CASE=${CMAKE_PATH}/tf_adapter/tests/st/tfadapter_stest && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
      echo "!!! ST FAILED, PLEASE CHECK YOUR CHANGES !!!"
      echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
      exit 1;
    fi
    logging "Generating coverage statistics, please wait..."
    rm -rf ${BASE_PATH}/coverage
    mkdir ${BASE_PATH}/coverage
    lcov -c -d ${CMAKE_PATH}/tf_adapter/tests/st/ -o coverage/tmp.info
    lcov -r coverage/tmp.info '*/tests/*' '*/nlohmann_json-src/*' '*/tensorflow-src/*' \
      '*/inc/*' '*/output/*' '*/usr/*' '*/Eigen/*' '*/absl/*' '*/google/*' '*/tensorflow/core/*' \
      -o coverage/coverage.info
  fi
  logging "---------------- tfadapter build finished ----------------"
}

main "$@"
