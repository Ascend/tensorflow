#!/bin/bash
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -e
COMMON_INSTALL_DIR=/usr/local/Ascend
COMMON_INSTALL_TYPE=full
DEFAULT_USERNAME=HwHiAiUser
DEFAULT_USERGROUP=HwHiAiUser
is_quiet=n
pylocal=n
install_for_all=n
sourcedir="$PWD/tfplugin"
curpath="$(dirname "$(readlink -f "$0")")"
common_parse_dir=$COMMON_INSTALL_DIR
common_parse_type=$COMMON_INSTALL_TYPE

get_install_param() {
    local _key="$1"
    local _file="$2"
    local _param

    if [ ! -f "${_file}" ];then
        exit 1
    fi
    install_info_key_array=("Tfplugin_Install_Type" "Tfplugin_UserName" "Tfplugin_UserGroup" "Tfplugin_Install_Path_Param")
    for key_param in "${install_info_key_array[@]}"; do
        if [ "${key_param}" == "${_key}" ]; then
            _param=$(grep -r "${_key}=" "${_file}" | cut -d"=" -f2-)
            break
        fi
    done
    echo "${_param}"
}

if [ "$1" ];then
    common_parse_dir="$2"
    common_parse_type=$3
    is_quiet=$4
    pylocal=$5
    install_for_all=$6
fi

install_info="${common_parse_dir}/tfplugin/ascend_install.info"

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi

log_file="${log_dir}/ascend_install.log"

username=$(get_install_param "Tfplugin_UserName" "${install_info}")
usergroup=$(get_install_param "Tfplugin_UserGroup" "${install_info}")
if [ "$username" == "" ]; then
    username="$DEFAULT_USERNAME"
    usergroup="$DEFAULT_USERGROUP"
fi

log() {
    local cur_date_=$(date +"%Y-%m-%d %H:%M:%S")
    local log_type_=${1}
    local msg_="${2}"
    if [ "$log_type_" == "INFO" ]; then
        local log_format_="[Tfplugin] [$cur_date_] [$log_type_]: ${msg_}"
        echo "${log_format_}"
    elif [ "$log_type_" == "WARNING" ]; then
        local log_format_="[Tfplugin] [$cur_date_] [$log_type_]: ${msg_}"
        echo "${log_format_}"
    elif [ "$log_type_" == "ERROR" ]; then
        local log_format_="[Tfplugin] [$cur_date_] [$log_type_]: ${msg_}"
        echo "${log_format_}"
    elif [ "$log_type_" == "DEBUG" ]; then
        local log_format_="[Tfplugin] [$cur_date_] [$log_type_]: ${msg_}"
    fi
    echo "${log_format_}" >> $log_file
}

new_echo() {
    local log_type_=${1}
    local log_msg_=${2}
    if  [ "${is_quiet}" = "n" ]; then
        log ${log_type_} ${log_msg_} 1> /dev/null
    fi
}

output_progress()
{
    new_echo "INFO" "install upgradePercentage：$1%"
    log "INFO" "install upgradePercentage：$1%"
}

update_group_right_recursive() {
    local permission="${1}"
    local file_path="${2}"
    if [ "${install_for_all}" = "y" ]; then
        new_permission=${permission:0:-1}${permission: -2:1}
        chmod -R "${new_permission}" "${file_path}"
    else
        chmod -R "${permission}" "${file_path}"
    fi
}

##########################################################################
log "INFO" "step into run_tfplugin_install.sh ......"

log "INFO" "install targetdir $common_parse_dir , type $common_parse_type"

if [ ! -d "${common_parse_dir}" ];then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir is not exist."
    exit 1
fi

install_package() {
    local _package
    local _pythonlocalpath
    _package="$1"
    _pythonlocalpath="$2"
    log "INFO" "install python module package in "${_package}""
    if [ -f "${_package}" ]; then
        if [ "${pylocal}" = "y" ]; then
            if [ ! -d "$_pythonlocalpath" ]; then
                mkdir -p "$_pythonlocalpath"
                chown -R "$username":"$usergroup" "$PYTHONDIR"
                update_group_right_recursive "750" "$PYTHONDIR"
            fi
            pip3.7 install --upgrade --no-deps --force-reinstall "${_package}" -t "${_pythonlocalpath}" 1> /dev/null
        else
            if [ $(id -u) -ne 0 ]; then
                pip3.7 install --upgrade --no-deps --force-reinstall "${_package}" --user 1> /dev/null
            else
                pip3.7 install --upgrade --no-deps --force-reinstall "${_package}" 1> /dev/null
            fi
        fi
        if [ $? -ne 0 ]; then
            log "WARNING" "install ${_package} failed."
            exit 1
        else
            log "INFO" "install ${_package} successful"
        fi
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:install ${_package} faied, can not find the matched package for this platform."
        exit 1
    fi
}

PYTHONDIR="${common_parse_dir}/tfplugin"
WHL_INSTALL_DIR_PATH="${common_parse_dir}/tfplugin/python/site-packages"
SOURCE_PATH="${sourcedir}/bin"
PYTHON_NPU_BRIDGE_WHL="npu_bridge-1.15.0-py3-none-any.whl"
PYTHON_NPU_BRIDGE_NAME="npu_bridge"

new_install() {
    if [ ! -d "${sourcedir}" ]; then
        log "INFO" "no need to install tfplugin files."
        return 0
    fi

    # mkdir
    bash "${curpath}/install_common_parser.sh" --makedir --username="${username}" --usergroup="${usergroup}" $common_parse_type "${common_parse_dir}" "${curpath}/filelist.csv" 1> /dev/null
    if [ $? -ne 0 ];then
        log "ERROR" "ERR_NO:0x0085;ERR_DES:failed to create folder."
        return 1
    fi

    # copy
    if [ "${install_for_all}" = "y" ]; then
        bash "${curpath}/install_common_parser.sh" --copy --username="${username}" --usergroup="${usergroup}" --install_for_all $common_parse_type "${common_parse_dir}" "${curpath}/filelist.csv" 1> /dev/null
    else
        bash "${curpath}/install_common_parser.sh" --copy --username="${username}" --usergroup="${usergroup}" $common_parse_type "${common_parse_dir}" "${curpath}/filelist.csv" 1> /dev/null
    fi
    if [ $? -ne 0 ];then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to copy files."
        return 1
    fi

    # chown
    if [ "${install_for_all}" = "y" ]; then
        bash "${curpath}/install_common_parser.sh" --chmoddir --username="${username}" --usergroup="${usergroup}" --install_for_all $common_parse_type "${common_parse_dir}" "${curpath}/filelist.csv" 1> /dev/null
    else
        bash "${curpath}/install_common_parser.sh" --chmoddir --username="${username}" --usergroup="${usergroup}" $common_parse_type "${common_parse_dir}" "${curpath}/filelist.csv" 1> /dev/null
    fi
    if [ $? -ne 0 ];then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to chown files."
        return 1
    fi
    output_progress 25
    # install npu bridge
    if [[ -z "${PYTHONDIR}" ]]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:runtime directory is empty"
        exit 1
    else
        log "INFO" "install npu bridge begin..."
        install_package "${SOURCE_PATH}/${PYTHON_NPU_BRIDGE_WHL}" "${WHL_INSTALL_DIR_PATH}"
        output_progress 50
        log "INFO" "successful install the npu bridge..."

        if [ "$(arch)" == "x86_64" ];then
          log "INFO" "install npu device begin..."
          install_package "${SOURCE_PATH}/npu_device-0.1-py3-none-any.whl" "${WHL_INSTALL_DIR_PATH}"
          output_progress 60
          log "INFO" "successful install the npu device..."
        fi
		
        output_progress 75
        if [ "${pylocal}" = "y" ]; then
            new_echo "INFO" "please make sure PYTHONPATH is correct !"
        fi
    fi
    return 0
}

new_install
if [ $? -ne 0 ];then
    exit 1
fi

output_progress 100
exit 0
