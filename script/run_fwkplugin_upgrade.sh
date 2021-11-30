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

COMMON_INSTALL_DIR=/usr/local/Ascend
COMMON_INSTALL_TYPE=full
DEFAULT_USERNAME=$(id -un)
DEFAULT_USERGROUP=$(id -gn)
is_quiet=n
pylocal=n
install_for_all=n
setenv_flag=n
docker_root=""
sourcedir="$PWD/fwkplugin"
curpath=$(dirname $(readlink -f $0))
common="${curpath}"/common_func.inc
common_parse_dir=$COMMON_INSTALL_DIR
common_parse_type=$COMMON_INSTALL_TYPE
unset PYTHONPATH

. "${common}"

get_install_param() {
    local _key="$1"
    local _file="$2"
    local _param

    if [ ! -f "${_file}" ]; then
        exit 1
    fi
    install_info_key_array=("Fwkplugin_Install_Type" "Fwkplugin_UserName" "Fwkplugin_UserGroup" "Fwkplugin_Install_Path_Param")
    for key_param in "${install_info_key_array[@]}"; do
        if [ "${key_param}" == "${_key}" ]; then
            _param=$(grep -r "${_key}=" "${_file}" | cut -d"=" -f2-)
            break
        fi
    done
    echo "${_param}"
}

if [ "$1" ]; then
    input_install_dir="${2}"
    common_parse_type="${3}"
    is_quiet="${4}"
    pylocal="${5}"
    setenv_flag="${6}"
    docker_root="${7}"
    in_install_for_all="${8}"
fi

if [ x"${docker_root}" != "x" ]; then
    common_parse_dir=${docker_root}${input_install_dir}
else
    common_parse_dir=${input_install_dir}
fi

install_info="${common_parse_dir}/fwkplugin/ascend_install.info"
install_info_old="/etc/ascend_install.info"

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi

log_file="${log_dir}/ascend_install.log"

log() {
    local cur_date_=$(date +"%Y-%m-%d %H:%M:%S")
    local log_type_=${1}
    local msg_="${2}"
    if [ "$log_type_" == "INFO" ]; then
        local log_format_="[Fwkplugin] [$cur_date_] [$log_type_]: ${msg_}"
        echo "${log_format_}"
    elif [ "$log_type_" == "WARNING" ]; then
        local log_format_="[Fwkplugin] [$cur_date_] [$log_type_]: ${msg_}"
         echo "${log_format_}"
    elif [ "$log_type_" == "ERROR" ]; then
        local log_format_="[Fwkplugin] [$cur_date_] [$log_type_]: ${msg_}"
         echo "${log_format_}"
    elif [ "$log_type_" == "DEBUG" ]; then
        local log_format_="[Fwkplugin] [$cur_date_] [$log_type_]: ${msg_}"
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

output_progress() {
    new_echo "INFO" "upgrade upgradePercentage:$1%"
    log "INFO" "upgrade upgradePercentage:$1%"
}

if [ -f "$install_info" ]; then
    fwkplugin_username=$(get_install_param "Fwkplugin_UserName" "${install_info}")
    fwkplugin_usergroup=$(get_install_param "Fwkplugin_UserGroup" "${install_info}")
    fwkplugin_install_type=$(get_install_param "Fwkplugin_Install_Type" "${install_info}")
    username="$fwkplugin_username"
    usergroup="$fwkplugin_usergroup"
elif [ -f "$install_info_old" ] && [ $(grep -c -i "Fwkplugin_Install_Path_Param" "$install_info_old") -ne 0]; then
    . $install_info_old
    username=$DEFAULT_USERNAME
    usergroup=$DEFAULT_USERGROUP
else
    echo "ERR_NO:0x0080;ERR_DES:Installation information no longer exists,please complete "${install_info}" or ${install_info_old}"
    exit 1
fi

if [ "$username" == "" ]; then
    username="$DEFAULT_USERNAME"
    usergroup="$DEFAULT_USERGROUP"
fi

##########################################################################
log "INFO" "step into run_fwkplugin_upgrade.sh ......"

log "INFO" "upgrade target dir $common_parse_dir , type $common_parse_type"

if [ ! -d "$common_parse_dir" ]; then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir is not exist."
    exit 1
fi

install_package() {
    local _package="$1"
    local _pythonlocalpath="$2"
    log "INFO" "install python module package in ${_package}"
    if [ -f "${_package}" ]; then
        if [ "${pylocal}" = "y" ]; then
            pip3 install --upgrade --no-deps --force-reinstall "${_package}" -t "${_pythonlocalpath}" 1> /dev/null
        else
            if [ $(id -u) -ne 0 ]; then
                pip3 install --upgrade --no-deps --force-reinstall "${_package}" --user 1> /dev/null
            else
                pip3 install --upgrade --no-deps --force-reinstall "${_package}" 1> /dev/null
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

copy_check_shell() {
    local type_arr=("bash" "csh" "fish")
    for type in ${type_arr[@]}; do
        local check_shell_path=${input_install_dir}/fwkplugin/bin/prereq_check.${type}
        local common_path=${common_parse_dir}/bin/prereq_check.${type}
        local path_regex="\/\(.\+\/\)\?fwkplugin\/bin\/prereq_check.${type}"
        if [ -f "${common_path}" ]; then
            chmod u+w ${common_path}
            sed -i "/^${path_regex}$/d" ${common_path}
            echo "${check_shell_path}" >> ${common_path}
            chmod u-w ${common_path}
        else
            echo "#!/usr/bin/env ${type}" > ${common_path}
            echo "${check_shell_path}" >> ${common_path}
            chmod 550 ${common_path}
        fi
        chown ${username}:${usergroup} ${common_path}
    done
}

set_env() {
    local setenv_option=""
    if [ "${setenv_flag}" = y ]; then
        setenv_option="--setenv"
    fi

    BASH_ENV_PATH="${input_install_dir}/fwkplugin/bin/setenv.bash"
    bash "${curpath}/install_common_parser.sh" --add-env-rc --package="fwkplugin" --username="${username}" --usergroup="${usergroup}" \
        $setenv_option --docker-root="${docker_root}" "${common_parse_dir}" "${BASH_ENV_PATH}" "bash"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to set bash env."
        return 1
    fi

    CSH_ENV_PATH="${input_install_dir}/fwkplugin/bin/setenv.csh"
    bash "${curpath}/install_common_parser.sh" --add-env-rc --package="fwkplugin" --username="${username}" --usergroup="${usergroup}" \
        $setenv_option --docker-root="${docker_root}" "${common_parse_dir}" "${CSH_ENV_PATH}" "csh"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to set csh env."
        return 1
    fi

    FISH_ENV_PATH="${input_install_dir}/fwkplugin/bin/setenv.fish"
    bash "${curpath}/install_common_parser.sh" --add-env-rc --package="fwkplugin" --username="${username}" --usergroup="${usergroup}" \
        $setenv_option --docker-root="${docker_root}" "${common_parse_dir}" "${FISH_ENV_PATH}" "fish"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to set fish env."
        return 1
    fi
}

PYTHONDIR="${common_parse_dir}"
WHL_INSTALL_DIR_PATH="${common_parse_dir}/python/site-packages"
WHL_SOFTLINK_INSTALL_DIR_PATH="${common_parse_dir}/fwkplugin/python/site-packages"
SOURCE_PATH="${sourcedir}/bin"
PYTHON_NPU_BRIDGE_WHL="npu_bridge-1.15.0-py3-none-any.whl"
PYTHON_NPU_DEVICE_WHL="npu_device-0.1-py3-none-any.whl"

new_upgrade() {
    if [ ! -d "${sourcedir}" ]; then
        log "INFO" "no need to upgrade fwkplugin files."
        return 0
    fi

    # 创建目录
    bash "${curpath}/install_common_parser.sh" --package="fwkplugin" --makedir --username="${username}" --usergroup="${usergroup}" ${in_install_for_all} "${common_parse_type}" "${common_parse_dir}" "${curpath}/filelist.csv"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0085;ERR_DES:failed to create folder."
        return 1
    fi

    # 拷贝目录与文件
    bash "${curpath}/install_common_parser.sh" --package="fwkplugin" --copy --username="${username}" --usergroup="${usergroup}" --set-cann-uninstall ${in_install_for_all} "${common_parse_type}" "${common_parse_dir}" "${curpath}/filelist.csv"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to copy files."
        return 1
    fi

    # 文件与目录赋权
    copy_check_shell

    log "INFO" "Fwkplugin do setenv."
    set_env
    if [ $? -ne 0 ]; then
        return 1
    fi

    bash "${curpath}/install_common_parser.sh" --package="fwkplugin" --chmoddir --username="${username}" --usergroup="${usergroup}" ${in_install_for_all} "${common_parse_type}" "${common_parse_dir}" "${curpath}/filelist.csv"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to chown files."
        return 1
    fi

    output_progress 25
    if [[ -z "${PYTHONDIR}" ]]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:runtime directory is empty"
        exit 1
    else
        log "INFO" "install npu bridge begin..."
        install_package "${SOURCE_PATH}/${PYTHON_NPU_BRIDGE_WHL}" "${WHL_INSTALL_DIR_PATH}"
        output_progress 50
        log "INFO" "successful install the npu bridge..."
        if [ "$(arch)" == "x86_64" ];then
            log "INFO" "install adapter for tensorflow 2.x begin..."
            install_package "${SOURCE_PATH}/${PYTHON_NPU_DEVICE_WHL}" "${WHL_INSTALL_DIR_PATH}"
            output_progress 60
            log "INFO" "successfully installed adapter for tensorflow 2.x"
        fi

        if [ "${pylocal}" = "y" ]; then
            mkdir -p "$WHL_SOFTLINK_INSTALL_DIR_PATH"
            create_softlink "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "npu_bridge"
            create_softlink "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "npu_bridge-*.dist-info"
            if [ "$(arch)" == "x86_64" ];then
                create_softlink "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "npu_device"
                create_softlink "${WHL_INSTALL_DIR_PATH}" "$WHL_SOFTLINK_INSTALL_DIR_PATH" "npu_device-*.dist-info"
            fi
        fi

        output_progress 75
    fi

    return 0
}

new_upgrade
if [ $? -ne 0 ]; then
    exit 1
fi

output_progress 100
exit 0
