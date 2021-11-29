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
install_for_all=n
curpath=$(dirname $(readlink -f $0))
common_parse_dir=$COMMON_INSTALL_DIR
common_parse_type=$COMMON_INSTALL_TYPE
unset PYTHONPATH

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
    install_for_all="${5}"
    docker_root="${6}"
fi

if [ x"${docker_root}" != "x" ]; then
    common_parse_dir=${docker_root}${input_install_dir}
else
    common_parse_dir=${input_install_dir}
fi

install_info="${common_parse_dir}/fwkplugin/ascend_install.info"
sourcedir="${common_parse_dir}/fwkplugin"
SOURCE_INSTALL_COMMON_PARSER_FILE="${curpath}/install_common_parser.sh"

if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi

log_file="${log_dir}/ascend_install.log"

if [ -f "${install_info}" ]; then
    username=$(get_install_param "Fwkplugin_UserName" "${install_info}")
    usergroup=$(get_install_param "Fwkplugin_UserGroup" "${install_info}")
fi
if [ "$username" == "" ]; then
    username="$DEFAULT_USERNAME"
    usergroup="$DEFAULT_USERGROUP"
fi

log() {
    local cur_date_=$(date +"%Y-%m-%d %H:%M:%S")
    local log_type_="${1}"
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
log "INFO" "step into run_fwkplugin_uninstall.sh ......"

log "INFO" "uninstall targetdir $common_parse_dir , type $common_parse_type"

if [ ! -d "${common_parse_dir}/fwkplugin" ]; then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir/fwkplugin is not exist."
    exit 1
fi

uninstall_package() {
    local _module="$1"
    local _module_apth="$2"
    if [ ! -d "${WHL_INSTALL_DIR_PATH}/${_module}" ]; then
        pip3 show ${_module} >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "WARNING" "${_module} is not exist."
        else
            pip3 uninstall -y "${_module}" 1> /dev/null
            if [ $? -ne 0 ]; then
                log "WARNING" "uninstall ${_module} failed."
                exit 1
            else
                log "INFO" "uninstall ${_module} successful."
            fi
        fi
    else
        export PYTHONPATH="${_module_apth}"
        pip3 uninstall -y "${_module}" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "WARNING" "uninstall ${_module} failed."
            exit 1
        else
            log "INFO" "uninstall ${_module} successful."
        fi
    fi
}

remove_empty_dir() {
    local _path="${1}"
    if [ -d "${_path}" ]; then
        is_empty=$(ls "${_path}" | wc -l)
        if [ $is_empty -ne 0 ]; then
            log "INFO" "${_path} dir is not empty."
        else
            prev_path=$(dirname "${_path}")
            chmod +w "${prev_path}" >/dev/null 2>&1
            rm -rf "${_path}" >/dev/null 2>&1
        fi
    fi
}

remove_check_shell() {
    local type_arr=("bash" "csh" "fish")
    for type in ${type_arr[@]}; do
        local check_shell_path=${sourcedir}/bin/prereq_check.${type}
        local common_path=${common_parse_dir}/bin/prereq_check.${type}
        local package_name="fwkplugin"
        [ ! -f "${common_path}" ] && continue

        chmod u+w ${common_path}
        local path_regex="\/\(.\+\/\)\?${package_name}\/bin\/prereq_check.${type}"
        sed -i "/^${path_regex}$/d" ${common_path}
        chmod u-w ${common_path}

        num=$(grep -r "prereq_check.${type}" "${common_path}" | wc -l)
        if [ ${num} -eq 0 ]; then
            rm -f ${common_path}
        fi
    done
}

FWKPLUGIN_INSTALL_DIR_PATH="${common_parse_dir}/fwkplugin"
WHL_INSTALL_DIR_PATH="${common_parse_dir}/python/site-packages"

new_uninstall() {
    if [ ! -d "${FWKPLUGIN_INSTALL_DIR_PATH}" ]; then
        log "INFO" "no need to uninstall fwkplugin files."
        return 0
    else
        chmod +w -R "$curpath"
        log "INFO" "uninstall npu bridge begin..."
        uninstall_package "npu_bridge" ${WHL_INSTALL_DIR_PATH}
        log "INFO" "successful uninstall npu bridge."
        if [ "$(arch)" == "x86_64" ];then
            log "INFO" "uninstall npu device begin..."
            uninstall_package "npu_device" ${WHL_INSTALL_DIR_PATH}
            log "INFO" "successful uninstall npu device."
        fi
    fi

    # 赋可写权限
    bash "${SOURCE_INSTALL_COMMON_PARSER_FILE}" --package="fwkplugin" --restoremod "${common_parse_type}" "${common_parse_dir}" "${curpath}/filelist.csv" $feature_type

    BASH_ENV_PATH="${input_install_dir}/fwkplugin/bin/setenv.bash"
    bash "${SOURCE_INSTALL_COMMON_PARSER_FILE}" --del-env-rc --package="fwkplugin" --username="${username}" "${common_parse_dir}" "${BASH_ENV_PATH}" "bash"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to unset bash env."
        return 1
    fi

    CSH_ENV_PATH="${input_install_dir}/fwkplugin/bin/setenv.csh"
    bash "${SOURCE_INSTALL_COMMON_PARSER_FILE}" --del-env-rc --package="fwkplugin" --username="${username}" "${common_parse_dir}" "${CSH_ENV_PATH}" "csh"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to unset csh env."
        return 1
    fi

    FISH_ENV_PATH="${input_install_dir}/fwkplugin/bin/setenv.fish"
    bash "${SOURCE_INSTALL_COMMON_PARSER_FILE}" --del-env-rc --package="fwkplugin" --username="${username}" "${common_parse_dir}" "${FISH_ENV_PATH}" "fish"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to unset fish env."
        return 1
    fi

    remove_check_shell
    # 删除目录与文件
    bash "${SOURCE_INSTALL_COMMON_PARSER_FILE}" --package="fwkplugin" --remove "${common_parse_type}" "${common_parse_dir}" "${curpath}/filelist.csv"
    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0090;ERR_DES:remove tfplguin files failed."
        return 1
    fi

    if [ -d "${common_parse_dir}/python/site-packages" ]; then
        if [ -f "${common_parse_dir}/python/site-packages/LICENSE" ]; then
            rm -rf "${common_parse_dir}/python/site-packages/LICENSE"
        fi
    fi

    remove_empty_dir "${common_parse_dir}/python/site-packages"
    remove_empty_dir "${common_parse_dir}/python"

    if [ -d "${FWKPLUGIN_INSTALL_DIR_PATH}/python/site-packages" ]; then
        rm -rf "${FWKPLUGIN_INSTALL_DIR_PATH}/python/site-packages"
    fi

    remove_empty_dir "${FWKPLUGIN_INSTALL_DIR_PATH}/python"

    if [ $? -ne 0 ]; then
        log "ERROR" "ERR_NO:0x0090;ERR_DES:delete fwkplugin files failed."
        return 1
    fi
    return 0
}

new_uninstall
if [ $? -ne 0 ]; then
    exit 1
fi

exit 0
