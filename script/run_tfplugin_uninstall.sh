#!/bin/bash
COMMON_INSTALL_DIR=/usr/local/Ascend
COMMON_INSTALL_TYPE=full
DEFAULT_USERNAME=HwHiAiUser
DEFAULT_USERGROUP=HwHiAiUser
is_quiet=n
install_for_all=n
curpath="$(dirname "$(readlink -f "$0")")"
common_parse_dir=$COMMON_INSTALL_DIR
common_parse_type=$COMMON_INSTALL_TYPE

if [ $1 ];then
    common_parse_dir=$2
    common_parse_type=$3
    is_quiet=$4
    install_for_all=$5
fi

installInfo="${common_parse_dir}/tfplugin/ascend_install.info"
sourcedir="${common_parse_dir}/tfplugin"
if [ $(id -u) -ne 0 ]; then
    log_dir="${HOME}/var/log/ascend_seclog"
else
    log_dir="/var/log/ascend_seclog"
fi

logFile="${log_dir}/ascend_install.log"

log() {
    local cur_date_=`date +"%Y-%m-%d %H:%M:%S"`
    local log_type_=${1}
    local msg_="${2}"
    if [ $log_type_ == "INFO" ]; then
        local log_format_="[Tfplugin] [$cur_date_] [$log_type_]: ${msg_}"
        echo "${log_format_}"
    elif [ $log_type_ == "WARNING" ]; then
        local log_format_="[Tfplugin] [$cur_date_] [$log_type_]: ${msg_}"
        echo "${log_format_}"
    elif [ $log_type_ == "ERROR" ]; then
        local log_format_="[Tfplugin] [$cur_date_] [$log_type_]: ${msg_}"
        echo "${log_format_}"
    elif [ $log_type_ == "DEBUG" ]; then
        local log_format_="[Tfplugin] [$cur_date_] [$log_type_]: ${msg_}"
    fi
    echo "${log_format_}" >> $logFile
}

newEcho() {
    local log_type_=${1}
    local log_msg_=${2}
    if  [ "${is_quiet}" = "n" ]; then
        log ${log_type_} ${log_msg_} 1> /dev/null
    fi
}

updateGroupRightRecursive() {
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
log "INFO" "step into run_tfplugin_uninstall.sh ......"

log "INFO" "uninstall targetdir $common_parse_dir , type $common_parse_type"

if [ ! -d "${common_parse_dir}/tfplugin" ];then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir/tfplugin is not exist."
    exit 1
fi

UninstallPackage() {
    local _module
    _module="$1"
    ls -A "${WHL_INSTALL_DIR_PATH}/${_module}" >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        pip3.7 show "${_module}" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "WARNING" "ERR_NO:0x0080;ERR_DES:${_module} is not installed before."
        else
            pip3.7 uninstall -y "${_module}" >/dev/null 2>&1
            if [ $? -ne 0 ]; then
                log "WARNING" "uninstall ${_module} failed."
                exit 1
            else
                log "INFO" "uninstall ${_module} successful."
            fi
        fi
    else
        chmod +w -R "${WHL_INSTALL_DIR_PATH}" >/dev/null 2>&1
        rm -rf "${WHL_INSTALL_DIR_PATH}/${_module}" >/dev/null 2>&1
        rm -rf "${WHL_INSTALL_DIR_PATH}/${_module}-1.15.0.dist-info" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            log "WARNING" "uninstall ${_module} failed."
            exit 1
        else
            log "INFO" "uninstall ${_module} successful."
        fi
    fi
}

PYTHONDIR="${common_parse_dir}""/tfplugin"
WHL_INSTALL_DIR_PATH="${PYTHONDIR}/python/site-packages"
NPU_BRIDGE_NAME="npu_bridge"

newUninstall() {
    if [ ! -d "${sourcedir}" ]; then
        log "INFO" "no need to uninstall tfplugin files."
        return 0
    fi

    if [[ -z "${PYTHONDIR}" ]]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:tfplugin directory is empty"
        exit 1
    else
        chmod +w -R "$curpath"
        log "INFO" "uninstall npu bridge begin..."
        UninstallPackage "${NPU_BRIDGE_NAME}"
        log "INFO" "successful uninstall npu bridge."

        if [ "$(arch)" == "x86_64" ];then
          log "INFO" "uninstall npu device begin..."
          UninstallPackage "npu_device"
          log "INFO" "successful uninstall the npu device..."
        fi

        if [ -d "${WHL_INSTALL_DIR_PATH}" ] && [ "`ls -A "${WHL_INSTALL_DIR_PATH}"`" = "" ]; then
            rm -rf "${PYTHONDIR}/python"
        fi
    fi

    # chown
    bash "${curpath}/install_common_parser.sh" --restoremod --username="${username}" --usergroup="${usergroup}" $common_parse_type "${common_parse_dir}" "${curpath}/filelist.csv" $feature_type 1> /dev/null

    # remove
    bash "${curpath}/install_common_parser.sh" --remove $common_parse_type "${common_parse_dir}" "${curpath}/filelist.csv" 1> /dev/null
    if [ $? -ne 0 ];then
        log "ERROR" "ERR_NO:0x0090;ERR_DES:delete tfplugin files failed."
        return 1
    fi
    return 0
}

newUninstall
if [ $? -ne 0 ];then
    exit 1
fi

exit 0
