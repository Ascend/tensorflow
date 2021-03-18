#!/bin/bash
COMMON_INSTALL_DIR=/usr/local/Ascend
COMMON_INSTALL_TYPE=full
DEFAULT_USERNAME=HwHiAiUser
DEFAULT_USERGROUP=HwHiAiUser
is_quiet=n
pylocal=n
install_for_all=n
curpath="$(dirname "$(readlink -f "$0")")"
sourcedir="$PWD/tfplugin"
common_parse_dir=$COMMON_INSTALL_DIR
common_parse_type=$COMMON_INSTALL_TYPE

getInstallParam() {
    local _key="$1"
    local _file="$2"
    local _param

    if [ ! -f "${_file}" ];then
        exit 1
    fi
    install_info_key_array=("Tfplugin_Install_Type" "Tfplugin_UserName" "Tfplugin_UserGroup" "Tfplugin_Install_Path_Param")
    for key_param in "${install_info_key_array[@]}"; do
        if [ ${key_param} == ${_key} ]; then
            _param=`grep -r "${_key}=" "${_file}" | cut -d"=" -f2-`
            break
        fi
    done
    echo "${_param}"
}

if [ $1 ];then
    common_parse_dir="$2"
    common_parse_type=$3
    is_quiet=$4
    pylocal=$5
    install_for_all=$6
fi

installInfo="${common_parse_dir}/tfplugin/ascend_install.info"
installInfo_old="/etc/ascend_install.info"

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

output_progress() {
    newEcho "INFO" "upgrade upgradePercentage:$1%"
    log "INFO" "upgrade upgradePercentage:$1%"
}

if [ -f "$installInfo" ]; then
    Tfplugin_UserName=$(getInstallParam "Tfplugin_UserName" "${installInfo}")
    Tfplugin_UserGroup=$(getInstallParam "Tfplugin_UserGroup" "${installInfo}")
    Tfplugin_Install_Type=$(getInstallParam "Tfplugin_Install_Type" "${installInfo}")
    username="$Tfplugin_UserName"
    usergroup="$Tfplugin_UserGroup"
elif [ -f $installInfo_old ] && [ ` grep -c -i "Acllib_Install_Path_Param" $installInfo_old ` -ne 0]; then
    . $installInfo_old
    username=$UserName
    usergroup=$UserGroup
else
    echo "ERR_NO:0x0080;ERR_DES:Installation information no longer exists,please complete "${installInfo}" or ${installInfo_old}"
    exit 1
fi

if [ "$username" == "" ]; then
    username="$DEFAULT_USERNAME"
    usergroup="$DEFAULT_USERGROUP"
fi

##########################################################################
log "INFO" "step into run_tfplugin_upgrade.sh ......"

log "INFO" "upgrade target dir $common_parse_dir , type $common_parse_type"

if [ ! -d "$common_parse_dir" ];then
    log "ERROR" "ERR_NO:0x0001;ERR_DES:path $common_parse_dir is not exist."
    exit 1
fi

InstallPackage() {
    local _package
    local _pythonlocalpath
    _package="$1"
    _pythonlocalpath="$2"
    log "INFO" "install python module package in ${_package}"
    if [ -f "${_package}" ]; then
        if [ "${pylocal}" = "y" ]; then
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

PYTHONDIR="${common_parse_dir}""/tfplugin"
WHL_INSTALL_DIR_PATH="${PYTHONDIR}/python/site-packages"
SOURCE_PATH="${sourcedir}/bin"
PYTHON_NPU_BRIDGE_WHL="npu_bridge-1.15.0-py3-none-any.whl"
PYTHON_NPU_BRIDGE_NAME="npu_bridge"

newUpgrade() {
    if [ ! -d ${sourcedir} ]; then
        log "INFO" "no need to upgrade tfplugin files."
        return 0
    fi

    # mkdir
    bash "${curpath}/install_common_parser.sh" --makedir --username="$username" --usergroup="$usergroup" $common_parse_type "$common_parse_dir" "${curpath}/filelist.csv" 1> /dev/null
    if [ $? -ne 0 ];then
        log "ERROR" "ERR_NO:0x0085;ERR_DES:failed to create folder."
        return 1
    fi
    # copy
    if [ "${install_for_all}" = "y" ]; then
        bash "${curpath}/install_common_parser.sh" --copy --username="$username" --usergroup="$usergroup" --install_for_all $common_parse_type "$common_parse_dir" "${curpath}/filelist.csv" 1> /dev/null
    else
        bash "${curpath}/install_common_parser.sh" --copy --username="$username" --usergroup="$usergroup" $common_parse_type "$common_parse_dir" "${curpath}/filelist.csv" 1> /dev/null
    fi
    if [ $? -ne 0 ];then
        log "ERROR" "ERR_NO:0x0089;ERR_DES:failed to copy files."
        return 1
    fi
    # chown
    if [ "${install_for_all}" = "y" ]; then
        bash "${curpath}/install_common_parser.sh" --chmoddir --username="$username" --usergroup="$usergroup" --install_for_all $common_parse_type "$common_parse_dir" "${curpath}/filelist.csv" 1> /dev/null
    else
        bash "${curpath}/install_common_parser.sh" --chmoddir --username="$username" --usergroup="$usergroup" $common_parse_type "$common_parse_dir" "${curpath}/filelist.csv" 1> /dev/null
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
        InstallPackage "${SOURCE_PATH}/${PYTHON_NPU_BRIDGE_WHL}" "${WHL_INSTALL_DIR_PATH}"
        output_progress 50
        log "INFO" "successful install the npu bridge..."

        if [ "$(arch)" == "x86_64" ];then
          log "INFO" "install npu device begin..."
          InstallPackage "${SOURCE_PATH}/npu_device-0.1-py3-none-any.whl" "${WHL_INSTALL_DIR_PATH}"
          output_progress 60
          log "INFO" "successful install the npu device..."
        fi

        output_progress 75
        if [ "${pylocal}" = "y" ]; then
            newEcho "INFO" "please make sure PYTHONPATH is correct !"
        fi
    fi
    return 0
}

newUpgrade
if [ $? -ne 0 ];then
    exit 1
fi

output_progress 100
exit 0
