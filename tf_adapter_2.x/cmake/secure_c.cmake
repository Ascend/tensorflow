if (DEFINED ASCEND_INSTALLED_PATH)
    include(FetchContent)
    if (TF_PKG_SERVER)
        FetchContent_Declare(
                secure_c
                URL ${TF_PKG_SERVER}/libs/securec/v1.1.10.tar.gz
        )
    else ()
        FetchContent_Declare(
                secure_c
                URL https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.10.tar.gz
        )
    endif ()
    FetchContent_GetProperties(secure_c)
    if (NOT secure_c_POPULATED)
        FetchContent_Populate(secure_c)
        include_directories(${secure_c_SOURCE_DIR}/include)
    endif ()
else ()
    include_directories(${ASCEND_CI_BUILD_DIR}/libc_sec/include/)
    include_directories(${ASCEND_CI_BUILD_DIR}/abl/libc_sec/include/)
endif ()