include(FetchContent)
set(_json_url "")
if(TF_PKG_SERVER)
  set(_json_url "${TF_PKG_SERVER}/libs/securec/v1.1.10.tar.gz")
  FetchContent_Declare(
          secure_c
          URL ${_json_url}
  )
else()
  FetchContent_Declare(
          secure_c
          URL https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.10.tar.gz
          URL_HASH MD5=f3db321939ae17527b8939651f7e1c8b
  )
endif()
FetchContent_GetProperties(secure_c)
if (NOT secure_c_POPULATED)
    FetchContent_Populate(secure_c)
    include_directories(${secure_c_SOURCE_DIR}/include)
endif ()
