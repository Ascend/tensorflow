include(FetchContent)
set(_json_url "")
if(TF_PKG_SERVER)
  set(_json_url "${TF_PKG_SERVER}/libs/json/v3.6.1/include.zip")
  FetchContent_Declare(
          nlohmann_json
          URL https://ascend-cann.obs.myhuaweicloud.com/json/repository/archive/json-v3.10.1.zip
  )
else()
  FetchContent_Declare(
          nlohmann_json
          URL https://ascend-cann.obs.myhuaweicloud.com/json/repository/archive/json-v3.10.1.zip
  )
endif()
FetchContent_GetProperties(nlohmann_json)
if (NOT nlohmann_json_POPULATED)
    FetchContent_Populate(nlohmann_json)
    include_directories(${nlohmann_json_SOURCE_DIR}/include)
endif ()