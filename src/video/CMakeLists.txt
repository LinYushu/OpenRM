add_library(
    openrm_video
        SHARED
)

# 查找 gxiapi 动态链接库
find_library(GXIAPI_LIB
    NAMES gxiapi
)

# 检查库是否找到
if(GXIAPI_LIB)
    message(STATUS "Found DaHeng: ${GXIAPI_LIB}")
    set(HAVE_GXIAPI TRUE)
else()
    message(STATUS "Could NOT found DaHeng libgxiapi.so, disable DaHeng camera support.")
    set(HAVE_GXIAPI FALSE)
endif()

# 查找海康相机相关库文件
set(HIK_LIB_PATH "/opt/MVS/lib/64") # TODO 注意 可能需要根据海康SDK安装路径修改
find_library(HIK_MV_LIB       NAMES MvCameraControl   PATHS ${HIK_LIB_PATH})
find_library(HIK_CONVERT_LIB  NAMES FormatConversion  PATHS ${HIK_LIB_PATH})
find_library(HIK_RENDER_LIB   NAMES MVRender          PATHS ${HIK_LIB_PATH})
find_library(HIK_USB_LIB      NAMES MvUsb3vTL         PATHS ${HIK_LIB_PATH})
find_library(HIK_MEDIA_LIB    NAMES MediaProcess      PATHS ${HIK_LIB_PATH}) # 推荐添加

if(HIK_MV_LIB AND HIK_CONVERT_LIB AND HIK_RENDER_LIB AND HIK_USB_LIB AND HIK_MEDIA_LIB) 
    message(STATUS "Found Hikvision Library: ${HIK_MV_LIB}")
    set(HAVE_HIK TRUE)
else()
    message(STATUS "Hikvision Library NOT found in ${HIK_LIB_PATH}, disable Hik camera support")
    set(HAVE_HIK FALSE)
endif()

if(HAVE_HIK)
    set(HIK_LINK_LIBS 
        ${HIK_MV_LIB} 
        ${HIK_CONVERT_LIB} 
        ${HIK_RENDER_LIB} 
        ${HIK_USB_LIB} 
        ${HIK_MEDIA_LIB} 
        pthread 
        dl
    )
else()
    set(HIK_LINK_LIBS "")
endif()

target_sources(
    openrm_video
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/video/uvc.cpp
        ${CMAKE_SOURCE_DIR}/src/video/tools.cpp
        $<IF:$<BOOL:${HAVE_GXIAPI}>,${CMAKE_SOURCE_DIR}/src/video/daheng.cpp,>
        $<IF:$<BOOL:${HAVE_HIK}>,${CMAKE_SOURCE_DIR}/src/video/hik.cpp,>
)

target_include_directories(
    openrm_video
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/openrm>
)

target_link_libraries(
    openrm_video
        PRIVATE
        $<IF:$<BOOL:${HAVE_GXIAPI}>,gxiapi,>
)

if(HAVE_HIK)
    target_link_libraries(openrm_video PRIVATE ${HIK_LINK_LIBS})
    set_target_properties(openrm_video PROPERTIES
        BUILD_RPATH "/opt/MVS/lib/64"
        INSTALL_RPATH "/opt/MVS/lib/64"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()