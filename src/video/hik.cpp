#include <map>
#include <string>
#include <iostream>
#include "uniterm/uniterm.h"
#include "utils/timer.h"
#include "structure/stamp.hpp"
#include "structure/camera.hpp"
#include "video/video.h"
#include "video/hik/MvCameraControl.h"
#include "video/hik/PixelType.h"

using namespace rm;
using namespace std;

// 存储相机句柄的Map: device_index -> handle
static std::map<int, void*> hik_cam_map;

// 回调函数参数结构体
struct HikCallbackParam {
    Camera* camera;
    float* yaw;
    float* pitch;
    float* roll;
    bool flip = false;
};

// 海康相机取流回调函数
void __stdcall OnHikFrameCallback(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser) {
    TimePoint time_stamp = getTime();
    HikCallbackParam* callback_param = reinterpret_cast<HikCallbackParam*>(pUser);
    
    float yaw = 0, pitch = 0, roll = 0;
    if (callback_param->yaw != nullptr && callback_param->pitch != nullptr && callback_param->roll != nullptr) {
        yaw = *(callback_param->yaw);
        pitch = *(callback_param->pitch);
        roll = *(callback_param->roll);
    }

    Camera *camera = callback_param->camera;
    
    // 创建Frame对象
    shared_ptr<Frame> frame = make_shared<Frame>();
    frame->image = make_shared<cv::Mat>(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3);
    frame->time_point = time_stamp;
    frame->camera_id = camera->camera_id;
    frame->width = pFrameInfo->nWidth;
    frame->height = pFrameInfo->nHeight;
    frame->yaw = yaw;
    frame->pitch = pitch;
    frame->roll = roll;

    // 像素格式转换 (海康原始数据 -> BGR)
    // 即使是彩色相机，通常也是Bayer格式传输，需要转为BGR给OpenCV使用
    MV_CC_PIXEL_CONVERT_PARAM stConvertParam = {0};
    stConvertParam.nWidth = pFrameInfo->nWidth;
    stConvertParam.nHeight = pFrameInfo->nHeight;
    stConvertParam.pSrcData = pData;
    stConvertParam.nSrcDataLen = pFrameInfo->nFrameLen;
    stConvertParam.enSrcPixelType = pFrameInfo->enPixelType;
    stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed; // 转为OpenCV默认的BGR
    stConvertParam.pDstBuffer = frame->image->data;
    stConvertParam.nDstBufferSize = pFrameInfo->nWidth * pFrameInfo->nHeight * 3;

    void* handle = hik_cam_map[camera->camera_id];
    int nRet = MV_CC_ConvertPixelType(handle, &stConvertParam);
    if (MV_OK != nRet) {
        rm::message("Video Hik callback convert pixel failed", rm::MSG_ERROR);
        return;
    }

    // 处理翻转 (如果需要)
    if (callback_param->flip) {
        cv::flip(*(frame->image), *(frame->image), -1);
    }

    camera->buffer->push(frame);
}

// 获取海康相机数量
bool rm::getHikCameraNum(int& num) {
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    // 枚举设备 (支持USB和GigE)
    int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet) {
        rm::message("Video Hik enum devices failed", rm::MSG_ERROR);
        return false;
    }
    num = static_cast<int>(stDeviceList.nDeviceNum);
    return true;
}

// 设置相机参数
bool rm::setHikArgs(Camera *camera, double exposure, double gain, double fps) {
    void* handle = hik_cam_map[camera->camera_id];
    int nRet = MV_OK;

    // 设置自动曝光/增益为Off，以便手动控制
    // 注意：不同型号相机节点名称可能略有差异，通常遵循GenICam标准
    MV_CC_SetEnumValueByString(handle, "ExposureAuto", "Off");
    MV_CC_SetEnumValueByString(handle, "GainAuto", "Off");
    MV_CC_SetEnumValueByString(handle, "AcquisitionFrameRateEnable", "true"); // 启用帧率控制

    // 设置曝光时间 (单位: us)
    nRet = MV_CC_SetFloatValue(handle, "ExposureTime", static_cast<float>(exposure));
    if (MV_OK != nRet) rm::message("Video Hik set ExposureTime failed", rm::MSG_WARNING);

    // 设置增益 (单位: dB)
    nRet = MV_CC_SetFloatValue(handle, "Gain", static_cast<float>(gain));
    if (MV_OK != nRet) rm::message("Video Hik set Gain failed", rm::MSG_WARNING);

    // 设置帧率
    nRet = MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", static_cast<float>(fps));
    if (MV_OK != nRet) rm::message("Video Hik set FPS failed", rm::MSG_WARNING);

    return true;
}

// 打开海康相机
bool rm::openHik(
    Camera *camera,
    int device_num,
    float *yaw_ptr,
    float *pitch_ptr,
    float *roll_ptr,
    bool flip,
    double exposure,
    double gain,
    double fps
) {
    if(camera == nullptr) {
        rm::message("Video Hik error at nullptr camera", rm::MSG_ERROR);
        return false;
    }
    if (camera->buffer != nullptr) {
        delete camera->buffer;
    }
    camera->buffer = new SwapBuffer<Frame>();
    camera->camera_id = device_num;

    // 枚举设备以获取句柄创建所需的DeviceInfo
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    
    if (MV_OK != nRet || stDeviceList.nDeviceNum <= device_num) {
        rm::message("Video Hik device not found", rm::MSG_ERROR);
        return false;
    }

    // 创建句柄
    void* handle = nullptr;
    nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[device_num]);
    if (MV_OK != nRet) {
        rm::message("Video Hik create handle failed", rm::MSG_ERROR);
        return false;
    }

    // 打开设备
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet) {
        MV_CC_DestroyHandle(handle);
        rm::message("Video Hik open device failed", rm::MSG_ERROR);
        return false;
    }

    hik_cam_map[device_num] = handle;

    // 如果是GigE相机，建议设置包大小
    if (stDeviceList.pDeviceInfo[device_num]->nTLayerType == MV_GIGE_DEVICE) {
        int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
        if (nPacketSize > 0) {
            nRet = MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
            if(nRet != MV_OK) rm::message("Video Hik set packet size failed", rm::MSG_WARNING);
        }
    }

    // 获取图像宽高
    MVCC_INTVALUE stIntVal;
    MV_CC_GetIntValue(handle, "Width", &stIntVal);
    camera->width = stIntVal.nCurValue;
    MV_CC_GetIntValue(handle, "Height", &stIntVal);
    camera->height = stIntVal.nCurValue;

    // 设置参数
    rm::setHikArgs(camera, exposure, gain, fps);

    // 准备回调参数
    // 注意：这里new出来的内存需要管理，简单起见在closeHik时不delete，
    // 若需严格内存管理，可存在camera结构体中或使用shared_ptr
    HikCallbackParam* callback_param = new HikCallbackParam; 
    callback_param->camera = camera;
    callback_param->yaw = yaw_ptr;
    callback_param->pitch = pitch_ptr;
    callback_param->roll = roll_ptr;
    callback_param->flip = flip;

    // 注册回调函数
    nRet = MV_CC_RegisterImageCallBackEx(handle, OnHikFrameCallback, callback_param);
    if (MV_OK != nRet) {
        rm::message("Video Hik register callback failed", rm::MSG_ERROR);
        return false;
    }

    // 开始取流
    nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet) {
        rm::message("Video Hik start grabbing failed", rm::MSG_ERROR);
        return false;
    }

    rm::message("Video Hik opened", rm::MSG_OK);
    return true;
}

// 关闭海康相机
bool rm::closeHik() {
    for(auto it = hik_cam_map.begin(); it != hik_cam_map.end(); it++) {
        void* handle = it->second;
        if (handle) {
            MV_CC_StopGrabbing(handle);
            MV_CC_CloseDevice(handle);
            MV_CC_DestroyHandle(handle);
        }
    }
    hik_cam_map.clear();
    rm::message("Video Hik closed", rm::MSG_WARNING);
    return true;
}