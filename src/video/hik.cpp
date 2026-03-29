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
#include <opencv2/imgproc.hpp>

using namespace rm;
using namespace std;

// 存储相机句柄的Map: device_index -> handle
static std::map<rm::Camera *, void *> hik_cam_map;

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
    shared_ptr<Frame> frame = make_shared<Frame>();
    
    // 创建Frame对象
    frame->image = make_shared<cv::Mat>(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3);

    frame->time_point = time_stamp;
    frame->camera_id = camera->camera_id;
    frame->width = pFrameInfo->nWidth;
    frame->height = pFrameInfo->nHeight;
    frame->yaw = yaw;
    frame->pitch = pitch;
    frame->roll = roll;

    if (pFrameInfo->enPixelType == PixelType_Gvsp_BayerRG8) {
        cv::Mat raw_bayer(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        cv::cvtColor(raw_bayer, *(frame->image), cv::COLOR_BayerRG2RGB);
    } else {
        rm::message("Unexpected PixelFormat in Callback", rm::MSG_WARNING);
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
    void *handle = hik_cam_map[camera];
    int nRet = MV_OK;

    MV_CC_SetEnumValueByString(handle, "TriggerMode", "Off");
    MV_CC_SetEnumValueByString(handle, "AcquisitionMode", "Continuous");

    nRet = MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_BayerRG8); 
    if (MV_OK != nRet) {
        rm::message("Video Hik set PixelFormat to BayerRG8 failed, fallback to default", rm::MSG_WARNING);
    }

    nRet = MV_CC_SetEnumValueByString(handle, "ExposureAuto", "Off");
    nRet = MV_CC_SetEnumValueByString(handle, "GainAuto", "Off");
    nRet = MV_CC_SetEnumValueByString(handle, "BalanceWhiteAuto", "Off");

    // 设置曝光时间与增益
    nRet = MV_CC_SetFloatValue(handle, "ExposureTime", static_cast<float>(exposure));
    if (MV_OK != nRet) rm::message("Video Hik set ExposureTime failed", rm::MSG_WARNING);
    nRet = MV_CC_SetFloatValue(handle, "Gain", static_cast<float>(gain));
    if (MV_OK != nRet) rm::message("Video Hik set Gain failed", rm::MSG_WARNING);
    nRet = MV_CC_SetBoolValue(handle, "AcquisitionFrameRateEnable", true);
    if (MV_OK != nRet) rm::message("Video Hik enable frame rate failed", rm::MSG_WARNING);
    nRet = MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", static_cast<float>(fps));
    if (MV_OK != nRet) rm::message("Video Hik set fps failed", rm::MSG_WARNING);

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

    // 枚举设备
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

    hik_cam_map[camera] = handle;

    // 如果是GigE相机，建议设置包大小
    if (stDeviceList.pDeviceInfo[device_num]->nTLayerType == MV_GIGE_DEVICE) {
        int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
        if (nPacketSize > 0) {
            nRet = MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
            if(nRet != MV_OK) rm::message("Video Hik set packet size failed", rm::MSG_WARNING);
        }
    }

    // 设置参数 (包括曝光、Bayer配置等)
    rm::setHikArgs(camera, exposure, gain, fps);

    nRet = MV_CC_SetImageNodeNum(handle, 40);
    if (MV_OK != nRet) {
        rm::message("Video Hik set image node num failed", rm::MSG_WARNING);
    }

    // 获取图像宽高 (在设置完参数后获取，保证准确性)
    MVCC_INTVALUE stIntVal;
    MV_CC_GetIntValue(handle, "Width", &stIntVal);
    camera->width = stIntVal.nCurValue;
    MV_CC_GetIntValue(handle, "Height", &stIntVal);
    camera->height = stIntVal.nCurValue;

    // 准备回调参数
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