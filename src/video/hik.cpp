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
#include <opencv2/imgproc.hpp> // 【优化】确保引入OpenCV图像处理模块

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

    // 检查数据是否为 BayerRG8 格式 (0x01080009)
    if (pFrameInfo->enPixelType == PixelType_Gvsp_BayerRG8) {
        // 利用原始指针 pData 构建单通道 Mat（零拷贝，无内存开销）
        cv::Mat raw_bayer(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        // 使用 OpenCV 高效 NEON 指令集转为 BGR 彩色图
        cv::cvtColor(raw_bayer, *(frame->image), cv::COLOR_BayerRG2RGB);
    } 
    else {
        // 容错回退机制：万一未应用Bayer，退回海康原厂慢速转换
        MV_CC_PIXEL_CONVERT_PARAM stConvertParam = {0};
        stConvertParam.nWidth = pFrameInfo->nWidth;
        stConvertParam.nHeight = pFrameInfo->nHeight;
        stConvertParam.pSrcData = pData;
        stConvertParam.nSrcDataLen = pFrameInfo->nFrameLen;
        stConvertParam.enSrcPixelType = pFrameInfo->enPixelType;
        stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
        stConvertParam.pDstBuffer = frame->image->data;
        stConvertParam.nDstBufferSize = pFrameInfo->nWidth * pFrameInfo->nHeight * 3;

        void *handle = hik_cam_map[camera];
        int nRet = MV_CC_ConvertPixelType(handle, &stConvertParam);
        if (MV_OK != nRet) {
            rm::message("Video Hik callback convert pixel failed", rm::MSG_ERROR);
            return;
        }
    }

    // // 处理翻转 (如果需要)
    // if (callback_param->flip) {
    //     cv::flip(*(frame->image), *(frame->image), -1);
    // }

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

    MV_CC_SetEnumValueByString(handle, "ExposureAuto", "Off");
    MV_CC_SetEnumValueByString(handle, "GainAuto", "Off");

    nRet = MV_CC_SetBoolValue(handle, "AcquisitionFrameRateEnable", false);
    if (MV_OK != nRet) rm::message("Video Hik disable FrameRate limitation failed", rm::MSG_WARNING);

    MVCC_INTVALUE stParam;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    nRet = MV_CC_GetIntValue(handle, "DeviceLinkThroughputLimit", &stParam);
    if (MV_OK == nRet) {
        // 将带宽限制设置为相机支持的最大值
        MV_CC_SetIntValue(handle, "DeviceLinkThroughputLimit", stParam.nMax);
    }
    nRet = MV_CC_SetFloatValue(handle, "ExposureTime", static_cast<float>(exposure));
    if (MV_OK != nRet) rm::message("Video Hik set ExposureTime failed", rm::MSG_WARNING);

    // 设置增益 (单位: dB)
    nRet = MV_CC_SetFloatValue(handle, "Gain", static_cast<float>(gain));
    if (MV_OK != nRet) rm::message("Video Hik set Gain failed", rm::MSG_WARNING);

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

    // 将翻转任务交给相机 ISP，彻底解放 CPU
    if (flip) {
        MV_CC_SetBoolValue(handle, "ReverseX", true);
        MV_CC_SetBoolValue(handle, "ReverseY", true);
    } else {
        MV_CC_SetBoolValue(handle, "ReverseX", false);
        MV_CC_SetBoolValue(handle, "ReverseY", false);
    }

    nRet = MV_CC_SetImageNodeNum(handle, 10);
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