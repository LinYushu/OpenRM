#ifndef __OPENRM_CUDA_TOOLS_H__
#define __OPENRM_CUDA_TOOLS_H__
#include <cstdint>
#include <cuda_runtime.h>

namespace rm {

void resize(
    uint8_t* src,
    int src_width,
    int src_height,
    float* dst,
    int dst_width,
    int dst_height,
    void* cuda_stream
);

struct alignas(float) YoloDetectionRaw {
    float box[4];      
    float pose[16];    
    float confidence;  
    int class_id;      
    int color_id;      
    bool keep;         
};

void launch_yolo_decode_and_nms(
    const float* d_input, YoloDetectionRaw* d_output, int* d_count,
    int bboxes_num, 
    int tensor_locate_dim, int tensor_color_dim, int tensor_class_dim,
    int infer_width, int infer_height,
    float conf_thresh, float nms_thresh,
    float infer_to_input_ratio, float left_move, float top_move,
    int max_output_num, cudaStream_t stream
);

}

#endif