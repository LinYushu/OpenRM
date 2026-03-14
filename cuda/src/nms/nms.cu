// openrm/cuda/src/nms/nms.cu
#include "cudatools.h"
#include <math_constants.h>

__device__ float calculate_iou(const float* box1, const float* box2) {
    float x1 = max(box1[0] - box1[2]/2.0f, box2[0] - box2[2]/2.0f);
    float y1 = max(box1[1] - box1[3]/2.0f, box2[1] - box2[3]/2.0f);
    float x2 = min(box1[0] + box1[2]/2.0f, box2[0] + box2[2]/2.0f);
    float y2 = min(box1[1] + box1[3]/2.0f, box2[1] + box2[3]/2.0f);
    float w = max(0.0f, x2 - x1);
    float h = max(0.0f, y2 - y1);
    float over_area = w * h;
    float union_area = box1[2]*box1[3] + box2[2]*box2[3] - over_area + 1e-5f;
    return over_area / union_area;
}

__global__ void decode_kernel(
    const float* d_input, rm::YoloDetectionRaw* d_output, int* d_count,
    int bboxes_num, int locate_dim, int color_dim, int class_dim, 
    int infer_width, int infer_height,
    float conf_thresh, float ratio, float left_move, float top_move, int max_output_num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bboxes_num) return;

    // 1. 动态精准计算真实 Tensor 步长 (彻底解决内存错位！)
    int yolo_size = locate_dim + 1 + color_dim + class_dim;
    const float* raw = d_input + idx * yolo_size;
    
    // 2. 严格的边缘过滤 (仅对四点模型 FP/FPX 生效，防止角点飞出屏幕)
    if (locate_dim == 8) {
        bool flag_pose = true;
        for(int i = 0; i < 4; i++) {
            if(raw[2 * i] < 1e-3f || raw[2 * i] > (infer_width - 1.001f) ||
               raw[2 * i + 1] < 1e-3f || raw[2 * i + 1] > (infer_height - 1.001f)) {
                flag_pose = false;
                break;
            }
        }
        if(!flag_pose) return;
    }

    // 3. 提取前置 iou_conf
    float iou_conf = raw[locate_dim];
    if (iou_conf < conf_thresh) return;

    // 4. 颜色解析 (兼容无颜色的V5/FP)
    int color_id = 0;
    if (color_dim > 0) {
        float max_color_conf = 0.0f;
        int check_len = (color_dim >= 3) ? 3 : color_dim; // 还原 FPX 中只检查前3位的逻辑
        for (int i = 0; i < check_len; i++) {
            float conf = raw[locate_dim + 1 + i] * iou_conf;
            if (conf > max_color_conf) {
                color_id = i;
                max_color_conf = conf;
            }
        }
        if (max_color_conf < conf_thresh) return;
    }

    // 5. 类别解析
    int class_id = 0;
    float max_class_conf = 0.0f;
    if (class_dim > 0) {
        for (int i = 0; i < class_dim; i++) {
            float conf = raw[locate_dim + 1 + color_dim + i] * iou_conf;
            if (conf > max_class_conf) {
                class_id = i;
                max_class_conf = conf;
            }
        }
    }
    
    // 最终置信度判断：还原原版逻辑，最终得分仅看最高类别得分！
    if (max_class_conf < conf_thresh) return;

    // 6. 坐标解析
    float cx, cy, w, h;
    if (locate_dim == 4) {
        cx = raw[0]; cy = raw[1]; w = raw[2]; h = raw[3];
    } else {
        float min_x = raw[0], max_x = raw[0];
        float min_y = raw[1], max_y = raw[1];
        for (int i = 1; i < 4; i++) {
            min_x = min(min_x, raw[i * 2]);
            max_x = max(max_x, raw[i * 2]);
            min_y = min(min_y, raw[i * 2 + 1]);
            max_y = max(max_y, raw[i * 2 + 1]);
        }
        cx = (min_x + max_x) / 2.0f;
        cy = (min_y + max_y) / 2.0f;
        w = max_x - min_x;
        h = max_y - min_y;
    }

    // 写入有效数据
    int count = atomicAdd(d_count, 1);
    if (count < max_output_num) {
        rm::YoloDetectionRaw& det = d_output[count];
        det.box[0] = cx * ratio - left_move;
        det.box[1] = cy * ratio - top_move;
        det.box[2] = w * ratio;
        det.box[3] = h * ratio;
        
        if (locate_dim == 4) {
            // V5: 生成四个角点 TL, TR, BR, BL 顺序
            float pts[8] = {cx - w/2, cy - h/2, cx + w/2, cy - h/2, cx + w/2, cy + h/2, cx - w/2, cy + h/2};
            for(int i=0; i<8; i++) det.pose[i] = pts[i] * ratio - (i%2==0 ? left_move : top_move);
        } else {
            for(int i=0; i<8; i++) det.pose[i] = raw[i] * ratio - (i%2==0 ? left_move : top_move);
        }

        det.confidence = max_class_conf; 
        det.class_id = class_id;
        det.color_id = color_id;
        det.keep = true;
    }
}

__global__ void nms_kernel(rm::YoloDetectionRaw* d_output, int* d_count, float nms_thresh, int max_output_num) {
    int count = min(*d_count, max_output_num);
    int idx = threadIdx.x;
    if (idx >= count) return;
    
    rm::YoloDetectionRaw cur_det = d_output[idx];
    for (int j = 0; j < count; j++) {
        if (idx == j) continue;
        rm::YoloDetectionRaw other = d_output[j];
        
        // 核心修复：完全去除 class_id 判断，还原原版全局抑制，清除不同类别的重叠幽灵框
        if (other.confidence > cur_det.confidence || (other.confidence == cur_det.confidence && j < idx)) {
            if (calculate_iou(cur_det.box, other.box) > nms_thresh) {
                d_output[idx].keep = false;
                break;
            }
        }
    }
}

namespace rm {
void launch_yolo_decode_and_nms(
    const float* d_input, YoloDetectionRaw* d_output, int* d_count,
    int bboxes_num, int locate_dim, int color_dim, int class_dim, 
    int infer_width, int infer_height,
    float conf_thresh, float nms_thresh, float infer_to_input_ratio, 
    float left_move, float top_move, int max_output_num, cudaStream_t stream
) {
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    int block_size = 256;
    int grid_size = (bboxes_num + block_size - 1) / block_size;
    decode_kernel<<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, d_count, bboxes_num, locate_dim, color_dim, class_dim, 
        infer_width, infer_height,
        conf_thresh, infer_to_input_ratio, left_move, top_move, max_output_num
    );
    nms_kernel<<<1, 1024, 0, stream>>>(d_output, d_count, nms_thresh, max_output_num);
}
}