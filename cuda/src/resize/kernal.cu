#include "resize.cuh"

__global__ void warpaffine_kernel(
    uint8_t* src,
    int src_line_size,
    int src_width,
    int src_height,
    float* dst,
    int dst_width,
    int dst_height,
    uint8_t const_value_st,
    AffineMatrix d2s,
    int edge
) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;

    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;

    float c0, c1, c2; // c0=Blue, c1=Green, c2=Red (OpenCV默认序)

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // 越界填充
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } else {
        // 双线性插值
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        uint8_t const_val[3] = {const_value_st, const_value_st, const_value_st};
        uint8_t* v1 = const_val;
        uint8_t* v2 = const_val;
        uint8_t* v3 = const_val;
        uint8_t* v4 = const_val;
        
        if (y_low >= 0) {
            if (x_low >= 0) v1 = src + y_low * src_line_size + x_low * 3;
            if (x_high < src_width) v2 = src + y_low * src_line_size + x_high * 3;
        }
        if (y_high < src_height) {
            if (x_low >= 0) v3 = src + y_high * src_line_size + x_low * 3;
            if (x_high < src_width) v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    int plane_size = dst_width * dst_height;
    
    // rgbrgbrgb to rrrgggbbb
    int out_idx_r = position;                // R 通道位置
    int out_idx_g = position + plane_size;   // G 通道位置
    int out_idx_b = position + plane_size*2; // B 通道位置

    dst[out_idx_r] = c2 / 255.0f; // 写 R
    dst[out_idx_g] = c1 / 255.0f; // 写 G
    dst[out_idx_b] = c0 / 255.0f; // 写 B
}