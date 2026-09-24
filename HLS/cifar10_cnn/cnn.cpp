#include "cnn.h"
#include "model_params.h"

// Compile-time checks for the exact SmallCifarCNN_v1 graph.
static_assert(sizeof(INPUT_LUT) / sizeof(INPUT_LUT[0]) == 256, "Invalid input LUT");
static_assert(sizeof(CONV1_WEIGHT) / sizeof(CONV1_WEIGHT[0]) == 16*3*3*3, "Invalid conv1");
static_assert(sizeof(CONV2_WEIGHT) / sizeof(CONV2_WEIGHT[0]) == 32*16*3*3, "Invalid conv2");
static_assert(sizeof(CONV3_WEIGHT) / sizeof(CONV3_WEIGHT[0]) == 64*32*3*3, "Invalid conv3");
static_assert(sizeof(FC_WEIGHT) / sizeof(FC_WEIGHT[0]) == 10*64, "Invalid FC");
static_assert(sizeof(CONV1_BIAS) / sizeof(CONV1_BIAS[0]) == 16, "Invalid conv1 bias");
static_assert(sizeof(CONV2_BIAS) / sizeof(CONV2_BIAS[0]) == 32, "Invalid conv2 bias");
static_assert(sizeof(CONV3_BIAS) / sizeof(CONV3_BIAS[0]) == 64, "Invalid conv3 bias");
static_assert(sizeof(FC_BIAS) / sizeof(FC_BIAS[0]) == 10, "Invalid FC bias");

// Convolution + ReLU/requantization + 2x2 max-pooling.
// Compute the four convolution pixels belonging to each pooled output.
// This avoids storing a full unpooled feature map. All buffers use CHW.
// Weight order: OIHW, flattened in C order, matching model_params.h.
template<int CI, int CO, int H, int SHIFT>
static void conv_relu_pool(const int8_t input[CI*H*H],
                           int8_t output[CO*(H/2)*(H/2)],
                           const int8_t weights[CO*CI*9],
                           const int32_t biases[CO]) {
#pragma HLS INLINE off
    static_assert(SHIFT >= -31 && SHIFT <= 62, "Unsupported requantization shift");
    const int OH = H / 2;
    for (int oc = 0; oc < CO; ++oc) {
#pragma HLS PIPELINE off
        for (int oy = 0; oy < OH; ++oy) {
#pragma HLS PIPELINE off
            for (int ox = 0; ox < OH; ++ox) {
#pragma HLS PIPELINE off
                int8_t maximum = 0;
                for (int p = 0; p < 4; ++p) {
#pragma HLS PIPELINE off
                    const int cy = 2*oy + p/2;
                    const int cx = 2*ox + p%2;
                    int32_t accumulator = biases[oc];
                    for (int k = 0; k < CI*9; ++k) {
#pragma HLS PIPELINE II=1
                        const int ic = k/9;
                        const int ky = (k%9)/3;
                        const int kx = k%3;
                        const int iy = cy + ky - 1;
                        const int ix = cx + kx - 1;
                        int8_t pixel = 0;
                        if (iy >= 0 && iy < H && ix >= 0 && ix < H)
                            pixel = input[(ic*H + iy)*H + ix];
                        // INT8 operands, exact signed 16-bit product.
                        int16_t product = (int16_t)pixel * (int16_t)weights[oc*CI*9+k];
                        accumulator += product;
                    }
                    // Integer rounding/saturation comes from the quantizer header.
                    const int8_t value = cnn_requant_relu(accumulator, SHIFT);
                    if (value > maximum) maximum = value;
                }
                output[(oc*OH+oy)*OH+ox] = maximum;
            }
        }
    }
}

void cifar10_cnn(const uint8_t image[3072], int32_t logits[10]) {
#pragma HLS INTERFACE mode=m_axi port=image offset=slave bundle=gmem_in depth=3072
#pragma HLS INTERFACE mode=m_axi port=logits offset=slave bundle=gmem_out depth=10
#pragma HLS INTERFACE mode=s_axilite port=image bundle=control
#pragma HLS INTERFACE mode=s_axilite port=logits bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Each buffer is completely overwritten on every call (no reset contents needed).
    static int8_t input_chw[3*32*32];
    static int8_t stage1[16*16*16];
    static int8_t stage2[32*8*8];
    static int8_t stage3[64*4*4];
    int8_t gap[64];
    int32_t result[10];
#pragma HLS BIND_STORAGE variable=input_chw type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=stage1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=stage2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=stage3 type=ram_2p impl=bram

    // Read sequential RGB bytes from DDR, normalize/quantize by LUT, store CHW.
    for (int index = 0; index < 3072; ++index) {
#pragma HLS PIPELINE II=1
        const int pixel = index/3;
        const int channel = index%3;
        input_chw[channel*1024+pixel] = INPUT_LUT[image[index]];
    }

    conv_relu_pool<3,16,32,CONV1_SHIFT>(input_chw, stage1, CONV1_WEIGHT, CONV1_BIAS);
    conv_relu_pool<16,32,16,CONV2_SHIFT>(stage1, stage2, CONV2_WEIGHT, CONV2_BIAS);
    conv_relu_pool<32,64,8,CONV3_SHIFT>(stage2, stage3, CONV3_WEIGHT, CONV3_BIAS);

    for (int channel = 0; channel < 64; ++channel) {
#pragma HLS PIPELINE off
        int32_t sum = 0;
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE II=1
            sum += stage3[channel*16+i];
        }
        // Exact GAP rule from quantize_cifar10_hls.py; all inputs are nonnegative.
        gap[channel] = (int8_t)((sum+8) >> 4);
    }
    for (int oc = 0; oc < 10; ++oc) {
#pragma HLS PIPELINE off
        int32_t accumulator = FC_BIAS[oc];
        for (int ic = 0; ic < 64; ++ic) {
#pragma HLS PIPELINE II=1
            int16_t product = (int16_t)gap[ic] * (int16_t)FC_WEIGHT[oc*64+ic];
            accumulator += product;
        }
        result[oc] = accumulator;
    }
    for (int i = 0; i < 10; ++i) {
#pragma HLS PIPELINE II=1
        logits[i] = result[i];
    }
}
