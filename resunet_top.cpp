//ResUNet graph (Encoder-Decoder)
//resunet_top.cpp

#include "resunet_hls.hpp"
#include "conv2d.hpp"
#include "pool_upsample.hpp"
#include "ops.hpp"
#include "blocks.hpp"

extern "C" {
void resunet_accel(
    // AXI-Lite control registers would go here (omitted for brevity)
    volatile data_t *in_ptr,    // (256*256*1)
    volatile data_t *out_ptr,   // (256*256*1)
    // weight pointers (ROMs) – in practice bind to BRAM/ROM; here as arguments
    const data_t *weights_blob  // flat array with all layers; or pass per-layer pointers
)
{
#pragma HLS INTERFACE m_axi     port=in_ptr  offset=slave bundle=gmem0 depth=65536
#pragma HLS INTERFACE m_axi     port=out_ptr offset=slave bundle=gmem1 depth=65536
#pragma HLS INTERFACE m_axi     port=weights_blob offset=slave bundle=gmem2 depth=1
#pragma HLS INTERFACE s_axilite port=in_ptr   bundle=control
#pragma HLS INTERFACE s_axilite port=out_ptr  bundle=control
#pragma HLS INTERFACE s_axilite port=weights_blob bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int H = 256, W = 256;

    // Local feature maps (reduce BRAM)
    static data_t e1[256][256][ 64];
    static data_t p1[128][128][ 64];
    static data_t e2[128][128][128];
    static data_t p2[ 64][ 64][128];
    static data_t e3[ 64][ 64][256];
    static data_t p3[ 32][ 32][256];
    static data_t e4[ 32][ 32][512];

    static data_t d3u[ 64][ 64][512];   // up from e4, then conv→256
    static data_t d3c[ 64][ 64][512];   // concat(d3u, e3)
    static data_t d3[ 64][ 64][256];

    static data_t d2u[128][128][256];
    static data_t d2c[128][128][256+128];
    static data_t d2[128][128][128];

    static data_t d1u[256][256][128];
    static data_t d1c[256][256][128+64];
    static data_t d1[256][256][64];

    static data_t in [256][256][1];
    static data_t out[256][256][1];

#pragma HLS ARRAY_PARTITION variable=e1 complete dim=3
#pragma HLS ARRAY_PARTITION variable=e2 complete dim=3
#pragma HLS ARRAY_PARTITION variable=e3 complete dim=3
#pragma HLS ARRAY_PARTITION variable=e4 complete dim=3
#pragma HLS ARRAY_PARTITION variable=d3 complete dim=3
#pragma HLS ARRAY_PARTITION variable=d2 complete dim=3
#pragma HLS ARRAY_PARTITION variable=d1 complete dim=3

    // Load input
    for (int i=0;i<H;i++)
      for (int j=0;j<W;j++){
#pragma HLS PIPELINE II=1
        in[i][j][0] = in_ptr[i*W + j];
      }

    // === Weights binding ===
    // In a real project, you'd map each W and B pointer to sections of weights_blob or to separate BRAMs.
    // Here we declare them as extern const arrays generated from .npy → .hpp conversion.

    // e1 residual (Cin=1 → 64), with ECA
    extern const data_t W_e1_1[3][3][1][64],  B_e1_1[64];
    extern const data_t W_e1_2[3][3][64][64], B_e1_2[64];
    extern const data_t W_e1_sc[1][1][1][64], B_e1_sc[64];
    extern const data_t W_e1_eca[64][64],     B_e1_eca[64];

    residual_block<256,256,1,64,true>(in,e1, W_e1_1,B_e1_1, W_e1_2,B_e1_2, W_e1_sc,B_e1_sc, W_e1_eca,B_e1_eca);
    maxpool2x2<256,256,64>(e1, p1);

    // e2 residual (64 → 128)
    extern const data_t W_e2_1[3][3][64][128],  B_e2_1[128];
    extern const data_t W_e2_2[3][3][128][128], B_e2_2[128];
    extern const data_t W_e2_sc[1][1][64][128], B_e2_sc[128];
    extern const data_t W_e2_eca[128][128],     B_e2_eca[128];

    residual_block<128,128,64,128,true>(p1,e2, W_e2_1,B_e2_1, W_e2_2,B_e2_2, W_e2_sc,B_e2_sc, W_e2_eca,B_e2_eca);
    maxpool2x2<128,128,128>(e2, p2);

    // e3 residual (128 → 256)
    extern const data_t W_e3_1[3][3][128][256],  B_e3_1[256];
    extern const data_t W_e3_2[3][3][256][256],  B_e3_2[256];
    extern const data_t W_e3_sc[1][1][128][256], B_e3_sc[256];
    extern const data_t W_e3_eca[256][256],      B_e3_eca[256];

    residual_block<64,64,128,256,true>(p2,e3, W_e3_1,B_e3_1, W_e3_2,B_e3_2, W_e3_sc,B_e3_sc, W_e3_eca,B_e3_eca);
    maxpool2x2<64,64,256>(e3, p3);

    // e4 residual (256 → 512), no pooling
    extern const data_t W_e4_1[3][3][256][512],  B_e4_1[512];
    extern const data_t W_e4_2[3][3][512][512],  B_e4_2[512];
    extern const data_t W_e4_sc[1][1][256][512], B_e4_sc[512];
    extern const data_t W_e4_eca[512][512],      B_e4_eca[512];

    residual_block<32,32,256,512,true>(p3,e4, W_e4_1,B_e4_1, W_e4_2,B_e4_2, W_e4_sc,B_e4_sc, W_e4_eca,B_e4_eca);

    // Decoder d3: upsample e4 → 64×64×512, 3×3 conv to 256, concat with e3, residual to 256
    upsample2x_nn<32,32,512>(e4, d3u);   // 64×64×512

    // 3×3 conv to 256 channels (simulate your conv_bn_act before concat)
    extern const data_t W_d3u[3][3][512][256], B_d3u[256];
    conv2d_bn_relu<64,64,512,256,3,true>(d3u, d3u, W_d3u, B_d3u);

    concat_channels<64,64,256,256>(d3u, e3, d3c);

    extern const data_t W_d3_1[3][3][512][256],  B_d3_1[256];
    extern const data_t W_d3_2[3][3][256][256],  B_d3_2[256];
    extern const data_t W_d3_sc[1][1][512][256], B_d3_sc[256];
    extern const data_t W_d3_eca[256][256],      B_d3_eca[256];

    residual_block<64,64,512,256,true>(d3c,d3, W_d3_1,B_d3_1, W_d3_2,B_d3_2, W_d3_sc,B_d3_sc, W_d3_eca,B_d3_eca);

    // d2
    upsample2x_nn<64,64,256>(d3, d2u);
    extern const data_t W_d2u[3][3][256][128], B_d2u[128];
    conv2d_bn_relu<128,128,256,128,3,true>(d2u, d2u, W_d2u, B_d2u);
    concat_channels<128,128,128,128>(d2u, e2, d2c);

    extern const data_t W_d2_1[3][3][256][128],  B_d2_1[128];
    extern const data_t W_d2_2[3][3][128][128],  B_d2_2[128];
    extern const data_t W_d2_sc[1][1][256][128], B_d2_sc[128];
    extern const data_t W_d2_eca[128][128],      B_d2_eca[128];

    residual_block<128,128,256,128,true>(d2c,d2, W_d2_1,B_d2_1, W_d2_2,B_d2_2, W_d2_sc,B_d2_sc, W_d2_eca,B_d2_eca);

    // d1
    upsample2x_nn<128,128,128>(d2, d1u);
    extern const data_t W_d1u[3][3][128][64], B_d1u[64];
    conv2d_bn_relu<256,256,128,64,3,true>(d1u, d1u, W_d1u, B_d1u);
    concat_channels<256,256,64,64>(d1u, e1, d1c);

    extern const data_t W_d1_1[3][3][128][64],  B_d1_1[64];
    extern const data_t W_d1_2[3][3][64][64],   B_d1_2[64];
    extern const data_t W_d1_sc[1][1][128][64], B_d1_sc[64];
    extern const data_t W_d1_eca[64][64],       B_d1_eca[64];

    residual_block<256,256,128,64,true>(d1c,d1, W_d1_1,B_d1_1, W_d1_2,B_d1_2, W_d1_sc,B_d1_sc, W_d1_eca,B_d1_eca);

    // Final 1×1 conv → 1 channel (no activation)
    extern const data_t W_out[1][1][64][1], B_out[1];
    // Implement 1×1 conv
    for (int i=0;i<256;i++)
      for (int j=0;j<256;j++){
#pragma HLS PIPELINE II=1
        acc_t s = (acc_t)B_out[0];
        for (int c=0;c<64;c++) s += (acc_t)d1[i][j][c] * (acc_t)W_out[0][0][c][0];
        out[i][j][0] = (data_t)s;
      }

    // Store output
    for (int i=0;i<H;i++)
      for (int j=0;j<W;j++){
#pragma HLS PIPELINE II=1
        out_ptr[i*W + j] = out[i][j][0];
      }
}
} // extern "C"
