//blocks.hpp — Residual block (two 3×3 convs + optional ECA)

#pragma once
#include "resunet_hls.hpp"
#include "conv2d.hpp"
#include "ops.hpp"

template<int H, int W, int Cin, int Cout, bool USE_ECA=false>
void residual_block(
    data_t in[H][W][Cin],
    data_t out[H][W][Cout],
    // conv1 params
    const data_t W1[3][3][Cin][Cout],
    const data_t B1[Cout],
    // conv2 params
    const data_t W2[3][3][Cout][Cout],
    const data_t B2[Cout],
    // optional 1x1 for skip if Cin!=Cout
    const data_t Wsc[1][1][Cin][Cout],
    const data_t Bsc[Cout],
    // ECA params if used
    const data_t WECA[Cout][Cout],
    const data_t BECA[Cout]
)
{
#pragma HLS inline off
    static data_t c1[H][W][Cout];
    static data_t c2[H][W][Cout];
#pragma HLS ARRAY_PARTITION variable=c1 complete dim=3
#pragma HLS ARRAY_PARTITION variable=c2 complete dim=3

    // Conv1 + ReLU
    conv2d_bn_relu<H,W,Cin,Cout,3,true>(in, c1, W1, B1);
    // Conv2, no ReLU
    conv2d_bn_relu<H,W,Cout,Cout,3,false>(c1, c2, W2, B2);

    if (USE_ECA){
        data_t g[Cout];
        data_t vgap[Cout];
        gap<H,W,Cout>(c2, vgap);
        eca_gate<Cout>(nullptr, vgap, WECA, BECA, g);
        channel_multiply<H,W,Cout>(c2, g, c2); // in-place to c2
    }

    static data_t skip[H][W][Cout];
#pragma HLS ARRAY_PARTITION variable=skip complete dim=3

    if (Cin == Cout){
        // copy input to skip (type cast)
        for (int i=0;i<H;i++)
          for (int j=0;j<W;j++){
#pragma HLS PIPELINE II=1
            for (int c=0;c<Cout;c++){
                skip[i][j][c] = in[i][j][c];
            }
          }
    } else {
        // 1x1 projection (implemented via conv2d with K=1)
        // Reuse conv kernel with K=3 by providing only center tap is nonzero is cumbersome;
        // For brevity, do a small 1x1 nested loop:
        for (int i=0;i<H;i++)
          for (int j=0;j<W;j++){
#pragma HLS PIPELINE II=1
            for (int co=0; co<Cout; co++){
                acc_t s = (acc_t)Bsc[co];
                for (int ci=0; ci<Cin; ci++){
                    s += (acc_t)in[i][j][ci] * (acc_t)Wsc[0][0][ci][co];
                }
                skip[i][j][co] = (data_t)s;
            }
          }
    }

    add_residual<H,W,Cout>(c2, skip, out); // + ReLU
}
