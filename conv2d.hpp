//conv2d.hpp — 3×3 Conv (BN already folded), ReLU optional.
#pragma once
#include "resunet_hls.hpp"

// Weights layout in memory: W[kh][kw][Cin][Cout], bias[Cout]

template<int H, int W, int Cin, int Cout, int K=3, bool RELU=true>
void conv2d_bn_relu(
    data_t in[H][W][Cin],
    data_t out[H][W][Cout],
    const data_t Wt[K][K][Cin][Cout],
    const data_t B[Cout])
{
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable=Wt complete dim=1
#pragma HLS ARRAY_PARTITION variable=Wt complete dim=2
#pragma HLS RESOURCE variable=Wt core=ROM_1P
#pragma HLS RESOURCE variable=B  core=ROM_1P

    const int pad = K/2;

    Row_Loop:
    for (int i=0;i<H;i++){
        Col_Loop:
        for (int j=0;j<W;j++){
#pragma HLS PIPELINE II=1
            Co_Tile:
            for (int co=0; co<Cout; co++){
#pragma HLS UNROLL factor=1
                acc_t sum = (acc_t)B[co];
                Kx_Loop:
                for (int ki=0; ki<K; ki++){
                    int ii = i + ki - pad;
                    if (ii < 0 || ii >= H) continue;
                    Ky_Loop:
                    for (int kj=0; kj<K; kj++){
                        int jj = j + kj - pad;
                        if (jj < 0 || jj >= W) continue;
                        Ci_Loop:
                        for (int ci=0; ci<Cin; ci++){
#pragma HLS UNROLL factor=1
                            sum += (acc_t)in[ii][jj][ci] * (acc_t)Wt[ki][kj][ci][co];
                        }
                    }
                }
                data_t v = (data_t)sum;
                if (RELU && v < (data_t)0) v = (data_t)0;
                out[i][j][co] = v;
            }
        }
    }
}
