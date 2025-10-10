//pool_upsample.hpp

#pragma once
#include "resunet_hls.hpp"

template<int H, int W, int C>
void maxpool2x2(
    data_t in[H][W][C],
    data_t out[H/2][W/2][C])
{
#pragma HLS inline off
    for (int i=0;i<H;i+=2){
        for (int j=0;j<W;j+=2){
#pragma HLS PIPELINE II=1
            for (int c=0;c<C;c++){
                data_t a = in[i  ][j  ][c];
                data_t b = in[i  ][j+1][c];
                data_t d = in[i+1][j  ][c];
                data_t e = in[i+1][j+1][c];
                data_t m1 = (a>b)?a:b;
                data_t m2 = (d>e)?d:e;
                out[i/2][j/2][c] = (m1>m2)?m1:m2;
            }
        }
    }
}

template<int H, int W, int C>
void upsample2x_nn(
    data_t in[H][W][C],
    data_t out[H*2][W*2][C])
{
#pragma HLS inline off
    for (int i=0;i<H;i++){
        for (int j=0;j<W;j++){
#pragma HLS PIPELINE II=1
            for (int c=0;c<C;c++){
                data_t v = in[i][j][c];
                out[2*i  ][2*j  ][c] = v;
                out[2*i  ][2*j+1][c] = v;
                out[2*i+1][2*j  ][c] = v;
                out[2*i+1][2*j+1][c] = v;
            }
        }
    }
}
