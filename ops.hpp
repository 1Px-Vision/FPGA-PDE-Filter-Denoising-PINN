//ops.hpp — Add, Concat, GAP, ECA

#pragma once
#include "resunet_hls.hpp"

template<int H, int W, int C>
void add_residual(
    data_t a[H][W][C],
    data_t b[H][W][C],
    data_t out[H][W][C])
{
#pragma HLS inline off
    for (int i=0;i<H;i++)
      for (int j=0;j<W;j++){
#pragma HLS PIPELINE II=1
        for (int c=0;c<C;c++){
          data_t v = a[i][j][c] + b[i][j][c];
          out[i][j][c] = (v < (data_t)0) ? (data_t)0 : v; // ReLU after add
        }
      }
}

template<int H, int W, int C1, int C2>
void concat_channels(
    data_t a[H][W][C1],
    data_t b[H][W][C2],
    data_t out[H][W][C1+C2])
{
#pragma HLS inline off
    for (int i=0;i<H;i++)
      for (int j=0;j<W;j++){
#pragma HLS PIPELINE II=1
        for (int c=0;c<C1;c++) out[i][j][c] = a[i][j][c];
        for (int c=0;c<C2;c++) out[i][j][C1+c] = b[i][j][c];
      }
}

// Global Average Pool over (H,W) per channel
template<int H, int W, int C>
void gap(
    data_t in[H][W][C],
    data_t out[C])
{
#pragma HLS inline off
    for (int c=0;c<C;c++){
        acc_t s = 0;
        for (int i=0;i<H;i++)
          for (int j=0;j<W;j++)
            s += (acc_t)in[i][j][c];
        out[c] = (data_t)(s / (acc_t)(H*W));
    }
}

// ECA gate: 1x1 conv over C (dense vector) + hard-sigmoid; then multiply
template<int C>
void eca_gate(
    data_t feat_in[][C], // not used (for signature symmetry)
    data_t gap_vec[C],
    const data_t w1x1[C][C],
    const data_t b1x1[C],
    data_t gate[C])
{
#pragma HLS inline off
    for (int co=0; co<C; co++){
#pragma HLS PIPELINE II=1
        acc_t s = (acc_t)b1x1[co];
        for (int ci=0; ci<C; ci++){
            s += (acc_t)gap_vec[ci] * (acc_t)w1x1[ci][co];
        }
        gate[co] = hard_sigmoid((data_t)s);
    }
}

template<int H, int W, int C>
void channel_multiply(
    data_t in[H][W][C],
    data_t gate[C],
    data_t out[H][W][C])
{
#pragma HLS inline off
    for (int i=0;i<H;i++)
      for (int j=0;j<W;j++){
#pragma HLS PIPELINE II=1
        for (int c=0;c<C;c++){
          out[i][j][c] = in[i][j][c] * gate[c];
        }
      }
}
