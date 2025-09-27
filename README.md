# HLS-Optimized Parallel PDE Filter (FPGA)

A lightweight, line-rate PDE residual filter for images, designed for Vitis HLS and easy integration with DPU pipelines. It implements the 3×3 Laplacian stencil.

$$
\nabla^{2} u \=\
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}
*\ u
$$ 

* **Diffusion residual** ``` residual = α·(∇²u) − pred ```, where pred is typically the DPU model output.

* **Memory-mapped (AXI4-M) 2D arrays** — simplest to integrate if you’re using PL kernels on PS DDR buffers.

* **AXI4-Stream + line buffer** — fully pipelined streaming kernel to chain after your DPU or a DMA.

## Memory-mapped version

```#pragma HLS INLINE off ```

* Prevents Vivado/Vitis HLS from inlining diffusion_residual_mm into its caller.

* Function as a distinct RTL block , avoids the caller’s pragmas interfering with this kernel’s pipeline.

``` #pragma HLS ARRAY_PARTITION variable=... dim=2 cyclic factor=4 ```

* Splits the 2-D arrays across dimension 2 (the column index) into 4 interleaved banks (cyclic partitioning). Element [r][c] goes to bank c % 4.

* The inner loop accesses up to five columns per pixel (c_left, c, c_right on the same row and c on up/down rows) plus one write to out[r][c]. With a single BRAM, those would serialize and break II=1.

* Banking lets the tool schedule multiple parallel reads/writes in the same cycle by placing adjacent columns in different BRAM banks, eliminating port conflicts.

``` #pragma HLS PIPELINE II=1 ```

* Inner column loop so a new pixel is processed every clock

* Simple arithmetic (adds/mults) and lowering clock

```
// diffusion_residual_mm.h
#include <ap_int.h>
#include <ap_fixed.h>

template<int H, int W>
void diffusion_residual_mm(
    const ap_fixed<16,6> u   [H][W],   // input image u(x,y)
    const ap_fixed<16,6> pred[H][W],   // model(u)(x,y) from DPU
          ap_fixed<16,6> out [H][W],   // residual output
    const ap_fixed<16,6> alpha         // diffusion coefficient
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=u   dim=2 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=pred dim=2 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=out  dim=2 cyclic factor=4

Row_Loop:
    for (int r = 0; r < H; ++r) {
    Col_Loop:
        for (int c = 0; c < W; ++c) {
#pragma HLS PIPELINE II=1

            // Replicate padding (clamped borders)
            int r_up    = (r == 0   ) ? 0      : r-1;
            int r_down  = (r == H-1 ) ? H-1    : r+1;
            int c_left  = (c == 0   ) ? 0      : c-1;
            int c_right = (c == W-1 ) ? W-1    : c+1;

            ap_fixed<16,6> u_c  = u[r][c];
            ap_fixed<16,6> u_u  = u[r_up   ][c];
            ap_fixed<16,6> u_d  = u[r_down ][c];
            ap_fixed<16,6> u_l  = u[r      ][c_left ];
            ap_fixed<16,6> u_r  = u[r      ][c_right];

            // 3x3 Laplacian (0,1,0; 1,-4,1; 0,1,0)
            ap_fixed<18,8> lap = (ap_fixed<18,8>)u_u
                               + (ap_fixed<18,8>)u_d
                               + (ap_fixed<18,8>)u_l
                               + (ap_fixed<18,8>)u_r
                               - (ap_fixed<18,8>)(4.0) * (ap_fixed<18,8>)u_c;

            ap_fixed<18,8> diff = (ap_fixed<18,8>)alpha * lap;

            out[r][c] = (ap_fixed<16,6>)( diff - (ap_fixed<18,8>)pred[r][c] );
        }
    }
}

```

## AXI4-Stream version

```
// diffusion_residual_axis.h
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

template<int D, int U=1, int TI=1, int TO=1>
struct axis_t {
    ap_uint<D> data;
    ap_uint<U> user;
    ap_uint<1> last;
    ap_uint<TI> id;
    ap_uint<TO> dest;
};

static ap_uint<16> pack_fx16(ap_fixed<16,6> x) { return ap_uint<16>(x.range(15,0)); }
static ap_fixed<16,6> unpack_fx16(ap_uint<16> w){ ap_fixed<16,6> x; x.range(15,0)=w; return x; }

template<int W>
void diffusion_residual_axis(
    hls::stream< axis_t<16> >& u_in,      // pixel stream of u
    hls::stream< axis_t<16> >& pred_in,   // pixel stream of model(u)
    hls::stream< axis_t<16> >& out_stream,// residual stream
    int H, int W_in,
    ap_fixed<16,6> alpha
) {
#pragma HLS INTERFACE axis port=u_in
#pragma HLS INTERFACE axis port=pred_in
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=H       bundle=control
#pragma HLS INTERFACE s_axilite port=W_in    bundle=control
#pragma HLS INTERFACE s_axilite port=alpha   bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    // 3-line buffer for vertical neighborhood
    static ap_fixed<16,6> linebuf0[W];
    static ap_fixed<16,6> linebuf1[W];
    static ap_fixed<16,6> linebuf2[W];
#pragma HLS BIND_STORAGE variable=linebuf0 type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=linebuf1 type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=linebuf2 type=ram_1p impl=bram

    // column shift registers for horizontal neighbors
    ap_fixed<16,6> u_up_l=0,   u_up_c=0,   u_up_r=0;
    ap_fixed<16,6> u_mid_l=0,  u_mid_c=0,  u_mid_r=0;
    ap_fixed<16,6> u_dn_l=0,   u_dn_c=0,   u_dn_r=0;

Row:
    for (int r=0; r<H; ++r) {
    Col:
        for (int c=0; c<W_in; ++c) {
#pragma HLS PIPELINE II=1

            // Read inputs
            axis_t<16> iu  = u_in.read();
            axis_t<16> ipr = pred_in.read();
            ap_fixed<16,6> u_px   = unpack_fx16(iu.data);
            ap_fixed<16,6> pred_px= unpack_fx16(ipr.data);

            // Shift line buffers: roll (0<-1, 1<-2)
            ap_fixed<16,6> up   = linebuf0[c];
            ap_fixed<16,6> mid  = linebuf1[c];
            ap_fixed<16,6> down = linebuf2[c];

            linebuf0[c] = mid;
            linebuf1[c] = down;
            linebuf2[c] = u_px;

            // Shift horizontal regs: (l <- c, c <- r, r <- new_col)
            u_up_l  = u_up_c;   u_up_c  = u_up_r;   u_up_r  = up;
            u_mid_l = u_mid_c;  u_mid_c = u_mid_r;  u_mid_r = mid;
            u_dn_l  = u_dn_c;   u_dn_c  = u_dn_r;   u_dn_r  = down;

            bool valid = (r >= 2) && (c >= 2);

            ap_fixed<18,8> lap = 0;
            if (valid) {
                // 3x3 cross (up,down,left,right) around center u_mid_c
                lap = (ap_fixed<18,8>)u_up_c
                    + (ap_fixed<18,8>)u_dn_c
                    + (ap_fixed<18,8>)u_mid_l
                    + (ap_fixed<18,8>)u_mid_r
                    - (ap_fixed<18,8>)(4.0) * (ap_fixed<18,8>)u_mid_c;
            } else {
                lap = 0; // zero Laplacian for borders (simple)
            }

            ap_fixed<18,8> diff = (ap_fixed<18,8>)alpha * lap;
            ap_fixed<16,6> res  = (ap_fixed<16,6>)( diff - (ap_fixed<18,8>)pred_px );

            axis_t<16> o;
            o.data = pack_fx16(res);
            o.user = iu.user;
            o.id   = iu.id;
            o.dest = iu.dest;
            // propagate TLAST at the end of each frame
            o.last = ( (c == W_in-1) && (r == H-1) ) ? ap_uint<1>(1) : ap_uint<1>(0);
            out_stream.write(o);
        }
    }
}

```
