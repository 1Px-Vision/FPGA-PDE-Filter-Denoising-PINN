![](https://github.com/1Px-Vision/FPGA-PDE-Filter-Denoising-PINN/blob/main/PDE_denoise_1.jpg)

# Denoise PDE + Deep Learning (Physics-Informed Denoising)

This approach fuses classical partial differential equations (PDEs) for image smoothing with a CNN/UNet denoiser. The network predicts a clean image while a physics-informed loss penalizes violations of a diffusion model.

$$
 residual=\alpha \cdot \nabla^{2}u-\hat{u}
$$

## Why it helps

* **Robust at low SNR:** PDE prior stabilizes training and suppresses noise structures the CNN might hallucinate.

* **Better generalization:** Encodes domain physics, reducing overfitting and label dependence.

* **Edge-aware:** With proper diffusion (e.g., anisotropic/Perona–Malik), preserves boundaries while removing noise.

* **Hardware-friendly:** Split compute—DPU runs the CNN; an HLS kernel streams the 3×3 Laplacian/residual at II=1, enabling real-time, low-latency denoising on FPGA.

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

CNN runs on the DPU, and the PDE residual runs in a custom HLS kernel (HDL) that reads/writes DDR

* **TF code (no classes)** — build/export the DPU model.
	* Build/link hints (Vitis/Vitis-AI)

* **HLS kernel (MM)** for the PDE residual (AXI4-M + AXI-Lite)
	* **Diffusion residual** ``` residual = α·(∇²u) − pred ```, where pred is typically the DPU model output.

* **Memory-mapped (AXI4-M) 2D arrays** — simplest to integrate if you’re using PL kernels on PS DDR buffers.

* **AXI4-Stream + line buffer** — fully pipelined streaming kernel to chain after your DPU or a DMA.
	* Minimal XRT host to run DPU → PDE

## Memory-mapped version

* ```#pragma HLS INLINE off ```

    * Prevents Vivado/Vitis HLS from inlining diffusion_residual_mm into its caller.

    * Function as a distinct RTL block , avoids the caller’s pragmas interfering with this kernel’s pipeline.

* ``` #pragma HLS ARRAY_PARTITION variable=... dim=2 cyclic factor=4 ```

    * Splits the 2-D arrays across dimension 2 (the column index) into 4 interleaved banks (cyclic partitioning). Element [r][c] goes to bank c % 4.

    * The inner loop accesses up to five columns per pixel (c_left, c, c_right on the same row and c on up/down rows) plus one write to out[r][c]. With a single BRAM, those would serialize and break II=1.

    * Banking lets the tool schedule multiple parallel reads/writes in the same cycle by placing adjacent columns in different BRAM banks, eliminating port conflicts.

* ``` #pragma HLS PIPELINE II=1 ```

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
            ap_fixed<16,6> u_l  = u[r][c_left ];
            ap_fixed<16,6> u_r  = u[r][c_right];

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

* ```#pragma HLS INTERFACE axis port=u_in/pred_in/out_stream ```
	* Declares AXI4-Stream on the three pixel streams so the kernel can run at line-rate with back-pressure. The tool inserts FIFOs and TVALID/TREADY logic.

* ``` #pragma HLS INTERFACE s_axilite port=H/W_in/alpha/return bundle=control ```
	* Creates a lightweight AXI4-Lite control port for scalar args and the return (start/idle regs). Lets PS set image size and α at run time.

* ``` #pragma HLS BIND_STORAGE variable=linebuf* type=ram_1p impl=bram ```
	* Forces each line buffer into single-port BRAM. This is a resource binding directive: use BRAMs, not LUTRAM/URAM, and fix the port type.

* ``` #pragma HLS PIPELINE II=1 (inside the pixel loop) ```
	* Instructs HLS to schedule the inner loop with initiation interval = 1, i.e., accept one pixel per clock once the pipeline is full. This is the key to line-rate throughput.

### Case 1

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
### Case 2

* Partitioned the second dimension (columns) cyclic factor=4. Unrolling the inner loop by 4 lets HLS issue 4 column reads/writes in parallel for u, pred, and out

* ```PIPELINE II=1 ``` on the tiled loop (c += 4) means every clock you process 4 pixels. Throughput ≈ 4 px/clk (vs 1 px/clk previously), bounded by memory and timing.

```
// diffusion_residual_mm_unroll4.h
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
    // 4-way cyclic partition across the column dimension enables 4 parallel reads/writes
#pragma HLS ARRAY_PARTITION variable=u    dim=2 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=pred dim=2 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=out  dim=2 cyclic factor=4

Row_Loop:
    for (int r = 0; r < H; ++r) {
    Col4_Loop:
        for (int c = 0; c < W; c += 4) {
#pragma HLS PIPELINE II=1

            // Precompute row neighbors (same for all 4 lanes)
            const int r_up   = (r == 0   ) ? 0   : r - 1;
            const int r_down = (r == H-1 ) ? H-1 : r + 1;

        Lane_Loop:
            for (int k = 0; k < 4; ++k) {
#pragma HLS UNROLL factor=4

                const int cc = c + k;
                if (cc >= W) continue;  // tail-guard when W % 4 != 0

                // Column neighbors per-lane (replicate padding)
                const int c_left  = (cc == 0) ? 0   : cc - 1;
                const int c_right = (cc == W-1) ? W-1 : cc + 1;

                ap_fixed<16,6> u_c = u[r][cc];
                ap_fixed<16,6> u_u = u[r_up  ][cc];
                ap_fixed<16,6> u_d = u[r_down][cc];
                ap_fixed<16,6> u_l = u[r][c_left ];
                ap_fixed<16,6> u_r = u[r][c_right];

                // Laplacian (0,1,0; 1,-4,1; 0,1,0)
                ap_fixed<18,8> lap =
                      (ap_fixed<18,8>)u_u
                    + (ap_fixed<18,8>)u_d
                    + (ap_fixed<18,8>)u_l
                    + (ap_fixed<18,8>)u_r
                    - (ap_fixed<18,8>)4.0 * (ap_fixed<18,8>)u_c;

                // Multiply mapped to DSP (hint)
#pragma HLS RESOURCE variable=lap core=AddSub_DSP    // sums can also map to DSPs if available
                ap_fixed<18,8> diff = (ap_fixed<18,8>)alpha * lap;
#pragma HLS RESOURCE variable=diff core=Mul_DSP

                out[r][cc] = (ap_fixed<16,6>)(
                    diff - (ap_fixed<18,8>)pred[r][cc]
                );
            } // lane
        } // c
    } // r
}



```
