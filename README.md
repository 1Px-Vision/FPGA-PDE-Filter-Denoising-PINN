# HLS-Optimized Parallel PDE Filter (FPGA)

A lightweight, line-rate PDE residual filter for images, designed for Vitis HLS and easy integration with DPU pipelines. It implements the 3×3 Laplacian stencil.

$$
\nabla^{2} u \;=\
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}
*\ u
$$


diffusion residual: ``` residual = α·(∇²u) − pred ```, where pred is typically the DPU model output.
