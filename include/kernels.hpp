#pragma once
#include <cuda_runtime.h>

// ── Part A1: vector operations ────────────────────────────────────────────────
void launch_axpy(int n, float alpha, const float* x, float* y,
                 cudaStream_t stream = 0);
void launch_add (int n, const float* x, const float* y, float* z,
                 cudaStream_t stream = 0);   // z = x + y
void launch_copy(int n, const float* x, float* y,
                 cudaStream_t stream = 0);   // y = x

// ── Part A2: parallel reduction ───────────────────────────────────────────────
float gpu_reduce_sum(const float* d_x, int n, cudaStream_t stream = 0);

// ── Part B: GEMM ──────────────────────────────────────────────────────────────
void launch_gemm_naive  (int M, int N, int K,
                         const float* A, const float* B, float* C,
                         cudaStream_t stream = 0);
void launch_gemm_tiled8(int M, int N, int K,
                         const float* A, const float* B, float* C,
                         cudaStream_t stream = 0);
void launch_gemm_tiled16(int M, int N, int K,
                         const float* A, const float* B, float* C,
                         cudaStream_t stream = 0);
void launch_gemm_tiled32(int M, int N, int K,
                         const float* A, const float* B, float* C,
                         cudaStream_t stream = 0);