#include "kernels.hpp"
#include "check_cuda.hpp"
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════════════════
// Part A1 – Vector operations
// ═══════════════════════════════════════════════════════════════════════════════

// --- AXPY: y = alpha*x + y ---------------------------------------------------
__global__ void k_axpy(int n, float alpha,
                       const float* __restrict__ x,
                             float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i < n) y[i] = alpha * x[i] + y[i];
    int stride = blockDim.x * gridDim.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        y[i]            = fmaf(alpha, x[i],            y[i]);
        y[i + stride]   = fmaf(alpha, x[i + stride],   y[i + stride]);
        y[i + 2*stride] = fmaf(alpha, x[i + 2*stride], y[i + 2*stride]);
        y[i + 3*stride] = fmaf(alpha, x[i + 3*stride], y[i + 3*stride]);
    }
    for (; i < n; i += stride) {
        y[i] = fmaf(alpha, x[i], y[i]);
    }
}

void launch_axpy(int n, float alpha, const float* x, float* y, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    k_axpy<<<grid, block, 0, stream>>>(n, alpha, x, y);
    CUDA_CHECK_LAST("k_axpy");
}

// --- ADD: z = x + y ----------------------------------------------------------
__global__ void k_add(int n,
                      const float* __restrict__ x,
                      const float* __restrict__ y,
                            float* __restrict__ z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) z[i] = x[i] + y[i];
}

void launch_add(int n, const float* x, const float* y, float* z, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    k_add<<<grid, block, 0, stream>>>(n, x, y, z);
    CUDA_CHECK_LAST("k_add");
}

// --- COPY: y = x -------------------------------------------------------------
__global__ void k_copy(int n,
                       const float* __restrict__ x,
                             float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) y[i] = x[i];
}

void launch_copy(int n, const float* x, float* y, cudaStream_t stream) {
    int block = 256, grid = (n + block - 1) / block;
    k_copy<<<grid, block, 0, stream>>>(n, x, y);
    CUDA_CHECK_LAST("k_copy");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part A2 – Parallel reduction (tree reduction, multi-stage)
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr int REDUCE_BLOCK = 256;

// Warp-level reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

// Stage 1: each block reduces part of the original input into one partial sum
__global__ void k_reduce_block_sum(const float* __restrict__ x,
                                   float* __restrict__ partials,
                                   int n) {
    __shared__ float sdata[REDUCE_BLOCK];

    int tid = threadIdx.x;
    int i   = blockIdx.x * (blockDim.x * 2) + tid;
    int grid_stride = blockDim.x * 2 * gridDim.x;

    float sum = 0.f;

    // Grid-stride loop: each thread accumulates multiple elements
    while (i < n) {
        sum += x[i];
        if (i + blockDim.x < n) sum += x[i + blockDim.x];
        i += grid_stride;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    if (blockDim.x >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid <  64) sdata[tid] += sdata[tid +  64];
        __syncthreads();
    }

    // Final warp reduction (no __syncthreads needed within a warp)
    if (tid < 32) {
        float v = sdata[tid];
        if (blockDim.x >= 64) v += sdata[tid + 32];
        v = warp_reduce_sum(v);
        if (tid == 0) partials[blockIdx.x] = v;
    }
}

// Stage 2+: reduce partial sums into fewer partial sums
// Same idea as Stage 1, but input is now the partials array.
__global__ void k_reduce_partials(const float* __restrict__ x,
                                  float* __restrict__ partials,
                                  int n) {
    __shared__ float sdata[REDUCE_BLOCK];

    int tid = threadIdx.x;
    int i   = blockIdx.x * (blockDim.x * 2) + tid;

    float sum = 0.f;

    // For later stages the arrays are much smaller, so a simple 2-load pattern
    // is enough. A grid-stride loop is still safe and general.
    int grid_stride = blockDim.x * 2 * gridDim.x;
    while (i < n) {
        sum += x[i];
        if (i + blockDim.x < n) sum += x[i + blockDim.x];
        i += grid_stride;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    if (blockDim.x >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid <  64) sdata[tid] += sdata[tid +  64];
        __syncthreads();
    }

    // Final warp reduction
    if (tid < 32) {
        float v = sdata[tid];
        if (blockDim.x >= 64) v += sdata[tid + 32];
        v = warp_reduce_sum(v);
        if (tid == 0) partials[blockIdx.x] = v;
    }
}

float gpu_reduce_sum(const float* d_x, int n, cudaStream_t stream) {
    const int block = REDUCE_BLOCK;

    // Because each block handles up to 2*block elements initially
    int grid = (n + (2 * block - 1)) / (2 * block);
    if (grid < 1) grid = 1;

    float* d_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partials, grid * sizeof(float)));

    // Stage 1: reduce the original input into block partial sums
    k_reduce_block_sum<<<grid, block, 0, stream>>>(d_x, d_partials, n);
    CUDA_CHECK_LAST("k_reduce_block_sum");

    // Stage 2+: repeatedly reduce the partial sums until one value remains
    while (grid > 1) {
        int next = (grid + (2 * block - 1)) / (2 * block);
        if (next < 1) next = 1;

        float* d_next = nullptr;
        CUDA_CHECK(cudaMalloc(&d_next, next * sizeof(float)));

        k_reduce_partials<<<next, block, 0, stream>>>(d_partials, d_next, grid);
        CUDA_CHECK_LAST("k_reduce_partials");

        CUDA_CHECK(cudaFree(d_partials));
        d_partials = d_next;
        grid = next;
    }

    float h = 0.f;
    CUDA_CHECK(cudaMemcpyAsync(&h, d_partials, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_partials));
    return h;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part B1 – Naïve GEMM
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void k_gemm_naive(int M, int N, int K,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                                   float* __restrict__ C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

void launch_gemm_naive(int M, int N, int K,
                       const float* A, const float* B, float* C,
                       cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    k_gemm_naive<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAST("k_gemm_naive");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Part B2 – Tiled GEMM with shared memory
// ═══════════════════════════════════════════════════════════════════════════════

// TODO (students): tune TILE size (try 16 and 32), experiment with
//                  #pragma unroll and __ldg() for extra performance
template <int TILE>
__global__ void k_gemm_tiled(int M, int N, int K,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                                   float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;

    // Sweep over tiles in the K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // Load one tile of A and one tile of B into shared memory
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();

        // Multiply the two tiles together
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void launch_gemm_tiled8(int M, int N, int K,
                         const float* A, const float* B, float* C,
                         cudaStream_t stream) {
    constexpr int T = 8;
    dim3 block(T, T);
    dim3 grid((N + T - 1) / T, (M + T - 1) / T);

    k_gemm_tiled<T><<<grid, block, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAST("k_gemm_tiled<8>");
}

void launch_gemm_tiled16(int M, int N, int K,
                         const float* A, const float* B, float* C,
                         cudaStream_t stream) {
    constexpr int T = 16;
    dim3 block(T, T);
    dim3 grid((N + T - 1) / T, (M + T - 1) / T);

    k_gemm_tiled<T><<<grid, block, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAST("k_gemm_tiled<16>");
}

void launch_gemm_tiled32(int M, int N, int K,
                         const float* A, const float* B, float* C,
                         cudaStream_t stream) {
    constexpr int T = 32;
    dim3 block(T, T);
    dim3 grid((N + T - 1) / T, (M + T - 1) / T);

    k_gemm_tiled<T><<<grid, block, 0, stream>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAST("k_gemm_tiled<32>");
}
