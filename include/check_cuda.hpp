#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA ERROR %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

inline void cuda_check_last(const char* msg, const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA KERNEL ERROR %s:%d (%s): %s\n",
                     file, line, msg, cudaGetErrorString(err));
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK_LAST(msg) cuda_check_last((msg), __FILE__, __LINE__)
