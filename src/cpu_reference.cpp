#include "cpu_reference.hpp"
#include <cmath>
#include <numeric>

void cpu_axpy(int n, float alpha, const float* x, float* y) {
    for (int i = 0; i < n; ++i) y[i] = alpha * x[i] + y[i];
}

float cpu_reduce_sum(int n, const float* x) {
    // Use double accumulator to reduce rounding for the reference
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)x[i];
    return (float)s;
}

void cpu_gemm(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += A[i*K + k] * B[k*N + j];
            C[i*N + j] = acc;
        }
}

float max_abs_diff(int n, const float* a, const float* b) {
    float mx = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}
