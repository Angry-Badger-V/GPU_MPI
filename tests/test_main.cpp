/*
 * COMP3741 – Coursework: self-contained correctness tests
 * Run with:  mpirun -np <P> ./mpi_cuda_tests
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>

#include "mpi_utils.hpp"
#include "check_cuda.hpp"
#include "kernels.hpp"
#include "cpu_reference.hpp"

static int g_pass = 0, g_fail = 0;

static void report(const char* name, bool ok) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    ok ? ++g_pass : ++g_fail;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    auto info = mpi_info();

    // All ranks use GPU 0 (single-GPU assumption)
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { std::cerr << "No GPU found\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    CUDA_CHECK(cudaSetDevice(0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int N = 1 << 18;   // 256 K elements – quick to run
    auto d = dist_1d(N, info.rank, info.size);

    // ── Test 1: AXPY ─────────────────────────────────────────────────────────
    {
        std::vector<float> x(d.N_local, 1.f), y(d.N_local, 2.f), yref(d.N_local, 2.f);
        const float alpha = 3.f;
        cpu_axpy((int)d.N_local, alpha, x.data(), yref.data());  // ref: y = 3*1+2 = 5

        float *dx = nullptr, *dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dy, y.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        launch_axpy((int)d.N_local, alpha, dx, dy, stream);
        CUDA_CHECK(cudaMemcpyAsync(y.data(), dy, d.N_local*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float err = max_abs_diff((int)d.N_local, y.data(), yref.data());
        if (info.rank == 0) report("AXPY correctness", err < 1e-5f);
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }

    // ── Test 2: ADD ──────────────────────────────────────────────────────────
    {
        std::vector<float> x(d.N_local), y(d.N_local), z(d.N_local, 0.f), zref(d.N_local, 0.f);
        for (long long i = 0; i < d.N_local; ++i) {
            x[i] = 0.25f * (float)(i % 17);
            y[i] = -1.5f + 0.5f * (float)(i % 11);
            zref[i] = x[i] + y[i];
        }

        float *dx = nullptr, *dy = nullptr, *dz = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dz, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dy, y.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        launch_add((int)d.N_local, dx, dy, dz, stream);
        CUDA_CHECK(cudaMemcpyAsync(z.data(), dz, d.N_local*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float err = max_abs_diff((int)d.N_local, z.data(), zref.data());
        if (info.rank == 0) report("ADD correctness", err < 1e-5f);
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dz));
    }

    // ── Test 3: COPY ─────────────────────────────────────────────────────────
    {
        std::vector<float> x(d.N_local), y(d.N_local, 0.f), yref(d.N_local);
        for (long long i = 0; i < d.N_local; ++i) {
            x[i] = -2.0f + 0.125f * (float)(i % 31);
            yref[i] = x[i];
        }

        float *dx = nullptr, *dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(dy, 0, d.N_local*sizeof(float), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        launch_copy((int)d.N_local, dx, dy, stream);
        CUDA_CHECK(cudaMemcpyAsync(y.data(), dy, d.N_local*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float err = max_abs_diff((int)d.N_local, y.data(), yref.data());
        if (info.rank == 0) report("COPY correctness", err < 1e-5f);
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
    }

    // ── Test 4: Global reduction ─────────────────────────────────────────────
    {
        // All elements == 1  →  global sum == N
        std::vector<float> x(d.N_local, 1.f);
        float* dx = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, d.N_local*sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(dx, x.data(), d.N_local*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float local  = gpu_reduce_sum(dx, (int)d.N_local, stream);
        float global = 0.f;
        MPI_Allreduce(&local, &global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        if (info.rank == 0)
            report("Reduction global sum", std::fabs(global - (float)N) < 0.5f);
        CUDA_CHECK(cudaFree(dx));
    }

    // ── Test 5: Naïve GEMM ───────────────────────────────────────────────────
    {
        // Use a single rank's local portion only for this test
        const int m = 37, n = 29, k = 23;
        if (d.N_local >= m || info.size == 1) {
            std::vector<float> A(m*k), B(k*n), C(m*n, 0.f), Cref(m*n, 0.f);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < k; ++j)
                    A[i*k+j] = 0.1f * (float)((i + 2*j) % 13);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j)
                    B[i*n+j] = -0.2f + 0.05f * (float)((3*i + j) % 17);

            cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

            float *dA=nullptr,*dB=nullptr,*dC=nullptr;
            CUDA_CHECK(cudaMalloc(&dA, m*k*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dB, k*n*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dC, m*n*sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), m*k*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), k*n*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(dC, 0, m*n*sizeof(float), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            launch_gemm_naive(m, n, k, dA, dB, dC, stream);
            CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, m*n*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float err = max_abs_diff(m*n, C.data(), Cref.data());
            if (info.rank == 0) report("Naive GEMM correctness", err < 1e-3f);
            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        }
    }
    
    // ── Test 6: Tiled GEMM 8×8 identity-like check ────────────────────────
    {
        // Use a single rank's local portion only for this test
        const int m = 64, n = 64, k = 64;
        if (d.N_local >= m || info.size == 1) {
            std::vector<float> A(m*k, 0.f), B(k*n, 0.f), C(m*n, 0.f), Cref(m*n, 0.f);
            for (int i = 0; i < m && i < k; ++i) A[i*k+i] = 1.f;   // identity-like
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j) B[i*n+j] = (float)(i+j);

            cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

            float *dA=nullptr,*dB=nullptr,*dC=nullptr;
            CUDA_CHECK(cudaMalloc(&dA, m*k*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dB, k*n*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dC, m*n*sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), m*k*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), k*n*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(dC, 0, m*n*sizeof(float), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            launch_gemm_tiled8(m, n, k, dA, dB, dC, stream);
            CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, m*n*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float err = max_abs_diff(m*n, C.data(), Cref.data());
            if (info.rank == 0) report("Tiled GEMM 8x8 identity-like correctness", err < 1e-3f);
            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        }
    }    

    // ── Test 7: Tiled GEMM 8×8 ───────────────────────────────────────────────
    {
        // Use a single rank's local portion only for this test
        const int m = 23, n = 19, k = 17;   // dimensions not divisible by tile size
        if (d.N_local >= m || info.size == 1) {
            std::vector<float> A(m*k), B(k*n), C(m*n, 0.f), Cref(m*n, 0.f);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < k; ++j)
                    A[i*k+j] = (float)((i + j) % 7);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j)
                    B[i*n+j] = (float)((2*i + j) % 9) - 4.f;

            cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

            float *dA=nullptr,*dB=nullptr,*dC=nullptr;
            CUDA_CHECK(cudaMalloc(&dA, m*k*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dB, k*n*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dC, m*n*sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), m*k*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), k*n*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(dC, 0, m*n*sizeof(float), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            launch_gemm_tiled8(m, n, k, dA, dB, dC, stream);
            CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, m*n*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float err = max_abs_diff(m*n, C.data(), Cref.data());
            if (info.rank == 0) report("Tiled GEMM 8x8 correctness", err < 1e-3f);
            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        }
    }

    // ── Test 8: Tiled GEMM 16×16 identity-like check ────────────────────────
    {
        // Use a single rank's local portion only for this test
        const int m = 64, n = 64, k = 64;
        if (d.N_local >= m || info.size == 1) {
            std::vector<float> A(m*k, 0.f), B(k*n, 0.f), C(m*n, 0.f), Cref(m*n, 0.f);
            for (int i = 0; i < m && i < k; ++i) A[i*k+i] = 1.f;   // identity-like
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j) B[i*n+j] = (float)(i+j);

            cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

            float *dA=nullptr,*dB=nullptr,*dC=nullptr;
            CUDA_CHECK(cudaMalloc(&dA, m*k*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dB, k*n*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dC, m*n*sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), m*k*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), k*n*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(dC, 0, m*n*sizeof(float), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            launch_gemm_tiled16(m, n, k, dA, dB, dC, stream);
            CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, m*n*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float err = max_abs_diff(m*n, C.data(), Cref.data());
            if (info.rank == 0) report("Tiled GEMM 16x16 identity-like correctness", err < 1e-3f);
            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        }
    }
    
    // ── Test 9: Tiled GEMM 16×16 ───────────────────────────────────────────────
    {
        // Use a single rank's local portion only for this test
        const int m = 37, n = 29, k = 23;   // dimensions not divisible by tile size
        if (d.N_local >= m || info.size == 1) {
            std::vector<float> A(m*k), B(k*n), C(m*n, 0.f), Cref(m*n, 0.f);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < k; ++j)
                    A[i*k+j] = (float)((i + j) % 7);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j)
                    B[i*n+j] = (float)((2*i + j) % 9) - 4.f;

            cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

            float *dA=nullptr,*dB=nullptr,*dC=nullptr;
            CUDA_CHECK(cudaMalloc(&dA, m*k*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dB, k*n*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dC, m*n*sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), m*k*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), k*n*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(dC, 0, m*n*sizeof(float), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            launch_gemm_tiled16(m, n, k, dA, dB, dC, stream);
            CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, m*n*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float err = max_abs_diff(m*n, C.data(), Cref.data());
            if (info.rank == 0) report("Tiled GEMM 16x16 correctness", err < 1e-3f);
            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        }
    }
   
    // ── Test 10: Tiled GEMM 32×32 identity-like check ────────────────────────
    {
        // Use a single rank's local portion only for this test
        const int m = 64, n = 64, k = 64;
        if (d.N_local >= m || info.size == 1) {
            std::vector<float> A(m*k, 0.f), B(k*n, 0.f), C(m*n, 0.f), Cref(m*n, 0.f);
            for (int i = 0; i < m && i < k; ++i) A[i*k+i] = 1.f;   // identity-like
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j) B[i*n+j] = (float)(i+j);

            cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

            float *dA=nullptr,*dB=nullptr,*dC=nullptr;
            CUDA_CHECK(cudaMalloc(&dA, m*k*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dB, k*n*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dC, m*n*sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), m*k*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), k*n*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(dC, 0, m*n*sizeof(float), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            launch_gemm_tiled32(m, n, k, dA, dB, dC, stream);
            CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, m*n*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float err = max_abs_diff(m*n, C.data(), Cref.data());
            if (info.rank == 0) report("Tiled GEMM 32x32 identity-like correctness", err < 1e-3f);
            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        }
    }

    // ── Test 11: Tiled GEMM 32×32 ─────────────────────────────────────────────
    {
        // Use a single rank's local portion only for this test
        const int m = 65, n = 33, k = 70;   // dimensions not divisible by tile size
        if (d.N_local >= m || info.size == 1) {
            std::vector<float> A(m*k), B(k*n), C(m*n, 0.f), Cref(m*n, 0.f);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < k; ++j)
                    A[i*k+j] = 0.01f * (float)((i*j + j) % 23);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j)
                    B[i*n+j] = 0.02f * (float)((i + 4*j) % 19) - 0.1f;

            cpu_gemm(m, n, k, A.data(), B.data(), Cref.data());

            float *dA=nullptr,*dB=nullptr,*dC=nullptr;
            CUDA_CHECK(cudaMalloc(&dA, m*k*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dB, k*n*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dC, m*n*sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), m*k*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), k*n*sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(dC, 0, m*n*sizeof(float), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            launch_gemm_tiled32(m, n, k, dA, dB, dC, stream);
            CUDA_CHECK(cudaMemcpyAsync(C.data(), dC, m*n*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float err = max_abs_diff(m*n, C.data(), Cref.data());
            if (info.rank == 0) report("Tiled GEMM 32x32 correctness", err < 1e-3f);
            CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        }
    }

    if (info.rank == 0)
        std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed.\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Finalize();
    return g_fail > 0 ? 1 : 0;
}