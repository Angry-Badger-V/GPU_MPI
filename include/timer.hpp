#pragma once
#include <mpi.h>
#include <cuda_runtime.h>

struct CpuTimer {
    double t0 = 0.0;
    void   start()        { t0 = MPI_Wtime(); }
    double stop()  const  { return MPI_Wtime() - t0; }
};

struct GpuTimer {
    cudaEvent_t start_evt{}, stop_evt{};
    GpuTimer()  { cudaEventCreate(&start_evt); cudaEventCreate(&stop_evt); }
    ~GpuTimer() { cudaEventDestroy(start_evt); cudaEventDestroy(stop_evt); }

    void  start(cudaStream_t s = 0) { cudaEventRecord(start_evt, s); }
    float stop (cudaStream_t s = 0) {
        cudaEventRecord(stop_evt, s);
        cudaEventSynchronize(stop_evt);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_evt, stop_evt);
        return ms;
    }
};
