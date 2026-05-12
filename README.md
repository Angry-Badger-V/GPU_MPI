# COMP3741 – MPI + CUDA Coursework

## Project structure
```
.
├── report.pdf        # My report
├── CMakeLists.txt
├── README.md
├── include/          # All header files
│   ├── benchmarks.hpp
│   ├── check_cuda.hpp
│   ├── cli.hpp
│   ├── cpu_reference.hpp
│   ├── kernels.hpp
│   ├── mpi_distribution.hpp
│   ├── mpi_utils.hpp
│   └── timer.hpp
├── src/              # Implementation files
│   ├── benchmarks.cpp
│   ├── cpu_reference.cpp
│   ├── cuda_kernels.cu
│   ├── main.cpp
│   └── mpi_distribution.cpp
├── tests/
│   └── test_main.cpp
├── ncc_bandwidth.slurm  # run bandwidth experiments for axpy, add and copy kernels
├── ncc_tiles.slurm      # run GEMM tiled comparison experiments for 8x8, 16x16 and 32x32
├── ncc_compare.slurm    # run naive vs tiled 32x32 comparison experiment
├── ncc_scaling.slurm    # Strong-scaling sweep (1/2/4 ranks, 1 GPU)
└── ncc_profiler.slurm   # run profiler on 4 rank 32x32 tiled GEMM
```

## Single-GPU assumption
All MPI ranks share **one GPU (device 0)**. Each rank allocates its own device
memory and launches its own CUDA kernels. No multi-GPU or peer-access code is
needed.

## Build (NCC cluster)
```bash
module purge
module load cuda
module load intel-oneapi/2024.1.0/mpi

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..
```

> **CUDA architecture**: edit `CUDA_ARCHITECTURES` in `CMakeLists.txt` if
> your GPU is not V100 (sm_70) or A100 (sm_80).  
> Run `nvidia-smi` to find the GPU model, then look up its compute capability.

## Run
```bash
# AXPY bandwidth test
mpirun -np 2 ./build/mpi_cuda_coursework --mode axpy --N 50000000 --csv results.csv

# ADD bandwidth test
mpirun -np 2 ./build/mpi_cuda_coursework --mode add --N 50000000 --csv results.csv

# COPY bandwidth test
mpirun -np 2 ./build/mpi_cuda_coursework --mode copy --N 50000000 --csv results.csv

# Parallel reduction
mpirun -np 2 ./build/mpi_cuda_coursework --mode reduce --N 50000000 --seeds 5 --seed0 3000 --csv results.csv

# GEMM – naive kernel
mpirun -np 2 ./build/mpi_cuda_coursework --mode gemm \
    --M 4096 --N 4096 --K 4096 --kernel naive --csv results.csv

# GEMM – tiled 8x8 kernel
mpirun -np 2 ./build/mpi_cuda_coursework --mode gemm \
    --M 4096 --N 4096 --K 4096 --kernel tiled8 --csv results.csv
    
# GEMM – tiled 16x16 kernel
mpirun -np 2 ./build/mpi_cuda_coursework --mode gemm \
    --M 4096 --N 4096 --K 4096 --kernel tiled16 --csv results.csv
    
# GEMM – tiled 32x32 kernel
mpirun -np 2 ./build/mpi_cuda_coursework --mode gemm \
    --M 4096 --N 4096 --K 4096 --kernel tiled32 --csv results.csv

# Profiler for GEMM
nsys profile --stats=true -o gemm_tiled32_1rank \
    mpirun -np 4 ./build/mpi_cuda_coursework --mode gemm --M 4096 --N 4096 --K 4096 --kernel tiled32 --csv results.csv
```

## Correctness tests
```bash
mpirun -np 2 ./build/mpi_cuda_tests
```

## SLURM (NCC)
```bash
sbatch ncc_bandwidth.slurm  # run bandwidth experiments for axpy, add and copy kernels
sbatch ncc_tiles.slurm      # run GEMM tiled comparison experiments for 8x8, 16x16 and 32x32
sbatch ncc_compare.slurm    # run naive vs tiled 32x32 comparison experiment
sbatch ncc_scaling.slurm    # strong-scaling sweep
sbatch ncc_profiler.slurm   # run profiler on 4 rank 32x32 tiled GEMM
sbatch 
```
If for some reason there is a problem like:
sbatch: error: invalid partition specified: gpu 
sbatch: error: Batch job submission failed: Invalid partition name specified
Then this will stem from the partitions in the slurm files, I am using: #SBATCH --partition=ug-gpu-small
Because that is what my permissions allow me to use, this may need to be changed if you have different permissions

## Reproducibility
- Random data is seeded deterministically (`seed = base + rank`).
- For reduction kernel seeds can be selected using argument --seed0
- Each CSV row is tagged with the MPI rank, mode, kernel, and problem size.
- Warm-up kernel launches precede all timed runs.
