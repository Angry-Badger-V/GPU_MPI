#!/bin/bash
#SBATCH --job-name=gemm-tiles
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:20:00
#SBATCH --output=gemm-tiles-%j.out

module purge
module load cuda
module load intel-oneapi/2024.1.0/mpi

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..

echo "SLURM mem vars:"
env | grep '^SLURM_MEM' || true

for KERNEL in tiled8 tiled16 tiled32; do
    for SIZE in 2048 4096 8192; do
        echo "=== KERNEL=$KERNEL SIZE=$SIZE ==="
        env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_GPU -u SLURM_MEM_PER_NODE \
        srun --mem=0 -n 1 build/mpi_cuda_coursework \
            --mode gemm \
            --M $SIZE --N $SIZE --K $SIZE \
            --kernel $KERNEL \
            --csv ${KERNEL}_${SIZE}.csv
    done
done