#!/bin/bash
#SBATCH --job-name=mpi-cuda-scaling
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:30:00
#SBATCH --output=scaling-%j.out

module purge
module load cuda
module load intel-oneapi/2024.1.0/mpi

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..

echo "SLURM mem vars before runs:"
env | grep '^SLURM_MEM' || true

for NP in 1 2 4; do
    echo "=== NP=$NP ==="
    srun --mem=0 -n $NP build/mpi_cuda_coursework \
        --mode gemm --M 4096 --N 4096 --K 4096 \
        --kernel tiled32 --csv results_np${NP}.csv
done