#!/bin/bash
#SBATCH --job-name=bandwidth-tests
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:15:00
#SBATCH --output=bandwidth-%j.out

module purge
module load cuda
module load intel-oneapi/2024.1.0/mpi

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..

echo "SLURM mem vars:"
env | grep '^SLURM_MEM' || true

for MODE in axpy add copy; do
    for N in 5000000 25000000 50000000; do
        echo "=== MODE=$MODE N=$N ==="
        env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_GPU -u SLURM_MEM_PER_NODE \
        srun --mem=0 -n 1 build/mpi_cuda_coursework \
            --mode $MODE --N $N \
            --csv ${MODE}_N${N}.csv
    done
done