#pragma once
#include <mpi.h>
#include <vector>

struct MpiInfo { int rank = 0; int size = 1; };

inline MpiInfo mpi_info() {
    MpiInfo i;
    MPI_Comm_rank(MPI_COMM_WORLD, &i.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &i.size);
    return i;
}

// ── 1-D vector distribution ───────────────────────────────────────────────────
struct Dist1D { long long N_global = 0, N_local = 0, offset = 0; };

inline Dist1D dist_1d(long long N, int rank, int size) {
    Dist1D d; d.N_global = N;
    long long base = N / size, rem = N % size;
    d.N_local  = base + (rank < rem ? 1 : 0);
    d.offset   = base * rank + (rank < rem ? rank : rem);
    return d;
}

// ── Row-distributed matrix ────────────────────────────────────────────────────
struct DistRows { int M_global=0, N_global=0, K_global=0; int M_local=0; int row_offset=0; };

inline DistRows dist_rows(int M, int N, int K, int rank, int size) {
    DistRows d; d.M_global=M; d.N_global=N; d.K_global=K;
    int base = M / size, rem = M % size;
    d.M_local     = base + (rank < rem ? 1 : 0);
    d.row_offset  = base * rank + (rank < rem ? rank : rem);
    return d;
}
