#include "mpi_distribution.hpp"

void build_counts_displs_rows(int M, int cols, int size,
                               std::vector<int>& counts,
                               std::vector<int>& displs) {
    counts.resize(size);
    displs.resize(size);
    int base = M / size, rem = M % size, offset = 0;
    for (int r = 0; r < size; ++r) {
        int rows   = base + (r < rem ? 1 : 0);
        counts[r]  = rows * cols;
        displs[r]  = offset;
        offset    += counts[r];
    }
}
