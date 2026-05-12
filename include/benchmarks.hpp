#pragma once
#include <string>

void append_csv(const std::string& path, int rank,
                const std::string& mode, const std::string& kernel,
                long long N, int M, int Nmat, int K,
                double ms_gpu, double gflops, double gbs,
                double comm_ms, double scat_ms, double bcst_ms, double barr_ms);
