/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * GEMM example – runs all 6 GEMM versions, validates against CPU reference,
 * and prints elapsed time + max error for each.
 */

#include "fastcuda/gemm.hpp"
#include "fastcuda/gemm.h"
#include "fastcuda/runtime.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void FillMatrix(std::vector<float>* v, float scale) {
    for (std::size_t i = 0; i < v->size(); ++i)
    (*v)[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * scale;
}

void RunOne(fastcuda::GemmAlgorithm algo,
            const fastcuda::GemmConfig& cfg,
            const std::vector<float>& a,
            const std::vector<float>& b,
            const std::vector<float>& c_ref) {
    std::vector<float> c_out(c_ref.size(), 0.0f);
    fastcuda::GemmTiming t = fastcuda::RunSgemmHost(
        algo, cfg, a.data(), b.data(), NULL, c_out.data(), 5, 20);

    float err = fastcuda::MaxAbsDiff(c_out.data(), c_ref.data(), c_ref.size());
    std::cout << std::left << std::setw(20)
              << fastcuda::GemmAlgorithmName(algo)
              << std::fixed << std::setprecision(4)
              << "  elapsed_ms=" << std::setw(10) << t.elapsed_ms
              << std::scientific << std::setprecision(2)
              << "  max_err=" << err << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    int m = (argc > 1) ? std::atoi(argv[1]) : 512;
    int n = (argc > 2) ? std::atoi(argv[2]) : 512;
    int k = (argc > 3) ? std::atoi(argv[3]) : 512;

    std::cout << "== FastCuda GEMM Example ==\n";
    std::cout << fastcuda::FormatDeviceSummary(fastcuda::QueryDevices());
    std::cout << "m=" << m << "  n=" << n << "  k=" << k << "\n\n";

    fastcuda::GemmConfig cfg;
    cfg.m = m; cfg.n = n; cfg.k = k;
    cfg.lda = k; cfg.ldb = n; cfg.ldc = n;

    std::vector<float> a(static_cast<std::size_t>(m) * k);
    std::vector<float> b(static_cast<std::size_t>(k) * n);
    FillMatrix(&a, 0.125f);
    FillMatrix(&b, 0.0625f);

    /* CPU reference */
    std::vector<float> c_ref(static_cast<std::size_t>(m) * n, 0.0f);
    fastcuda::ReferenceSgemm(cfg, a.data(), b.data(), NULL, c_ref.data());

    /* SGEMM v1-v4 (FP32 CUDA core paths) */
    RunOne(fastcuda::GemmAlgorithm::kNaiveV1,     cfg, a, b, c_ref);
    RunOne(fastcuda::GemmAlgorithm::kSharedMemV2,  cfg, a, b, c_ref);
    RunOne(fastcuda::GemmAlgorithm::kRegisterV3,   cfg, a, b, c_ref);
    RunOne(fastcuda::GemmAlgorithm::kWarpV4,       cfg, a, b, c_ref);

    /* TF32 v5 – note: result will differ slightly due to TF32 precision */
    RunOne(fastcuda::GemmAlgorithm::kTF32V5,       cfg, a, b, c_ref);

    std::cout << "\n(HGEMM v6 requires FP16 input - run fastcuda_hgemm_example for a dedicated test)\n";
    return 0;
}
