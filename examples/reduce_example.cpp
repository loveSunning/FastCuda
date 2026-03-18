/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Reduce example – runs all 8 reduce versions on the same input,
 * compares against CPU reference, prints elapsed time + error.
 */

#include "fastcuda/reduce.hpp"
#include "fastcuda/runtime.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void FillInput(std::vector<float>* v) {
    for (std::size_t i = 0; i < v->size(); ++i)
        (*v)[i] = static_cast<float>((i % 7) - 3) * 0.1f;
}

void RunOne(fastcuda::ReduceAlgorithm algo,
            const fastcuda::ReduceConfig& cfg,
            const std::vector<float>& input,
            float ref_sum) {
    float gpu_sum = 0.0f;
    fastcuda::ReduceTiming t = fastcuda::RunReduceSumHost(
        algo, cfg, input.data(), &gpu_sum, 5, 20);

    float err = std::fabs(gpu_sum - ref_sum);
    std::cout << std::left << std::setw(28)
              << fastcuda::ReduceAlgorithmName(algo)
              << std::fixed << std::setprecision(4)
              << "  elapsed_ms=" << std::setw(10) << t.elapsed_ms
              << std::scientific << std::setprecision(4)
              << "  abs_err=" << err
              << std::fixed << std::setprecision(4)
              << "  sum=" << gpu_sum << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);  /* 1M elements */

    std::cout << "== FastCuda Reduce Example ==\n";
    std::cout << fastcuda::FormatDeviceSummary(fastcuda::QueryDevices());
    std::cout << "n=" << n << "\n\n";

    std::vector<float> input(static_cast<std::size_t>(n));
    FillInput(&input);

    float ref = fastcuda::ReduceSumReference(input.data(), n);
    std::cout << "reference_sum=" << ref << "\n\n";

    fastcuda::ReduceConfig cfg;
    cfg.n = n;

    RunOne(fastcuda::ReduceAlgorithm::kBaselineV0,        cfg, input, ref);
    RunOne(fastcuda::ReduceAlgorithm::kNoDivergenceV1,    cfg, input, ref);
    RunOne(fastcuda::ReduceAlgorithm::kNoBankConflictV2,  cfg, input, ref);
    RunOne(fastcuda::ReduceAlgorithm::kAddDuringLoadV3,   cfg, input, ref);
    RunOne(fastcuda::ReduceAlgorithm::kUnrollLastWarpV4,  cfg, input, ref);
    RunOne(fastcuda::ReduceAlgorithm::kCompletelyUnrollV5,cfg, input, ref);
    RunOne(fastcuda::ReduceAlgorithm::kMultiAddV6,        cfg, input, ref);
    RunOne(fastcuda::ReduceAlgorithm::kShuffleV7,         cfg, input, ref);

    return 0;
}
