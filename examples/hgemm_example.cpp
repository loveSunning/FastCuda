/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * HGEMM example - runs FP16 input / FP32 accumulate GEMM and validates
 * against a CPU FP32 reference.
 */

#include "fastcuda/gemm.hpp"
#include "fastcuda/runtime.hpp"

#include <cuda_fp16.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void FillMatrix(std::vector<float>* values, float scale) {
    for (std::size_t i = 0; i < values->size(); ++i)
    (*values)[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * scale;
}

void ConvertToHalf(const std::vector<float>& src, std::vector<half>* dst) {
    dst->resize(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        (*dst)[i] = __float2half(src[i]);
}

}  // namespace

int main(int argc, char** argv) {
    int m = (argc > 1) ? std::atoi(argv[1]) : 512;
    int n = (argc > 2) ? std::atoi(argv[2]) : 512;
    int k = (argc > 3) ? std::atoi(argv[3]) : 512;

    std::cout << "== FastCuda HGEMM Example ==\n";
    std::cout << fastcuda::FormatDeviceSummary(fastcuda::QueryDevices());
    std::cout << "m=" << m << "  n=" << n << "  k=" << k << "\n\n";

    fastcuda::GemmConfig cfg;
    cfg.m = m;
    cfg.n = n;
    cfg.k = k;
    cfg.lda = k;
    cfg.ldb = n;
    cfg.ldc = n;

    std::vector<float> a_fp32(static_cast<std::size_t>(m) * k);
    std::vector<float> b_fp32(static_cast<std::size_t>(k) * n);
    FillMatrix(&a_fp32, 0.125f);
    FillMatrix(&b_fp32, 0.0625f);

    std::vector<half> a_fp16;
    std::vector<half> b_fp16;
    ConvertToHalf(a_fp32, &a_fp16);
    ConvertToHalf(b_fp32, &b_fp16);

    std::vector<float> c_ref(static_cast<std::size_t>(m) * n, 0.0f);
    std::vector<float> c_out(static_cast<std::size_t>(m) * n, 0.0f);
    fastcuda::ReferenceSgemm(cfg, a_fp32.data(), b_fp32.data(), NULL, c_ref.data());

    fastcuda::GemmTiming t = fastcuda::RunHgemmHost(
        cfg, a_fp16.data(), b_fp16.data(), NULL, c_out.data(), 5, 20);

    float err = fastcuda::MaxAbsDiff(c_out.data(), c_ref.data(), c_ref.size());
    std::cout << std::left << std::setw(20) << "hgemm_v6"
              << std::fixed << std::setprecision(4)
              << "  elapsed_ms=" << std::setw(10) << t.elapsed_ms
              << std::scientific << std::setprecision(2)
              << "  max_err=" << err << "\n";
    return 0;
}