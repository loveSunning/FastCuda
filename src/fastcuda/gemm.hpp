/*
 Copyright 2026 victor
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0
*/

#pragma once

#include "fastcuda/export.hpp"

#include <cstddef>

namespace fastcuda {

enum class GemmAlgorithm {
    kNaive = 0,
    kTiled = 1,
    kRegisterBlocked = 2,
};

struct GemmConfig {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    float alpha;
    float beta;
    void* stream;

    GemmConfig()
        : m(0),
          n(0),
          k(0),
          lda(0),
          ldb(0),
          ldc(0),
          alpha(1.0f),
          beta(0.0f),
          stream(NULL) {}
};

struct GemmTiming {
    float elapsed_ms;

    GemmTiming() : elapsed_ms(0.0f) {}
};

FASTCUDA_API const char* GemmAlgorithmName(GemmAlgorithm algorithm);
FASTCUDA_API void LaunchSgemmDevice(
    GemmAlgorithm algorithm,
    const GemmConfig& config,
    const float* a_device,
    const float* b_device,
    float* c_device);
FASTCUDA_API GemmTiming BenchmarkSgemmDevice(
    GemmAlgorithm algorithm,
    const GemmConfig& config,
    const float* a_device,
    const float* b_device,
    float* c_device,
    int warmup_iterations,
    int timed_iterations);
FASTCUDA_API GemmTiming RunSgemmHost(
    GemmAlgorithm algorithm,
    const GemmConfig& config,
    const float* a_host,
    const float* b_host,
    const float* c_input_host,
    float* c_output_host,
    int warmup_iterations,
    int timed_iterations);
FASTCUDA_API void ReferenceSgemm(
    const GemmConfig& config,
    const float* a_host,
    const float* b_host,
    const float* c_input_host,
    float* c_output_host);
FASTCUDA_API float MaxAbsDiff(
    const float* lhs,
    const float* rhs,
    std::size_t element_count);

}  // namespace fastcuda