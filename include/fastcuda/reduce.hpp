/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "fastcuda/export.h"
#include <cstddef>

namespace fastcuda {

enum class ReduceAlgorithm {
    kBaselineV0        = 0,
    kNoDivergenceV1    = 1,
    kNoBankConflictV2  = 2,
    kAddDuringLoadV3   = 3,
    kUnrollLastWarpV4  = 4,
    kCompletelyUnrollV5= 5,
    kMultiAddV6        = 6,
    kShuffleV7         = 7,
};

struct ReduceConfig {
    int   n;
    void* stream;

    ReduceConfig() : n(0), stream(NULL) {}
};

struct ReduceTiming {
    float elapsed_ms;
    ReduceTiming() : elapsed_ms(0.0f) {}
};

FASTCUDA_API const char* ReduceAlgorithmName(ReduceAlgorithm algo);

/* Reduce sum – device pointers */
FASTCUDA_API void LaunchReduceSum(
    ReduceAlgorithm algo,
    const ReduceConfig& config,
    const float* input,
    float* output);

FASTCUDA_API ReduceTiming BenchmarkReduceSum(
    ReduceAlgorithm algo,
    const ReduceConfig& config,
    const float* input,
    float* output,
    int warmup_iters,
    int timed_iters);

/* Host convenience */
FASTCUDA_API ReduceTiming RunReduceSumHost(
    ReduceAlgorithm algo,
    const ReduceConfig& config,
    const float* input_host,
    float* output_host,
    int warmup_iters,
    int timed_iters);

/* CPU reference */
FASTCUDA_API float ReduceSumReference(const float* input, int n);

}  // namespace fastcuda
