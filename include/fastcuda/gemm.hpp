/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "fastcuda/export.h"
#include <cstddef>

namespace fastcuda {

enum class GemmAlgorithm {
    kNaiveV1     = 0,
    kSharedMemV2 = 1,
    kRegisterV3  = 2,
    kWarpV4      = 3,
    kTF32V5      = 4,
    kHgemmV6     = 5,
};

struct GemmConfig {
    int   m;
    int   n;
    int   k;
    int   lda;
    int   ldb;
    int   ldc;
    float alpha;
    float beta;
    void* stream;

    GemmConfig()
        : m(0), n(0), k(0),
          lda(0), ldb(0), ldc(0),
          alpha(1.0f), beta(0.0f),
          stream(NULL) {}
};

struct GemmTiming {
    float elapsed_ms;
    GemmTiming() : elapsed_ms(0.0f) {}
};

FASTCUDA_API const char* GemmAlgorithmName(GemmAlgorithm algo);

/* SGEMM v1-v4 (FP32, CUDA core paths) – device pointers */
FASTCUDA_API void LaunchSgemm(
    GemmAlgorithm algo,
    const GemmConfig& config,
    const float* A,
    const float* B,
    float* C);

/* TF32 GEMM v5 – device pointers, FP32 I/O, TF32 Tensor Core compute */
FASTCUDA_API void LaunchGemmTF32(
    const GemmConfig& config,
    const float* A,
    const float* B,
    float* C);

/* HGEMM v6 – device pointers, FP16 input, FP32 output */
FASTCUDA_API void LaunchHgemm(
    const GemmConfig& config,
    const void* A,
    const void* B,
    float* C);

/* Benchmark variants */
FASTCUDA_API GemmTiming BenchmarkSgemm(
    GemmAlgorithm algo,
    const GemmConfig& config,
    const float* A,
    const float* B,
    float* C,
    int warmup_iters,
    int timed_iters);

FASTCUDA_API GemmTiming BenchmarkGemmTF32(
    const GemmConfig& config,
    const float* A,
    const float* B,
    float* C,
    int warmup_iters,
    int timed_iters);

FASTCUDA_API GemmTiming BenchmarkHgemm(
    const GemmConfig& config,
    const void* A,
    const void* B,
    float* C,
    int warmup_iters,
    int timed_iters);

/* Host convenience – manages GPU allocation internally */
FASTCUDA_API GemmTiming RunSgemmHost(
    GemmAlgorithm algo,
    const GemmConfig& config,
    const float* a_host,
    const float* b_host,
    const float* c_input_host,
    float* c_output_host,
    int warmup_iters,
    int timed_iters);

/* CPU reference SGEMM for validation */
FASTCUDA_API void ReferenceSgemm(
    const GemmConfig& config,
    const float* a_host,
    const float* b_host,
    const float* c_input_host,
    float* c_output_host);

FASTCUDA_API float MaxAbsDiff(
    const float* lhs,
    const float* rhs,
    std::size_t count);

}  // namespace fastcuda
