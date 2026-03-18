/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 */

#ifndef FASTCUDA_GEMM_H
#define FASTCUDA_GEMM_H

#include "fastcuda/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FastCudaGemmAlgorithm {
    FASTCUDA_GEMM_NAIVE_V1     = 0,
    FASTCUDA_GEMM_SHARED_V2    = 1,
    FASTCUDA_GEMM_REGISTER_V3  = 2,
    FASTCUDA_GEMM_WARP_V4      = 3,
    FASTCUDA_GEMM_TF32_V5      = 4,
    FASTCUDA_GEMM_HGEMM_V6     = 5
} FastCudaGemmAlgorithm;

typedef struct FastCudaGemmConfig {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    float alpha;
    float beta;
    void* stream;
} FastCudaGemmConfig;

FASTCUDA_API const char* fastcuda_gemm_algorithm_name(FastCudaGemmAlgorithm algo);

/* SGEMM: v1-v4 (FP32 CUDA-core paths) */
FASTCUDA_API FastCudaStatus fastcuda_sgemm(
    FastCudaGemmAlgorithm algo,
    const FastCudaGemmConfig* config,
    const float* A,
    const float* B,
    float* C);

/* TF32 GEMM: v5 (FP32 input, TF32 Tensor Core compute, FP32 output) */
FASTCUDA_API FastCudaStatus fastcuda_gemm_tf32(
    const FastCudaGemmConfig* config,
    const float* A,
    const float* B,
    float* C);

/* HGEMM: v6 (FP16 input, FP32 accumulate, FP32 output) */
FASTCUDA_API FastCudaStatus fastcuda_hgemm(
    const FastCudaGemmConfig* config,
    const void* A,
    const void* B,
    float* C);

/* Benchmark wrappers – return elapsed_ms per iteration */
FASTCUDA_API FastCudaStatus fastcuda_benchmark_sgemm(
    FastCudaGemmAlgorithm algo,
    const FastCudaGemmConfig* config,
    const float* A,
    const float* B,
    float* C,
    int warmup_iters,
    int timed_iters,
    float* elapsed_ms);

FASTCUDA_API FastCudaStatus fastcuda_benchmark_gemm_tf32(
    const FastCudaGemmConfig* config,
    const float* A,
    const float* B,
    float* C,
    int warmup_iters,
    int timed_iters,
    float* elapsed_ms);

FASTCUDA_API FastCudaStatus fastcuda_benchmark_hgemm(
    const FastCudaGemmConfig* config,
    const void* A,
    const void* B,
    float* C,
    int warmup_iters,
    int timed_iters,
    float* elapsed_ms);

#ifdef __cplusplus
}
#endif

#endif /* FASTCUDA_GEMM_H */
