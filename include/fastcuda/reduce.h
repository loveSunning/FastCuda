/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 */

#ifndef FASTCUDA_REDUCE_H
#define FASTCUDA_REDUCE_H

#include "fastcuda/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FastCudaReduceAlgorithm {
    FASTCUDA_REDUCE_BASELINE_V0         = 0,
    FASTCUDA_REDUCE_NO_DIVERGENCE_V1    = 1,
    FASTCUDA_REDUCE_NO_BANK_CONFLICT_V2 = 2,
    FASTCUDA_REDUCE_ADD_DURING_LOAD_V3  = 3,
    FASTCUDA_REDUCE_UNROLL_LAST_WARP_V4 = 4,
    FASTCUDA_REDUCE_COMPLETELY_UNROLL_V5= 5,
    FASTCUDA_REDUCE_MULTI_ADD_V6        = 6,
    FASTCUDA_REDUCE_SHUFFLE_V7          = 7
} FastCudaReduceAlgorithm;

typedef struct FastCudaReduceConfig {
    int   n;
    void* stream;
} FastCudaReduceConfig;

FASTCUDA_API const char* fastcuda_reduce_algorithm_name(FastCudaReduceAlgorithm algo);

/* Reduce sum – device pointers.  output must hold at least 1 float. */
FASTCUDA_API FastCudaStatus fastcuda_reduce_sum(
    FastCudaReduceAlgorithm algo,
    const FastCudaReduceConfig* config,
    const float* input,
    float* output);

/* Benchmark wrapper */
FASTCUDA_API FastCudaStatus fastcuda_benchmark_reduce(
    FastCudaReduceAlgorithm algo,
    const FastCudaReduceConfig* config,
    const float* input,
    float* output,
    int warmup_iters,
    int timed_iters,
    float* elapsed_ms);

#ifdef __cplusplus
}
#endif

#endif /* FASTCUDA_REDUCE_H */
