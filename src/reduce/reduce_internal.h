/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Internal Reduce kernel declarations – not part of the public API.
 */

#ifndef FASTCUDA_REDUCE_INTERNAL_H
#define FASTCUDA_REDUCE_INTERNAL_H

#include <cuda_runtime.h>

namespace fastcuda {
namespace internal {

static const int REDUCE_BLOCK_SIZE = 256;

/* Each version launches a block-level reduction kernel.
 * The dispatch layer handles multi-block → single result. */

void launch_reduce_v0_baseline(
    const float* input, float* output, int n, cudaStream_t stream);

void launch_reduce_v1_no_divergence(
    const float* input, float* output, int n, cudaStream_t stream);

void launch_reduce_v2_no_bank_conflict(
    const float* input, float* output, int n, cudaStream_t stream);

void launch_reduce_v3_add_during_load(
    const float* input, float* output, int n, cudaStream_t stream);

void launch_reduce_v4_unroll_last_warp(
    const float* input, float* output, int n, cudaStream_t stream);

void launch_reduce_v5_completely_unroll(
    const float* input, float* output, int n, cudaStream_t stream);

void launch_reduce_v6_multi_add(
    const float* input, float* output, int n, cudaStream_t stream);

void launch_reduce_v7_shuffle(
    const float* input, float* output, int n, cudaStream_t stream);

}  // namespace internal
}  // namespace fastcuda

#endif /* FASTCUDA_REDUCE_INTERNAL_H */
