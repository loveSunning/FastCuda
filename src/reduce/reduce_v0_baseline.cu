/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Reduce V0: Baseline shared-memory reduction
 * - Interleaved addressing with divergent branching.
 * - Simple but suffers from warp divergence.
 */

#include "reduce_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

__global__ void reduce_v0_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    /* Interleaved addressing – causes warp divergence */
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void launch_reduce_v0_baseline(
    const float* input, float* output, int n, cudaStream_t stream)
{
    int blocks = CeilDiv(n, REDUCE_BLOCK_SIZE);
    reduce_v0_kernel<<<blocks, REDUCE_BLOCK_SIZE,
                       REDUCE_BLOCK_SIZE * sizeof(float), stream>>>(
        input, output, n);
    CheckCuda(cudaGetLastError(), "reduce_v0_baseline");

    /* Second pass to reduce block results */
    if (blocks > 1) {
        launch_reduce_v0_baseline(output, output, blocks, stream);
    }
}

}  // namespace internal
}  // namespace fastcuda
