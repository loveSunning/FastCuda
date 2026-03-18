/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Reduce V1: No divergent branching
 * - Strided index replaces modulo to remove warp divergence.
 * - Still has shared memory bank conflict issue.
 */

#include "reduce_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

__global__ void reduce_v1_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    /* Strided index – no divergent branching within a warp */
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        unsigned int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void launch_reduce_v1_no_divergence(
    const float* input, float* output, int n, cudaStream_t stream)
{
    int blocks = CeilDiv(n, REDUCE_BLOCK_SIZE);
    reduce_v1_kernel<<<blocks, REDUCE_BLOCK_SIZE,
                       REDUCE_BLOCK_SIZE * sizeof(float), stream>>>(
        input, output, n);
    CheckCuda(cudaGetLastError(), "reduce_v1_no_divergence");

    if (blocks > 1) {
        launch_reduce_v1_no_divergence(output, output, blocks, stream);
    }
}

}  // namespace internal
}  // namespace fastcuda
