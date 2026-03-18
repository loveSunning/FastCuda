/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Reduce V2: No bank conflict (sequential addressing)
 * - Loop counts down from blockDim.x/2, accessing contiguous elements.
 * - Eliminates bank conflicts from V1 stride pattern.
 */

#include "reduce_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

__global__ void reduce_v2_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    /* Sequential addressing – no bank conflicts */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void launch_reduce_v2_no_bank_conflict(
    const float* input, float* output, int n, cudaStream_t stream)
{
    int blocks = CeilDiv(n, REDUCE_BLOCK_SIZE);
    reduce_v2_kernel<<<blocks, REDUCE_BLOCK_SIZE,
                       REDUCE_BLOCK_SIZE * sizeof(float), stream>>>(
        input, output, n);
    CheckCuda(cudaGetLastError(), "reduce_v2_no_bank_conflict");

    if (blocks > 1) {
        launch_reduce_v2_no_bank_conflict(output, output, blocks, stream);
    }
}

}  // namespace internal
}  // namespace fastcuda
