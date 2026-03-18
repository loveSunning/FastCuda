/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Reduce V3: Add during load
 * - Each thread first loads two elements and adds them.
 * - Halves the number of blocks needed.
 * - Reduces idle threads during the first iteration.
 */

#include "reduce_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

__global__ void reduce_v3_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (i < n)                  val  = input[i];
    if (i + blockDim.x < n)    val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void launch_reduce_v3_add_during_load(
    const float* input, float* output, int n, cudaStream_t stream)
{
    int blocks = CeilDiv(n, REDUCE_BLOCK_SIZE * 2);
    reduce_v3_kernel<<<blocks, REDUCE_BLOCK_SIZE,
                       REDUCE_BLOCK_SIZE * sizeof(float), stream>>>(
        input, output, n);
    CheckCuda(cudaGetLastError(), "reduce_v3_add_during_load");

    if (blocks > 1) {
        launch_reduce_v3_add_during_load(output, output, blocks, stream);
    }
}

}  // namespace internal
}  // namespace fastcuda
