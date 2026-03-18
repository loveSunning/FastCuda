/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Reduce V4: Unroll last warp
 * - When reduction stride <= 32 (one warp), synchronisation is unnecessary.
 * - Last warp is unrolled to avoid __syncthreads overhead.
 */

#include "reduce_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

__device__ void warp_reduce_v4(volatile float* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v4_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (i < n)                val  = input[i];
    if (i + blockDim.x < n)  val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) warp_reduce_v4(sdata, tid);

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void launch_reduce_v4_unroll_last_warp(
    const float* input, float* output, int n, cudaStream_t stream)
{
    int blocks = CeilDiv(n, REDUCE_BLOCK_SIZE * 2);
    reduce_v4_kernel<<<blocks, REDUCE_BLOCK_SIZE,
                       REDUCE_BLOCK_SIZE * sizeof(float), stream>>>(
        input, output, n);
    CheckCuda(cudaGetLastError(), "reduce_v4_unroll_last_warp");

    if (blocks > 1) {
        launch_reduce_v4_unroll_last_warp(output, output, blocks, stream);
    }
}

}  // namespace internal
}  // namespace fastcuda
