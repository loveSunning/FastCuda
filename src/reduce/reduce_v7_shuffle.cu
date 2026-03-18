/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Reduce V7: Warp shuffle reduction
 * - Uses __shfl_down_sync for warp-level reduction – no shared memory needed
 *   for the intra-warp phase.
 * - Cross-warp partial sums are collected in shared memory.
 * - Combined with multi-element load for high throughput.
 */

#include "reduce_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

static const int ELEMS_PER_THREAD_V7 = 8;

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduce_v7_kernel(const float* input, float* output, int n) {
    const int warpSize32 = 32;
    float val = 0.0f;

    unsigned int base = blockIdx.x * (blockDim.x * ELEMS_PER_THREAD_V7) + threadIdx.x;
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD_V7; ++e) {
        unsigned int idx = base + e * blockDim.x;
        if (idx < n) val += input[idx];
    }

    /* Intra-warp reduction via shuffle */
    val = warp_reduce_sum(val);

    /* Collect warp results in shared memory */
    __shared__ float warp_sums[32];   /* max 32 warps per block */
    int lane = threadIdx.x % warpSize32;
    int wid  = threadIdx.x / warpSize32;

    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();

    /* First warp reduces the warp sums */
    int num_warps = (blockDim.x + warpSize32 - 1) / warpSize32;
    val = (threadIdx.x < (unsigned int)num_warps) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0) output[blockIdx.x] = val;
}

void launch_reduce_v7_shuffle(
    const float* input, float* output, int n, cudaStream_t stream)
{
    const int bs = REDUCE_BLOCK_SIZE;   /* 256 */
    int blocks = CeilDiv(n, bs * ELEMS_PER_THREAD_V7);
    reduce_v7_kernel<<<blocks, bs, 0, stream>>>(input, output, n);
    CheckCuda(cudaGetLastError(), "reduce_v7_shuffle");

    if (blocks > 1) {
        launch_reduce_v7_shuffle(output, output, blocks, stream);
    }
}

}  // namespace internal
}  // namespace fastcuda
