/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Reduce V5: Completely unrolled
 * - Template on block size so the compiler unrolls every iteration.
 * - Combined with add-during-load and warp-unroll from V3/V4.
 */

#include "reduce_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

template <unsigned int blockSize>
__device__ void warp_reduce_v5(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce_v5_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockSize * 2) + threadIdx.x;

    float val = 0.0f;
    if (i < n)              val  = input[i];
    if (i + blockSize < n)  val += input[i + blockSize];
    sdata[tid] = val;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }

    if (tid < 32) warp_reduce_v5<blockSize>(sdata, tid);

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void launch_reduce_v5_completely_unroll(
    const float* input, float* output, int n, cudaStream_t stream)
{
    const int bs = REDUCE_BLOCK_SIZE;  /* 256 */
    int blocks = CeilDiv(n, bs * 2);
    reduce_v5_kernel<256><<<blocks, bs, bs * sizeof(float), stream>>>(
        input, output, n);
    CheckCuda(cudaGetLastError(), "reduce_v5_completely_unroll");

    if (blocks > 1) {
        launch_reduce_v5_completely_unroll(output, output, blocks, stream);
    }
}

}  // namespace internal
}  // namespace fastcuda
