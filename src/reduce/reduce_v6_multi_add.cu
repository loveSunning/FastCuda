/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Reduce V6: Multi-element accumulation
 * - Each thread first accumulates ELEMENTS_PER_THREAD values from global memory.
 * - Increases per-thread work, amortises launch overhead and memory latency.
 * - Then performs the standard block-level reduction.
 */

#include "reduce_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

static const int ELEMS_PER_THREAD_V6 = 8;

template <unsigned int blockSize>
__device__ void warp_reduce_v6(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize, int elemsPerThread>
__global__ void reduce_v6_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int base = blockIdx.x * (blockSize * elemsPerThread) + threadIdx.x;

    float val = 0.0f;
#pragma unroll
    for (int e = 0; e < elemsPerThread; ++e) {
        unsigned int idx = base + e * blockSize;
        if (idx < n) val += input[idx];
    }
    sdata[tid] = val;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }

    if (tid < 32) warp_reduce_v6<blockSize>(sdata, tid);

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void launch_reduce_v6_multi_add(
    const float* input, float* output, int n, cudaStream_t stream)
{
    const int bs  = REDUCE_BLOCK_SIZE;  /* 256 */
    int blocks = CeilDiv(n, bs * ELEMS_PER_THREAD_V6);
    reduce_v6_kernel<256, ELEMS_PER_THREAD_V6><<<blocks, bs,
        bs * sizeof(float), stream>>>(input, output, n);
    CheckCuda(cudaGetLastError(), "reduce_v6_multi_add");

    if (blocks > 1) {
        launch_reduce_v6_multi_add(output, output, blocks, stream);
    }
}

}  // namespace internal
}  // namespace fastcuda
