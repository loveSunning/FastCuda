/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * GEMM Version 2: Shared Memory Block-Tiled SGEMM
 * - Block tiling with TILE x TILE shared memory tiles.
 * - Padding (+1) to avoid shared memory bank conflicts.
 * - First major speed-up over the naive kernel.
 */

#include "gemm_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

static const int TILE_V2 = 32;

__global__ void sgemm_v2_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta)
{
    /* +1 padding eliminates bank conflicts on 32-wide tiles */
    __shared__ float As[TILE_V2][TILE_V2 + 1];
    __shared__ float Bs[TILE_V2][TILE_V2 + 1];

    const int row = blockIdx.y * TILE_V2 + threadIdx.y;
    const int col = blockIdx.x * TILE_V2 + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < K; t += TILE_V2) {
        int a_col = t + threadIdx.x;
        int b_row = t + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * lda + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * ldb + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE_V2; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

void launch_sgemm_v2_shared(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream)
{
    const dim3 block(TILE_V2, TILE_V2);
    const dim3 grid(CeilDiv(N, TILE_V2), CeilDiv(M, TILE_V2));
    sgemm_v2_shared_kernel<<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    CheckCuda(cudaGetLastError(), "sgemm_v2_shared");
}

}  // namespace internal
}  // namespace fastcuda
