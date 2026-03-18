/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * GEMM Version 1: Naive SGEMM
 * - Each thread computes one output element C[row][col].
 * - No shared memory, no tiling.
 * - Main bottleneck: redundant global memory accesses.
 */

#include "gemm_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

__global__ void sgemm_v1_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * lda + i] * B[i * ldb + col];
    }

    C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
}

void launch_sgemm_v1_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid(CeilDiv(N, 16), CeilDiv(M, 16));
    sgemm_v1_naive_kernel<<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    CheckCuda(cudaGetLastError(), "sgemm_v1_naive");
}

}  // namespace internal
}  // namespace fastcuda
