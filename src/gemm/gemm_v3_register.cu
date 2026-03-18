/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * GEMM Version 3: Register Blocking + Vectorised (float4) SGEMM
 * - Micro-kernel: each thread computes a THREAD_M x THREAD_N sub-tile.
 * - Cooperative tile loads via float4 for 128-bit global transactions.
 * - Larger block tile (BM=64, BN=64, BK=16) for better data reuse.
 */

#include "gemm_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

static const int BM_V3 = 64;
static const int BN_V3 = 64;
static const int BK_V3 = 16;
static const int TM_V3 = 4;
static const int TN_V3 = 4;

__global__ void sgemm_v3_register_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta)
{
    __shared__ float As[BM_V3][BK_V3];
    __shared__ float Bs[BK_V3][BN_V3];

    const int tx = threadIdx.x;   /* 0..15 */
    const int ty = threadIdx.y;   /* 0..15 */
    const int tid = ty * blockDim.x + tx;
    const int threads = blockDim.x * blockDim.y;

    const int block_row = blockIdx.y * BM_V3;
    const int block_col = blockIdx.x * BN_V3;

    float accum[TM_V3][TN_V3];
#pragma unroll
    for (int i = 0; i < TM_V3; ++i)
        for (int j = 0; j < TN_V3; ++j)
            accum[i][j] = 0.0f;

    for (int tk = 0; tk < K; tk += BK_V3) {
        /* ---- cooperative load A tile (BM x BK) via float4 ---- */
        for (int idx = tid; idx < (BM_V3 * BK_V3) / 4; idx += threads) {
            int elem = idx * 4;
            int tr = elem / BK_V3;
            int tc = elem % BK_V3;
            int gr = block_row + tr;
            int gc = tk + tc;
            if (gr < M && gc + 3 < K) {
                float4 v = *reinterpret_cast<const float4*>(&A[gr * lda + gc]);
                As[tr][tc]     = v.x;
                As[tr][tc + 1] = v.y;
                As[tr][tc + 2] = v.z;
                As[tr][tc + 3] = v.w;
            } else {
                for (int d = 0; d < 4; ++d) {
                    int r = block_row + tr;
                    int c = tk + tc + d;
                    As[tr][tc + d] = (r < M && c < K) ? A[r * lda + c] : 0.0f;
                }
            }
        }

        /* ---- cooperative load B tile (BK x BN) via float4 ---- */
        for (int idx = tid; idx < (BK_V3 * BN_V3) / 4; idx += threads) {
            int elem = idx * 4;
            int tr = elem / BN_V3;
            int tc = elem % BN_V3;
            int gr = tk + tr;
            int gc = block_col + tc;
            if (gr < K && gc + 3 < N) {
                float4 v = *reinterpret_cast<const float4*>(&B[gr * ldb + gc]);
                Bs[tr][tc]     = v.x;
                Bs[tr][tc + 1] = v.y;
                Bs[tr][tc + 2] = v.z;
                Bs[tr][tc + 3] = v.w;
            } else {
                for (int d = 0; d < 4; ++d) {
                    int r = tk + tr;
                    int c = block_col + tc + d;
                    Bs[tr][tc + d] = (r < K && c < N) ? B[r * ldb + c] : 0.0f;
                }
            }
        }

        __syncthreads();

        /* ---- micro-kernel: register-blocked outer product ---- */
#pragma unroll
        for (int bk = 0; bk < BK_V3; ++bk) {
            float a_frag[TM_V3];
            float b_frag[TN_V3];

#pragma unroll
            for (int i = 0; i < TM_V3; ++i)
                a_frag[i] = As[ty * TM_V3 + i][bk];
#pragma unroll
            for (int j = 0; j < TN_V3; ++j)
                b_frag[j] = Bs[bk][tx * TN_V3 + j];

#pragma unroll
            for (int i = 0; i < TM_V3; ++i)
#pragma unroll
                for (int j = 0; j < TN_V3; ++j)
                    accum[i][j] += a_frag[i] * b_frag[j];
        }

        __syncthreads();
    }

    /* ---- write back ---- */
    const int row_base = block_row + ty * TM_V3;
    const int col_base = block_col + tx * TN_V3;
#pragma unroll
    for (int i = 0; i < TM_V3; ++i) {
        int r = row_base + i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < TN_V3; ++j) {
            int c = col_base + j;
            if (c < N) {
                C[r * ldc + c] = alpha * accum[i][j]
                               + beta  * C[r * ldc + c];
            }
        }
    }
}

void launch_sgemm_v3_register(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream)
{
    const dim3 block(BN_V3 / TN_V3, BM_V3 / TM_V3);   /* 16 x 16 */
    const dim3 grid(CeilDiv(N, BN_V3), CeilDiv(M, BM_V3));
    sgemm_v3_register_kernel<<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    CheckCuda(cudaGetLastError(), "sgemm_v3_register");
}

}  // namespace internal
}  // namespace fastcuda
