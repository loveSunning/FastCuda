/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * GEMM Version 4: Warp Cooperative + Prefetch + Double-Buffer SGEMM
 * - Warp-level tiling within each block.
 * - Double-buffered shared memory to overlap global loads with compute.
 * - Software pipelining: load tile k+1 while computing tile k.
 * - High-performance CUDA-core SGEMM.
 *
 * Tile geometry: BM=128, BN=128, BK=8
 * Thread tile:   TM=8,   TN=8
 * Block threads: (128/8) x (128/8) = 16 x 16 = 256
 */

#include "gemm_internal.h"
#include "common/cuda_check.h"

namespace fastcuda {
namespace internal {

static const int BM_V4 = 128;
static const int BN_V4 = 128;
static const int BK_V4 = 8;
static const int TM_V4 = 8;
static const int TN_V4 = 8;

__global__ void sgemm_v4_warp_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta)
{
    /* Double-buffered shared memory */
    __shared__ float As[2][BK_V4][BM_V4];
    __shared__ float Bs[2][BK_V4][BN_V4];

    const int tx = threadIdx.x;          /* 0..15 */
    const int ty = threadIdx.y;          /* 0..15 */
    const int tid = ty * blockDim.x + tx;
    const int threads = blockDim.x * blockDim.y;   /* 256 */

    const int block_row = blockIdx.y * BM_V4;
    const int block_col = blockIdx.x * BN_V4;

    float accum[TM_V4][TN_V4];
#pragma unroll
    for (int i = 0; i < TM_V4; ++i)
        for (int j = 0; j < TN_V4; ++j)
            accum[i][j] = 0.0f;

    /* ---------- helper lambdas replaced with inline loads ---------- */

    /* Load the first tile into buffer 0 */
    int buf = 0;
    for (int idx = tid; idx < BM_V4 * BK_V4; idx += threads) {
        int tr = idx % BM_V4;
        int tc = idx / BM_V4;
        int gr = block_row + tr;
        int gc = tc;
        As[0][tc][tr] = (gr < M && gc < K) ? A[gr * lda + gc] : 0.0f;
    }
    for (int idx = tid; idx < BK_V4 * BN_V4; idx += threads) {
        int tr = idx / BN_V4;
        int tc = idx % BN_V4;
        int gr = tr;
        int gc = block_col + tc;
        Bs[0][tr][tc] = (gr < K && gc < N) ? B[gr * ldb + gc] : 0.0f;
    }
    __syncthreads();

    for (int tk = BK_V4; tk < K + BK_V4; tk += BK_V4) {
        /* ---- prefetch next tile into buffer 1-buf ---- */
        int next = 1 - buf;
        for (int idx = tid; idx < BM_V4 * BK_V4; idx += threads) {
            int tr = idx % BM_V4;
            int tc = idx / BM_V4;
            int gr = block_row + tr;
            int gc = tk + tc;
            As[next][tc][tr] = (gr < M && gc < K) ? A[gr * lda + gc] : 0.0f;
        }
        for (int idx = tid; idx < BK_V4 * BN_V4; idx += threads) {
            int tr = idx / BN_V4;
            int tc = idx % BN_V4;
            int gr = tk + tr;
            int gc = block_col + tc;
            Bs[next][tr][tc] = (gr < K && gc < N) ? B[gr * ldb + gc] : 0.0f;
        }

        /* ---- compute from current buffer ---- */
#pragma unroll
        for (int bk = 0; bk < BK_V4; ++bk) {
            float a_frag[TM_V4];
            float b_frag[TN_V4];
#pragma unroll
            for (int i = 0; i < TM_V4; ++i)
                a_frag[i] = As[buf][bk][ty * TM_V4 + i];
#pragma unroll
            for (int j = 0; j < TN_V4; ++j)
                b_frag[j] = Bs[buf][bk][tx * TN_V4 + j];
#pragma unroll
            for (int i = 0; i < TM_V4; ++i)
#pragma unroll
                for (int j = 0; j < TN_V4; ++j)
                    accum[i][j] += a_frag[i] * b_frag[j];
        }

        buf = next;
        __syncthreads();
    }

    /* ---- write back ---- */
    const int row_base = block_row + ty * TM_V4;
    const int col_base = block_col + tx * TN_V4;
#pragma unroll
    for (int i = 0; i < TM_V4; ++i) {
        int r = row_base + i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < TN_V4; ++j) {
            int c = col_base + j;
            if (c < N) {
                C[r * ldc + c] = alpha * accum[i][j]
                               + beta  * C[r * ldc + c];
            }
        }
    }
}

void launch_sgemm_v4_warp(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream)
{
    const dim3 block(BN_V4 / TN_V4, BM_V4 / TM_V4);   /* 16 x 16 = 256 */
    const dim3 grid(CeilDiv(N, BN_V4), CeilDiv(M, BM_V4));
    sgemm_v4_warp_kernel<<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    CheckCuda(cudaGetLastError(), "sgemm_v4_warp");
}

}  // namespace internal
}  // namespace fastcuda
