/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * GEMM Version 5: TF32 Tensor Core GEMM
 * - Input / output are FP32 arrays.
 * - Internally uses TF32 Tensor Core (19-bit: 1s + 8e + 10m) via WMMA.
 * - Requires sm_80+ (Ampere, Ada, Blackwell); covers RTX 4090 and RTX 5060.
 * - WMMA shape: M=16, N=16, K=8 (TF32).
 * - Block tile: BM=64, BN=64, BK=8.
 */

#include "gemm_internal.h"
#include "common/cuda_check.h"

#include <mma.h>
using namespace nvcuda;

namespace fastcuda {
namespace internal {

static const int BM_V5  = 64;
static const int BN_V5  = 64;
static const int BK_V5  = 8;
static const int WM_V5  = 16;
static const int WN_V5  = 16;
static const int WK_V5  = 8;

/*
 * Block thread layout: 128 threads = 4 warps
 * Each warp owns a 16x16 output tile.
 * The block has 4x4 = 16 such tiles (64x64).
 * We assign 4 warps, each warp iterates over 4 column tiles.
 */
__global__ void gemm_v5_tf32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta)
{
    __shared__ float As[BM_V5][BK_V5 + 1];
    __shared__ float Bs[BK_V5][BN_V5 + 1];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int total_threads = blockDim.x;       /* 128 */

    const int block_row = blockIdx.y * BM_V5;
    const int block_col = blockIdx.x * BN_V5;

    /* Each warp handles a row of 4 wmma tiles (16 x 64).
     * warp 0 => rows [0..15],  warp 1 => rows [16..31], etc. */
    const int warp_row = warp_id * WM_V5;

    /* Declare accumulator fragments: 4 column tiles per warp */
    wmma::fragment<wmma::accumulator, WM_V5, WN_V5, WK_V5, float> c_frag[4];
    for (int t = 0; t < 4; ++t)
        wmma::fill_fragment(c_frag[t], 0.0f);

    for (int tk = 0; tk < K; tk += BK_V5) {
        /* Cooperative load A tile (BM x BK) */
        for (int idx = threadIdx.x; idx < BM_V5 * BK_V5; idx += total_threads) {
            int r = idx / BK_V5;
            int c = idx % BK_V5;
            int gr = block_row + r;
            int gc = tk + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * lda + gc] : 0.0f;
        }

        /* Cooperative load B tile (BK x BN) */
        for (int idx = threadIdx.x; idx < BK_V5 * BN_V5; idx += total_threads) {
            int r = idx / BN_V5;
            int c = idx % BN_V5;
            int gr = tk + r;
            int gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * ldb + gc] : 0.0f;
        }

        __syncthreads();

        /* WMMA mma_sync over 4 column tiles */
        wmma::fragment<wmma::matrix_a, WM_V5, WN_V5, WK_V5,
                       wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &As[warp_row][0], BK_V5 + 1);

        for (int col_tile = 0; col_tile < 4; ++col_tile) {
            wmma::fragment<wmma::matrix_b, WM_V5, WN_V5, WK_V5,
                           wmma::precision::tf32, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, &Bs[0][col_tile * WN_V5], BN_V5 + 1);
            wmma::mma_sync(c_frag[col_tile], a_frag, b_frag, c_frag[col_tile]);
        }

        __syncthreads();
    }

    /* Store accumulated results */
    for (int col_tile = 0; col_tile < 4; ++col_tile) {
        int gr = block_row + warp_row;
        int gc = block_col + col_tile * WN_V5;
        if (gr < M && gc < N) {
            /* Apply alpha/beta scaling */
            if (beta == 0.0f && alpha == 1.0f) {
                wmma::store_matrix_sync(&C[gr * ldc + gc], c_frag[col_tile],
                                        ldc, wmma::mem_row_major);
            } else {
                /* Load existing C, scale, and add */
                float tmp[WM_V5 * WN_V5];
                wmma::store_matrix_sync(tmp, c_frag[col_tile],
                                        WN_V5, wmma::mem_row_major);
                for (int i = 0; i < WM_V5; ++i) {
                    for (int j = 0; j < WN_V5; ++j) {
                        int r = gr + i;
                        int c_col = gc + j;
                        if (r < M && c_col < N) {
                            C[r * ldc + c_col] = alpha * tmp[i * WN_V5 + j]
                                               + beta  * C[r * ldc + c_col];
                        }
                    }
                }
            }
        }
    }
}

void launch_gemm_v5_tf32(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream)
{
    const int threads = 128;   /* 4 warps per block */
    const dim3 grid(CeilDiv(N, BN_V5), CeilDiv(M, BM_V5));
    gemm_v5_tf32_kernel<<<grid, threads, 0, stream>>>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta);
    CheckCuda(cudaGetLastError(), "gemm_v5_tf32");
}

}  // namespace internal
}  // namespace fastcuda
