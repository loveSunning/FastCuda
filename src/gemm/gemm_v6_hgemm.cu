/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * GEMM Version 6: FP16 Tensor Core HGEMM
 * - Input:  FP16 (half)
 * - Accumulate: FP32
 * - Output: FP32 (caller supplies float* C)
 * - Uses WMMA intrinsics with half fragments.
 * - WMMA shape: M=16, N=16, K=16 (half).
 * - Block tile: BM=64, BN=64, BK=16.
 * - Requires sm_70+; covers RTX 4090 (sm_89) and RTX 5060 (sm_120).
 */

#include "gemm_internal.h"
#include "common/cuda_check.h"

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

namespace fastcuda {
namespace internal {

static const int BM_V6  = 64;
static const int BN_V6  = 64;
static const int BK_V6  = 16;
static const int WM_V6  = 16;
static const int WN_V6  = 16;
static const int WK_V6  = 16;

__global__ void hgemm_v6_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta)
{
    __shared__ half As[BM_V6][BK_V6 + 8];   /* pad to avoid bank conflict */
    __shared__ half Bs[BK_V6][BN_V6 + 8];
    __shared__ float Cscratch[4][4][WM_V6 * WN_V6];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int total_threads = blockDim.x;      /* 128 */

    const int block_row = blockIdx.y * BM_V6;
    const int block_col = blockIdx.x * BN_V6;

    const int warp_row = warp_id * WM_V6;

    wmma::fragment<wmma::accumulator, WM_V6, WN_V6, WK_V6, float> c_frag[4];
    for (int t = 0; t < 4; ++t)
        wmma::fill_fragment(c_frag[t], 0.0f);

    for (int tk = 0; tk < K; tk += BK_V6) {
        /* Load A tile */
        for (int idx = threadIdx.x; idx < BM_V6 * BK_V6; idx += total_threads) {
            int r = idx / BK_V6;
            int c = idx % BK_V6;
            int gr = block_row + r;
            int gc = tk + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * lda + gc] : __float2half(0.0f);
        }
        /* Load B tile */
        for (int idx = threadIdx.x; idx < BK_V6 * BN_V6; idx += total_threads) {
            int r = idx / BN_V6;
            int c = idx % BN_V6;
            int gr = tk + r;
            int gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * ldb + gc] : __float2half(0.0f);
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WM_V6, WN_V6, WK_V6,
                       half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &As[warp_row][0], BK_V6 + 8);

        for (int col_tile = 0; col_tile < 4; ++col_tile) {
            wmma::fragment<wmma::matrix_b, WM_V6, WN_V6, WK_V6,
                           half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, &Bs[0][col_tile * WN_V6], BN_V6 + 8);
            wmma::mma_sync(c_frag[col_tile], a_frag, b_frag, c_frag[col_tile]);
        }

        __syncthreads();
    }

    /* Store to FP32 output */
    for (int col_tile = 0; col_tile < 4; ++col_tile) {
        int gr = block_row + warp_row;
        int gc = block_col + col_tile * WN_V6;
        if (gr < M && gc < N) {
            if (alpha == 1.0f && beta == 0.0f &&
                gr + WM_V6 <= M && gc + WN_V6 <= N) {
                wmma::store_matrix_sync(&C[gr * ldc + gc], c_frag[col_tile],
                                        ldc, wmma::mem_row_major);
            } else {
                float* scratch = &Cscratch[warp_id][col_tile][0];
                wmma::store_matrix_sync(scratch, c_frag[col_tile],
                                        WN_V6, wmma::mem_row_major);
                __syncwarp();
                for (int idx = lane_id; idx < WM_V6 * WN_V6; idx += 32) {
                    int i = idx / WN_V6;
                    int j = idx % WN_V6;
                    int r = gr + i;
                    int c_j = gc + j;
                    if (r < M && c_j < N) {
                        C[r * ldc + c_j] = alpha * scratch[idx]
                                         + beta * C[r * ldc + c_j];
                    }
                }
                __syncwarp();
            }
        }
    }
}

void launch_hgemm_v6(
    const void* A, const void* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream)
{
    const int threads = 128;   /* 4 warps */
    const dim3 grid(CeilDiv(N, BN_V6), CeilDiv(M, BM_V6));
    hgemm_v6_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const half*>(A),
        static_cast<const half*>(B),
        C, M, N, K, lda, ldb, ldc, alpha, beta);
    CheckCuda(cudaGetLastError(), "hgemm_v6");
}

}  // namespace internal
}  // namespace fastcuda
