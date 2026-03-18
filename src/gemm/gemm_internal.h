/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Internal GEMM kernel declarations – not part of the public API.
 */

#ifndef FASTCUDA_GEMM_INTERNAL_H
#define FASTCUDA_GEMM_INTERNAL_H

#include <cuda_runtime.h>

namespace fastcuda {
namespace internal {

void launch_sgemm_v1_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream);

void launch_sgemm_v2_shared(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream);

void launch_sgemm_v3_register(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream);

void launch_sgemm_v4_warp(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream);

void launch_gemm_v5_tf32(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream);

void launch_hgemm_v6(
    const void* A, const void* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta, cudaStream_t stream);

}  // namespace internal
}  // namespace fastcuda

#endif /* FASTCUDA_GEMM_INTERNAL_H */
