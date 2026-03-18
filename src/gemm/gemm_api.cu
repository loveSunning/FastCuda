/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * GEMM public API – dispatches to versioned kernels and wraps the C ABI.
 */

#include "fastcuda/gemm.hpp"
#include "fastcuda/gemm.h"
#include "gemm_internal.h"
#include "common/cuda_check.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

thread_local std::string g_last_error;

void SetLastError(const std::string& msg) { g_last_error = msg; }

void ValidateConfig(const fastcuda::GemmConfig& c) {
    if (c.m <= 0 || c.n <= 0 || c.k <= 0)
        throw std::invalid_argument("GEMM dimensions must be positive");
    if (c.lda < c.k || c.ldb < c.n || c.ldc < c.n)
        throw std::invalid_argument("Leading dimensions must cover row-major widths");
}

FastCudaStatus HandleCError(const std::exception& ex) {
    SetLastError(ex.what());
    if (dynamic_cast<const std::invalid_argument*>(&ex))
        return FASTCUDA_STATUS_INVALID_VALUE;
    if (dynamic_cast<const std::runtime_error*>(&ex))
        return FASTCUDA_STATUS_CUDA_ERROR;
    return FASTCUDA_STATUS_INTERNAL_ERROR;
}

fastcuda::GemmTiming TimedLaunch(
    void (*launch)(cudaStream_t),
    cudaStream_t stream,
    int warmup, int timed)
{
    using fastcuda::internal::CheckCuda;
    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "eventCreate");
    CheckCuda(cudaEventCreate(&stop),  "eventCreate");

    for (int i = 0; i < warmup; ++i) launch(stream);

    CheckCuda(cudaEventRecord(start, stream), "eventRecord");
    for (int i = 0; i < timed; ++i) launch(stream);
    CheckCuda(cudaEventRecord(stop, stream), "eventRecord");
    CheckCuda(cudaEventSynchronize(stop), "eventSync");

    fastcuda::GemmTiming t;
    CheckCuda(cudaEventElapsedTime(&t.elapsed_ms, start, stop), "elapsedTime");
    t.elapsed_ms /= static_cast<float>(timed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return t;
}

}  // anonymous namespace

/* ================================================================
 * C++ API
 * ================================================================ */
namespace fastcuda {

const char* GemmAlgorithmName(GemmAlgorithm algo) {
    switch (algo) {
        case GemmAlgorithm::kNaiveV1:     return "naive_v1";
        case GemmAlgorithm::kSharedMemV2: return "shared_v2";
        case GemmAlgorithm::kRegisterV3:  return "register_v3";
        case GemmAlgorithm::kWarpV4:      return "warp_v4";
        case GemmAlgorithm::kTF32V5:      return "tf32_v5";
        case GemmAlgorithm::kHgemmV6:     return "hgemm_v6";
        default:                          return "unknown";
    }
}

void LaunchSgemm(GemmAlgorithm algo, const GemmConfig& cfg,
                 const float* A, const float* B, float* C) {
    ValidateConfig(cfg);
    cudaStream_t s = internal::ToCudaStream(cfg.stream);
    switch (algo) {
        case GemmAlgorithm::kNaiveV1:
            internal::launch_sgemm_v1_naive(A, B, C, cfg.m, cfg.n, cfg.k,
                cfg.lda, cfg.ldb, cfg.ldc, cfg.alpha, cfg.beta, s);
            break;
        case GemmAlgorithm::kSharedMemV2:
            internal::launch_sgemm_v2_shared(A, B, C, cfg.m, cfg.n, cfg.k,
                cfg.lda, cfg.ldb, cfg.ldc, cfg.alpha, cfg.beta, s);
            break;
        case GemmAlgorithm::kRegisterV3:
            internal::launch_sgemm_v3_register(A, B, C, cfg.m, cfg.n, cfg.k,
                cfg.lda, cfg.ldb, cfg.ldc, cfg.alpha, cfg.beta, s);
            break;
        case GemmAlgorithm::kWarpV4:
            internal::launch_sgemm_v4_warp(A, B, C, cfg.m, cfg.n, cfg.k,
                cfg.lda, cfg.ldb, cfg.ldc, cfg.alpha, cfg.beta, s);
            break;
        case GemmAlgorithm::kTF32V5:
            internal::launch_gemm_v5_tf32(A, B, C, cfg.m, cfg.n, cfg.k,
                cfg.lda, cfg.ldb, cfg.ldc, cfg.alpha, cfg.beta, s);
            break;
        default:
            throw std::invalid_argument("Use LaunchHgemm() for HGEMM_V6");
    }
}

void LaunchGemmTF32(const GemmConfig& cfg,
                    const float* A, const float* B, float* C) {
    ValidateConfig(cfg);
    internal::launch_gemm_v5_tf32(A, B, C, cfg.m, cfg.n, cfg.k,
        cfg.lda, cfg.ldb, cfg.ldc, cfg.alpha, cfg.beta,
        internal::ToCudaStream(cfg.stream));
}

void LaunchHgemm(const GemmConfig& cfg,
                 const void* A, const void* B, float* C) {
    ValidateConfig(cfg);
    internal::launch_hgemm_v6(A, B, C, cfg.m, cfg.n, cfg.k,
        cfg.lda, cfg.ldb, cfg.ldc, cfg.alpha, cfg.beta,
        internal::ToCudaStream(cfg.stream));
}

/* ---- Benchmark helpers ---- */

GemmTiming BenchmarkSgemm(GemmAlgorithm algo, const GemmConfig& cfg,
                           const float* A, const float* B, float* C,
                           int warmup, int timed) {
    ValidateConfig(cfg);
    struct Ctx {
        GemmAlgorithm algo; const GemmConfig* cfg;
        const float* A; const float* B; float* C;
    } ctx = {algo, &cfg, A, B, C};

    /* We cannot capture in C++11 lambdas passed as function ptrs,
       so use a small loop instead. */
    cudaStream_t stream = internal::ToCudaStream(cfg.stream);
    cudaEvent_t start, stop;
    internal::CheckCuda(cudaEventCreate(&start), "eventCreate");
    internal::CheckCuda(cudaEventCreate(&stop),  "eventCreate");

    for (int i = 0; i < warmup; ++i)
        LaunchSgemm(algo, cfg, A, B, C);

    internal::CheckCuda(cudaEventRecord(start, stream), "eventRecord");
    for (int i = 0; i < timed; ++i)
        LaunchSgemm(algo, cfg, A, B, C);
    internal::CheckCuda(cudaEventRecord(stop, stream), "eventRecord");
    internal::CheckCuda(cudaEventSynchronize(stop), "eventSync");

    GemmTiming t;
    internal::CheckCuda(cudaEventElapsedTime(&t.elapsed_ms, start, stop), "elapsedTime");
    t.elapsed_ms /= static_cast<float>(timed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return t;
}

GemmTiming BenchmarkGemmTF32(const GemmConfig& cfg,
                              const float* A, const float* B, float* C,
                              int warmup, int timed) {
    ValidateConfig(cfg);
    cudaStream_t stream = internal::ToCudaStream(cfg.stream);
    cudaEvent_t start, stop;
    internal::CheckCuda(cudaEventCreate(&start), "eventCreate");
    internal::CheckCuda(cudaEventCreate(&stop),  "eventCreate");

    for (int i = 0; i < warmup; ++i)
        LaunchGemmTF32(cfg, A, B, C);

    internal::CheckCuda(cudaEventRecord(start, stream), "eventRecord");
    for (int i = 0; i < timed; ++i)
        LaunchGemmTF32(cfg, A, B, C);
    internal::CheckCuda(cudaEventRecord(stop, stream), "eventRecord");
    internal::CheckCuda(cudaEventSynchronize(stop), "eventSync");

    GemmTiming t;
    internal::CheckCuda(cudaEventElapsedTime(&t.elapsed_ms, start, stop), "elapsedTime");
    t.elapsed_ms /= static_cast<float>(timed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return t;
}

GemmTiming BenchmarkHgemm(const GemmConfig& cfg,
                           const void* A, const void* B, float* C,
                           int warmup, int timed) {
    ValidateConfig(cfg);
    cudaStream_t stream = internal::ToCudaStream(cfg.stream);
    cudaEvent_t start, stop;
    internal::CheckCuda(cudaEventCreate(&start), "eventCreate");
    internal::CheckCuda(cudaEventCreate(&stop),  "eventCreate");

    for (int i = 0; i < warmup; ++i)
        LaunchHgemm(cfg, A, B, C);

    internal::CheckCuda(cudaEventRecord(start, stream), "eventRecord");
    for (int i = 0; i < timed; ++i)
        LaunchHgemm(cfg, A, B, C);
    internal::CheckCuda(cudaEventRecord(stop, stream), "eventRecord");
    internal::CheckCuda(cudaEventSynchronize(stop), "eventSync");

    GemmTiming t;
    internal::CheckCuda(cudaEventElapsedTime(&t.elapsed_ms, start, stop), "elapsedTime");
    t.elapsed_ms /= static_cast<float>(timed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return t;
}

/* ---- Host convenience ---- */

GemmTiming RunSgemmHost(GemmAlgorithm algo, const GemmConfig& cfg,
                        const float* a_host, const float* b_host,
                        const float* c_in, float* c_out,
                        int warmup, int timed) {
    ValidateConfig(cfg);
    const std::size_t a_bytes = (std::size_t)cfg.m * cfg.lda * sizeof(float);
    const std::size_t b_bytes = (std::size_t)cfg.k * cfg.ldb * sizeof(float);
    const std::size_t c_bytes = (std::size_t)cfg.m * cfg.ldc * sizeof(float);

    float *dA = NULL, *dB = NULL, *dC = NULL;
    internal::CheckCuda(cudaMalloc(&dA, a_bytes), "malloc A");
    internal::CheckCuda(cudaMalloc(&dB, b_bytes), "malloc B");
    internal::CheckCuda(cudaMalloc(&dC, c_bytes), "malloc C");

    internal::CheckCuda(cudaMemcpy(dA, a_host, a_bytes, cudaMemcpyHostToDevice), "H2D A");
    internal::CheckCuda(cudaMemcpy(dB, b_host, b_bytes, cudaMemcpyHostToDevice), "H2D B");
    if (c_in)
        internal::CheckCuda(cudaMemcpy(dC, c_in, c_bytes, cudaMemcpyHostToDevice), "H2D C");
    else
        internal::CheckCuda(cudaMemset(dC, 0, c_bytes), "memset C");

    GemmTiming t = BenchmarkSgemm(algo, cfg, dA, dB, dC, warmup, timed);
    internal::CheckCuda(cudaMemcpy(c_out, dC, c_bytes, cudaMemcpyDeviceToHost), "D2H C");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return t;
}

/* ---- Reference ---- */

void ReferenceSgemm(const GemmConfig& cfg,
                    const float* a, const float* b,
                    const float* c_in, float* c_out) {
    ValidateConfig(cfg);
    for (int r = 0; r < cfg.m; ++r) {
        for (int c = 0; c < cfg.n; ++c) {
            float acc = 0.0f;
            for (int i = 0; i < cfg.k; ++i)
                acc += a[r * cfg.lda + i] * b[i * cfg.ldb + c];
            float cv = c_in ? c_in[r * cfg.ldc + c] : 0.0f;
            c_out[r * cfg.ldc + c] = cfg.alpha * acc + cfg.beta * cv;
        }
    }
}

float MaxAbsDiff(const float* lhs, const float* rhs, std::size_t n) {
    float d = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        d = std::max(d, std::fabs(lhs[i] - rhs[i]));
    return d;
}

}  // namespace fastcuda

/* ================================================================
 * C API
 * ================================================================ */
extern "C" {

const char* fastcuda_get_status_string(FastCudaStatus s) {
    switch (s) {
        case FASTCUDA_STATUS_SUCCESS:        return "success";
        case FASTCUDA_STATUS_INVALID_VALUE:  return "invalid_value";
        case FASTCUDA_STATUS_CUDA_ERROR:     return "cuda_error";
        case FASTCUDA_STATUS_INTERNAL_ERROR: return "internal_error";
        case FASTCUDA_STATUS_NOT_SUPPORTED:  return "not_supported";
        default:                             return "unknown";
    }
}

const char* fastcuda_get_last_error(void) {
    return g_last_error.c_str();
}

const char* fastcuda_gemm_algorithm_name(FastCudaGemmAlgorithm algo) {
    return fastcuda::GemmAlgorithmName(
        static_cast<fastcuda::GemmAlgorithm>(algo));
}

FastCudaStatus fastcuda_sgemm(
    FastCudaGemmAlgorithm algo,
    const FastCudaGemmConfig* cfg,
    const float* A, const float* B, float* C)
{
    if (!cfg) { SetLastError("null config"); return FASTCUDA_STATUS_INVALID_VALUE; }
    try {
        fastcuda::GemmConfig cc;
        cc.m = cfg->m; cc.n = cfg->n; cc.k = cfg->k;
        cc.lda = cfg->lda; cc.ldb = cfg->ldb; cc.ldc = cfg->ldc;
        cc.alpha = cfg->alpha; cc.beta = cfg->beta; cc.stream = cfg->stream;
        fastcuda::LaunchSgemm(static_cast<fastcuda::GemmAlgorithm>(algo), cc, A, B, C);
        return FASTCUDA_STATUS_SUCCESS;
    } catch (const std::exception& e) { return HandleCError(e); }
}

FastCudaStatus fastcuda_gemm_tf32(
    const FastCudaGemmConfig* cfg,
    const float* A, const float* B, float* C)
{
    if (!cfg) { SetLastError("null config"); return FASTCUDA_STATUS_INVALID_VALUE; }
    try {
        fastcuda::GemmConfig cc;
        cc.m = cfg->m; cc.n = cfg->n; cc.k = cfg->k;
        cc.lda = cfg->lda; cc.ldb = cfg->ldb; cc.ldc = cfg->ldc;
        cc.alpha = cfg->alpha; cc.beta = cfg->beta; cc.stream = cfg->stream;
        fastcuda::LaunchGemmTF32(cc, A, B, C);
        return FASTCUDA_STATUS_SUCCESS;
    } catch (const std::exception& e) { return HandleCError(e); }
}

FastCudaStatus fastcuda_hgemm(
    const FastCudaGemmConfig* cfg,
    const void* A, const void* B, float* C)
{
    if (!cfg) { SetLastError("null config"); return FASTCUDA_STATUS_INVALID_VALUE; }
    try {
        fastcuda::GemmConfig cc;
        cc.m = cfg->m; cc.n = cfg->n; cc.k = cfg->k;
        cc.lda = cfg->lda; cc.ldb = cfg->ldb; cc.ldc = cfg->ldc;
        cc.alpha = cfg->alpha; cc.beta = cfg->beta; cc.stream = cfg->stream;
        fastcuda::LaunchHgemm(cc, A, B, C);
        return FASTCUDA_STATUS_SUCCESS;
    } catch (const std::exception& e) { return HandleCError(e); }
}

FastCudaStatus fastcuda_benchmark_sgemm(
    FastCudaGemmAlgorithm algo,
    const FastCudaGemmConfig* cfg,
    const float* A, const float* B, float* C,
    int warmup, int timed, float* elapsed_ms)
{
    if (!cfg || !elapsed_ms) {
        SetLastError("null config or elapsed_ms");
        return FASTCUDA_STATUS_INVALID_VALUE;
    }
    try {
        fastcuda::GemmConfig cc;
        cc.m = cfg->m; cc.n = cfg->n; cc.k = cfg->k;
        cc.lda = cfg->lda; cc.ldb = cfg->ldb; cc.ldc = cfg->ldc;
        cc.alpha = cfg->alpha; cc.beta = cfg->beta; cc.stream = cfg->stream;
        fastcuda::GemmTiming t = fastcuda::BenchmarkSgemm(
            static_cast<fastcuda::GemmAlgorithm>(algo), cc, A, B, C, warmup, timed);
        *elapsed_ms = t.elapsed_ms;
        return FASTCUDA_STATUS_SUCCESS;
    } catch (const std::exception& e) { return HandleCError(e); }
}

FastCudaStatus fastcuda_benchmark_gemm_tf32(
    const FastCudaGemmConfig* cfg,
    const float* A, const float* B, float* C,
    int warmup, int timed, float* elapsed_ms)
{
    if (!cfg || !elapsed_ms) {
        SetLastError("null config or elapsed_ms");
        return FASTCUDA_STATUS_INVALID_VALUE;
    }
    try {
        fastcuda::GemmConfig cc;
        cc.m = cfg->m; cc.n = cfg->n; cc.k = cfg->k;
        cc.lda = cfg->lda; cc.ldb = cfg->ldb; cc.ldc = cfg->ldc;
        cc.alpha = cfg->alpha; cc.beta = cfg->beta; cc.stream = cfg->stream;
        fastcuda::GemmTiming t = fastcuda::BenchmarkGemmTF32(cc, A, B, C, warmup, timed);
        *elapsed_ms = t.elapsed_ms;
        return FASTCUDA_STATUS_SUCCESS;
    } catch (const std::exception& e) { return HandleCError(e); }
}

FastCudaStatus fastcuda_benchmark_hgemm(
    const FastCudaGemmConfig* cfg,
    const void* A, const void* B, float* C,
    int warmup, int timed, float* elapsed_ms)
{
    if (!cfg || !elapsed_ms) {
        SetLastError("null config or elapsed_ms");
        return FASTCUDA_STATUS_INVALID_VALUE;
    }
    try {
        fastcuda::GemmConfig cc;
        cc.m = cfg->m; cc.n = cfg->n; cc.k = cfg->k;
        cc.lda = cfg->lda; cc.ldb = cfg->ldb; cc.ldc = cfg->ldc;
        cc.alpha = cfg->alpha; cc.beta = cfg->beta; cc.stream = cfg->stream;
        fastcuda::GemmTiming t = fastcuda::BenchmarkHgemm(cc, A, B, C, warmup, timed);
        *elapsed_ms = t.elapsed_ms;
        return FASTCUDA_STATUS_SUCCESS;
    } catch (const std::exception& e) { return HandleCError(e); }
}

}  // extern "C"
