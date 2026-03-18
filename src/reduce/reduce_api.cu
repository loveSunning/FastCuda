/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Reduce public API – dispatches to versioned kernels and wraps the C ABI.
 */

#include "fastcuda/reduce.hpp"
#include "fastcuda/reduce.h"
#include "reduce_internal.h"
#include "common/cuda_check.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

thread_local std::string g_reduce_error;

void SetReduceError(const std::string& msg) { g_reduce_error = msg; }

FastCudaStatus HandleReduceError(const std::exception& e) {
    SetReduceError(e.what());
    if (dynamic_cast<const std::invalid_argument*>(&e))
        return FASTCUDA_STATUS_INVALID_VALUE;
    if (dynamic_cast<const std::runtime_error*>(&e))
        return FASTCUDA_STATUS_CUDA_ERROR;
    return FASTCUDA_STATUS_INTERNAL_ERROR;
}

typedef void (*ReduceLauncher)(const float*, float*, int, cudaStream_t);

ReduceLauncher GetLauncher(fastcuda::ReduceAlgorithm algo) {
    using namespace fastcuda::internal;
    switch (algo) {
        case fastcuda::ReduceAlgorithm::kBaselineV0:        return launch_reduce_v0_baseline;
        case fastcuda::ReduceAlgorithm::kNoDivergenceV1:    return launch_reduce_v1_no_divergence;
        case fastcuda::ReduceAlgorithm::kNoBankConflictV2:  return launch_reduce_v2_no_bank_conflict;
        case fastcuda::ReduceAlgorithm::kAddDuringLoadV3:   return launch_reduce_v3_add_during_load;
        case fastcuda::ReduceAlgorithm::kUnrollLastWarpV4:  return launch_reduce_v4_unroll_last_warp;
        case fastcuda::ReduceAlgorithm::kCompletelyUnrollV5:return launch_reduce_v5_completely_unroll;
        case fastcuda::ReduceAlgorithm::kMultiAddV6:        return launch_reduce_v6_multi_add;
        case fastcuda::ReduceAlgorithm::kShuffleV7:         return launch_reduce_v7_shuffle;
        default: throw std::invalid_argument("Unknown reduce algorithm");
    }
}

}  // anonymous namespace

/* ================================================================
 * C++ API
 * ================================================================ */
namespace fastcuda {

const char* ReduceAlgorithmName(ReduceAlgorithm algo) {
    switch (algo) {
        case ReduceAlgorithm::kBaselineV0:        return "baseline_v0";
        case ReduceAlgorithm::kNoDivergenceV1:    return "no_divergence_v1";
        case ReduceAlgorithm::kNoBankConflictV2:  return "no_bank_conflict_v2";
        case ReduceAlgorithm::kAddDuringLoadV3:   return "add_during_load_v3";
        case ReduceAlgorithm::kUnrollLastWarpV4:  return "unroll_last_warp_v4";
        case ReduceAlgorithm::kCompletelyUnrollV5: return "completely_unroll_v5";
        case ReduceAlgorithm::kMultiAddV6:        return "multi_add_v6";
        case ReduceAlgorithm::kShuffleV7:         return "shuffle_v7";
        default:                                  return "unknown";
    }
}

void LaunchReduceSum(ReduceAlgorithm algo, const ReduceConfig& cfg,
                     const float* input, float* output) {
    if (cfg.n <= 0) throw std::invalid_argument("reduce n must be positive");
    ReduceLauncher fn = GetLauncher(algo);
    fn(input, output, cfg.n, internal::ToCudaStream(cfg.stream));
}

ReduceTiming BenchmarkReduceSum(ReduceAlgorithm algo, const ReduceConfig& cfg,
                                 const float* input, float* output,
                                 int warmup, int timed) {
    if (cfg.n <= 0) throw std::invalid_argument("reduce n must be positive");
    ReduceLauncher fn = GetLauncher(algo);
    cudaStream_t stream = internal::ToCudaStream(cfg.stream);

    for (int i = 0; i < warmup; ++i) fn(input, output, cfg.n, stream);

    cudaEvent_t start, stop;
    internal::CheckCuda(cudaEventCreate(&start), "eventCreate");
    internal::CheckCuda(cudaEventCreate(&stop),  "eventCreate");
    internal::CheckCuda(cudaEventRecord(start, stream), "eventRecord");

    for (int i = 0; i < timed; ++i) fn(input, output, cfg.n, stream);

    internal::CheckCuda(cudaEventRecord(stop, stream), "eventRecord");
    internal::CheckCuda(cudaEventSynchronize(stop), "eventSync");

    ReduceTiming t;
    internal::CheckCuda(cudaEventElapsedTime(&t.elapsed_ms, start, stop), "elapsedTime");
    t.elapsed_ms /= static_cast<float>(timed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return t;
}

ReduceTiming RunReduceSumHost(ReduceAlgorithm algo, const ReduceConfig& cfg,
                               const float* input_host, float* output_host,
                               int warmup, int timed) {
    if (cfg.n <= 0) throw std::invalid_argument("reduce n must be positive");

    std::size_t bytes = static_cast<std::size_t>(cfg.n) * sizeof(float);
    float* d_in  = NULL;
    float* d_out = NULL;
    /* allocate enough for partial block results */
    int max_blocks = (cfg.n + 255) / 256;
    std::size_t out_bytes = static_cast<std::size_t>(max_blocks) * sizeof(float);
    if (out_bytes < sizeof(float)) out_bytes = sizeof(float);

    internal::CheckCuda(cudaMalloc(&d_in,  bytes),     "malloc input");
    internal::CheckCuda(cudaMalloc(&d_out, out_bytes), "malloc output");
    internal::CheckCuda(cudaMemcpy(d_in, input_host, bytes, cudaMemcpyHostToDevice), "H2D");
    internal::CheckCuda(cudaMemset(d_out, 0, out_bytes), "memset");

    ReduceTiming t = BenchmarkReduceSum(algo, cfg, d_in, d_out, warmup, timed);
    internal::CheckCuda(cudaMemcpy(output_host, d_out, sizeof(float),
                                   cudaMemcpyDeviceToHost), "D2H");
    cudaFree(d_in);
    cudaFree(d_out);
    return t;
}

float ReduceSumReference(const float* input, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += static_cast<double>(input[i]);
    return static_cast<float>(sum);
}

}  // namespace fastcuda

/* ================================================================
 * C API
 * ================================================================ */
extern "C" {

const char* fastcuda_reduce_algorithm_name(FastCudaReduceAlgorithm algo) {
    return fastcuda::ReduceAlgorithmName(
        static_cast<fastcuda::ReduceAlgorithm>(algo));
}

FastCudaStatus fastcuda_reduce_sum(
    FastCudaReduceAlgorithm algo,
    const FastCudaReduceConfig* cfg,
    const float* input, float* output)
{
    if (!cfg) { SetReduceError("null config"); return FASTCUDA_STATUS_INVALID_VALUE; }
    try {
        fastcuda::ReduceConfig cc;
        cc.n = cfg->n;
        cc.stream = cfg->stream;
        fastcuda::LaunchReduceSum(
            static_cast<fastcuda::ReduceAlgorithm>(algo), cc, input, output);
        return FASTCUDA_STATUS_SUCCESS;
    } catch (const std::exception& e) { return HandleReduceError(e); }
}

FastCudaStatus fastcuda_benchmark_reduce(
    FastCudaReduceAlgorithm algo,
    const FastCudaReduceConfig* cfg,
    const float* input, float* output,
    int warmup, int timed, float* elapsed_ms)
{
    if (!cfg || !elapsed_ms) {
        SetReduceError("null config or elapsed_ms");
        return FASTCUDA_STATUS_INVALID_VALUE;
    }
    try {
        fastcuda::ReduceConfig cc;
        cc.n = cfg->n;
        cc.stream = cfg->stream;
        fastcuda::ReduceTiming t = fastcuda::BenchmarkReduceSum(
            static_cast<fastcuda::ReduceAlgorithm>(algo), cc, input, output,
            warmup, timed);
        *elapsed_ms = t.elapsed_ms;
        return FASTCUDA_STATUS_SUCCESS;
    } catch (const std::exception& e) { return HandleReduceError(e); }
}

}  // extern "C"
