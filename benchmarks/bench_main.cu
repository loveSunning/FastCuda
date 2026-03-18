/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Benchmark entry point – GEMM (v1-v6) and Reduce (v0-v7) vs cuBLAS.
 *
 * Usage:
 *   fastcuda_bench <operator> <kernel> <shape> [dtype]
 *
 *   operator = gemm | reduce
 *   kernel   = all | naive_v1 | shared_v2 | ... | shuffle_v7
 *   shape    = m=1024,n=1024,k=1024   (gemm)
 *              n=1048576               (reduce)
 *   dtype    = fp32 (default) | fp16
 */

#include "fastcuda/gemm.hpp"
#include "fastcuda/reduce.hpp"
#include "fastcuda/runtime.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

namespace {

const int WARMUP  = 5;
const int TIMED   = 20;

void check(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(m) + ": " + cudaGetErrorString(e));
}

void check_cublas(cublasStatus_t s, const char* m) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream os;
        os << m << ": cublas status " << static_cast<int>(s);
        throw std::runtime_error(os.str());
    }
}

int ParseDim(const std::string& s, const std::string& k, int fb) {
    std::string tok = k + "=";
    std::size_t p = s.find(tok);
    if (p == std::string::npos) return fb;
    std::size_t v = p + tok.size();
    return std::atoi(s.substr(v, s.find_first_of(", ", v) - v).c_str());
}

void FillFloat(std::vector<float>* v, float s) {
    for (std::size_t i = 0; i < v->size(); ++i)
    (*v)[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * s;
}

void ConvertToHalf(const std::vector<float>& src, std::vector<half>* dst) {
    dst->resize(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        (*dst)[i] = __float2half(src[i]);
}

bool ShouldRunKernel(const std::string& requested, const char* name) {
    return requested == "all" || requested == name;
}

/* ------------------------------------------------------------------ */
/*  GEMM benchmark                                                     */
/* ------------------------------------------------------------------ */

void PrintSgemmResult(fastcuda::GemmAlgorithm algo,
                      const fastcuda::GemmConfig& cfg,
                      const std::vector<float>& a,
                      const std::vector<float>& b,
                      const std::vector<float>& c_ref) {
    std::vector<float> c_out(c_ref.size(), 0.0f);
    fastcuda::GemmTiming t = fastcuda::RunSgemmHost(
        algo, cfg, a.data(), b.data(), NULL, c_out.data(), WARMUP, TIMED);

    double gflops = 2.0 * cfg.m * cfg.n * cfg.k / (t.elapsed_ms * 1e-3) / 1e9;
    float err = fastcuda::MaxAbsDiff(c_out.data(), c_ref.data(), c_ref.size());

    std::cout << "algorithm=" << fastcuda::GemmAlgorithmName(algo) << "\n"
              << std::fixed << std::setprecision(4)
              << "elapsed_ms=" << t.elapsed_ms << "\n"
              << std::fixed << std::setprecision(2)
              << "gflops=" << gflops << "\n"
              << std::scientific << std::setprecision(6)
              << "max_abs_error=" << err << "\n";
}

void BenchmarkCublasSgemm(const fastcuda::GemmConfig& cfg,
                          const std::vector<float>& a,
                          const std::vector<float>& b,
                          const std::vector<float>& c_ref) {
    std::size_t a_bytes = (std::size_t)cfg.m * cfg.lda * sizeof(float);
    std::size_t b_bytes = (std::size_t)cfg.k * cfg.ldb * sizeof(float);
    std::size_t c_bytes = (std::size_t)cfg.m * cfg.ldc * sizeof(float);

    float *dA, *dB, *dC;
    check(cudaMalloc(&dA, a_bytes), "malloc A");
    check(cudaMalloc(&dB, b_bytes), "malloc B");
    check(cudaMalloc(&dC, c_bytes), "malloc C");

    check(cudaMemcpy(dA, a.data(), a_bytes, cudaMemcpyHostToDevice), "H2D A");
    check(cudaMemcpy(dB, b.data(), b_bytes, cudaMemcpyHostToDevice), "H2D B");
    check(cudaMemset(dC, 0, c_bytes), "memset C");

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");

    float alpha = cfg.alpha, beta = cfg.beta;

    /* Warmup */
    for (int i = 0; i < WARMUP; ++i) {
        /* cuBLAS is column-major, so we compute C^T = B^T * A^T
         * which is equivalent to row-major C = A * B. */
        check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     cfg.n, cfg.m, cfg.k,
                     &alpha, dB, cfg.ldb, dA, cfg.lda,
                     &beta, dC, cfg.ldc),
                 "cublasSgemm");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < TIMED; ++i) {
        check(cudaMemset(dC, 0, c_bytes), "memset");
        check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     cfg.n, cfg.m, cfg.k,
                     &alpha, dB, cfg.ldb, dA, cfg.lda,
                     &beta, dC, cfg.ldc),
                 "cublasSgemm");
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= static_cast<float>(TIMED);

    std::vector<float> c_out(c_ref.size());
    check(cudaMemcpy(c_out.data(), dC, c_bytes, cudaMemcpyDeviceToHost), "D2H");

    double gflops = 2.0 * cfg.m * cfg.n * cfg.k / (ms * 1e-3) / 1e9;
    float err = fastcuda::MaxAbsDiff(c_out.data(), c_ref.data(), c_ref.size());

    std::cout << "algorithm=cublas_sgemm\n"
              << std::fixed << std::setprecision(4)
              << "elapsed_ms=" << ms << "\n"
              << std::fixed << std::setprecision(2)
              << "gflops=" << gflops << "\n"
              << std::scientific << std::setprecision(6)
              << "max_abs_error=" << err << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    check_cublas(cublasDestroy(handle), "cublasDestroy");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

void PrintHgemmResult(const fastcuda::GemmConfig& cfg,
                      const std::vector<half>& a,
                      const std::vector<half>& b,
                      const std::vector<float>& c_ref) {
    std::vector<float> c_out(c_ref.size(), 0.0f);
    fastcuda::GemmTiming t = fastcuda::RunHgemmHost(
        cfg, a.data(), b.data(), NULL, c_out.data(), WARMUP, TIMED);

    double gflops = 2.0 * static_cast<double>(cfg.m) * cfg.n * cfg.k
                  / (t.elapsed_ms * 1e-3) / 1e9;
    float err = fastcuda::MaxAbsDiff(c_out.data(), c_ref.data(), c_ref.size());

    std::cout << "algorithm=hgemm_v6\n"
              << std::fixed << std::setprecision(4)
              << "elapsed_ms=" << t.elapsed_ms << "\n"
              << std::fixed << std::setprecision(2)
              << "gflops=" << gflops << "\n"
              << std::scientific << std::setprecision(6)
              << "max_abs_error=" << err << "\n";
}

void BenchmarkCublasHgemm(const fastcuda::GemmConfig& cfg,
                          const std::vector<half>& a,
                          const std::vector<half>& b,
                          const std::vector<float>& c_ref) {
    std::size_t a_bytes = (std::size_t)cfg.m * cfg.lda * sizeof(half);
    std::size_t b_bytes = (std::size_t)cfg.k * cfg.ldb * sizeof(half);
    std::size_t c_bytes = (std::size_t)cfg.m * cfg.ldc * sizeof(float);

    half *dA = NULL, *dB = NULL;
    float* dC = NULL;
    check(cudaMalloc(&dA, a_bytes), "malloc A");
    check(cudaMalloc(&dB, b_bytes), "malloc B");
    check(cudaMalloc(&dC, c_bytes), "malloc C");

    check(cudaMemcpy(dA, a.data(), a_bytes, cudaMemcpyHostToDevice), "H2D A");
    check(cudaMemcpy(dB, b.data(), b_bytes, cudaMemcpyHostToDevice), "H2D B");
    check(cudaMemset(dC, 0, c_bytes), "memset C");

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");

    float alpha = cfg.alpha;
    float beta = cfg.beta;

    for (int i = 0; i < WARMUP; ++i) {
        check_cublas(cublasGemmEx(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  cfg.n, cfg.m, cfg.k,
                                  &alpha,
                                  dB, CUDA_R_16F, cfg.ldb,
                                  dA, CUDA_R_16F, cfg.lda,
                                  &beta,
                                  dC, CUDA_R_32F, cfg.ldc,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                     "cublasGemmEx warmup");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < TIMED; ++i) {
        check(cudaMemset(dC, 0, c_bytes), "memset C");
        check_cublas(cublasGemmEx(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  cfg.n, cfg.m, cfg.k,
                                  &alpha,
                                  dB, CUDA_R_16F, cfg.ldb,
                                  dA, CUDA_R_16F, cfg.lda,
                                  &beta,
                                  dC, CUDA_R_32F, cfg.ldc,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                     "cublasGemmEx");
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= static_cast<float>(TIMED);

    std::vector<float> c_out(c_ref.size(), 0.0f);
    check(cudaMemcpy(c_out.data(), dC, c_bytes, cudaMemcpyDeviceToHost), "D2H C");

    double gflops = 2.0 * static_cast<double>(cfg.m) * cfg.n * cfg.k / (ms * 1e-3) / 1e9;
    float err = fastcuda::MaxAbsDiff(c_out.data(), c_ref.data(), c_ref.size());

    std::cout << "algorithm=cublas_hgemm\n"
              << std::fixed << std::setprecision(4)
              << "elapsed_ms=" << ms << "\n"
              << std::fixed << std::setprecision(2)
              << "gflops=" << gflops << "\n"
              << std::scientific << std::setprecision(6)
              << "max_abs_error=" << err << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    check_cublas(cublasDestroy(handle), "cublasDestroy");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void RunGemmBenchmark(const std::string& kernel,
                      const std::string& shape,
                      const std::string& dtype) {
    int m = ParseDim(shape, "m", 1024);
    int n = ParseDim(shape, "n", 1024);
    int k = ParseDim(shape, "k", 1024);

    fastcuda::GemmConfig cfg;
    cfg.m = m; cfg.n = n; cfg.k = k;
    cfg.lda = k; cfg.ldb = n; cfg.ldc = n;

    std::cout << "shape=m=" << m << ",n=" << n << ",k=" << k << "\n";
    std::cout << "dtype=" << dtype << "\n";
    std::cout << fastcuda::FormatDeviceSummary(fastcuda::QueryDevices());

    std::vector<float> a_fp32((std::size_t)m * k);
    std::vector<float> b_fp32((std::size_t)k * n);
    FillFloat(&a_fp32, 0.125f);
    FillFloat(&b_fp32, 0.25f);

    std::vector<float> c_ref((std::size_t)m * n, 0.0f);
    fastcuda::ReferenceSgemm(cfg, a_fp32.data(), b_fp32.data(), NULL, c_ref.data());

    if (dtype == "fp32") {
        if (ShouldRunKernel(kernel, "naive_v1"))
            PrintSgemmResult(fastcuda::GemmAlgorithm::kNaiveV1, cfg, a_fp32, b_fp32, c_ref);
        if (ShouldRunKernel(kernel, "shared_v2"))
            PrintSgemmResult(fastcuda::GemmAlgorithm::kSharedMemV2, cfg, a_fp32, b_fp32, c_ref);
        if (ShouldRunKernel(kernel, "register_v3"))
            PrintSgemmResult(fastcuda::GemmAlgorithm::kRegisterV3, cfg, a_fp32, b_fp32, c_ref);
        if (ShouldRunKernel(kernel, "warp_v4"))
            PrintSgemmResult(fastcuda::GemmAlgorithm::kWarpV4, cfg, a_fp32, b_fp32, c_ref);
        if (ShouldRunKernel(kernel, "tf32_v5"))
            PrintSgemmResult(fastcuda::GemmAlgorithm::kTF32V5, cfg, a_fp32, b_fp32, c_ref);
        if (ShouldRunKernel(kernel, "cublas_sgemm"))
            BenchmarkCublasSgemm(cfg, a_fp32, b_fp32, c_ref);
    } else if (dtype == "fp16") {
        std::vector<half> a_fp16;
        std::vector<half> b_fp16;
        ConvertToHalf(a_fp32, &a_fp16);
        ConvertToHalf(b_fp32, &b_fp16);

        if (ShouldRunKernel(kernel, "hgemm_v6"))
            PrintHgemmResult(cfg, a_fp16, b_fp16, c_ref);
        if (ShouldRunKernel(kernel, "cublas_hgemm"))
            BenchmarkCublasHgemm(cfg, a_fp16, b_fp16, c_ref);
    } else {
        throw std::invalid_argument("gemm dtype must be fp32 or fp16");
    }

    std::cout << "status=ok\n";
}

/* ------------------------------------------------------------------ */
/*  Reduce benchmark                                                   */
/* ------------------------------------------------------------------ */

void PrintReduceResult(fastcuda::ReduceAlgorithm algo,
                       const fastcuda::ReduceConfig& cfg,
                       const std::vector<float>& input,
                       float ref) {
    float gpu_sum = 0.0f;
    fastcuda::ReduceTiming t = fastcuda::RunReduceSumHost(
        algo, cfg, input.data(), &gpu_sum, WARMUP, TIMED);

    double bw = static_cast<double>(cfg.n) * sizeof(float)
                / (t.elapsed_ms * 1e-3) / 1e9;
    float err = std::fabs(gpu_sum - ref);

    std::cout << "algorithm=" << fastcuda::ReduceAlgorithmName(algo) << "\n"
              << std::fixed << std::setprecision(4)
              << "elapsed_ms=" << t.elapsed_ms << "\n"
              << std::fixed << std::setprecision(2)
              << "bandwidth_gb_s=" << bw << "\n"
              << std::scientific << std::setprecision(6)
              << "abs_error=" << err << "\n";
}

void RunReduceBenchmark(const std::string& kernel, const std::string& shape) {
    int n = ParseDim(shape, "n", 1 << 20);

    std::cout << "n=" << n << "\n";
    std::cout << fastcuda::FormatDeviceSummary(fastcuda::QueryDevices());

    std::vector<float> input((std::size_t)n);
    FillFloat(&input, 0.1f);

    float ref = fastcuda::ReduceSumReference(input.data(), n);
    std::cout << "reference_sum=" << ref << "\n";

    fastcuda::ReduceConfig cfg;
    cfg.n = n;

    if (ShouldRunKernel(kernel, "baseline_v0"))
        PrintReduceResult(fastcuda::ReduceAlgorithm::kBaselineV0, cfg, input, ref);
    if (ShouldRunKernel(kernel, "no_divergence_v1"))
        PrintReduceResult(fastcuda::ReduceAlgorithm::kNoDivergenceV1, cfg, input, ref);
    if (ShouldRunKernel(kernel, "no_bank_conflict_v2"))
        PrintReduceResult(fastcuda::ReduceAlgorithm::kNoBankConflictV2, cfg, input, ref);
    if (ShouldRunKernel(kernel, "add_during_load_v3"))
        PrintReduceResult(fastcuda::ReduceAlgorithm::kAddDuringLoadV3, cfg, input, ref);
    if (ShouldRunKernel(kernel, "unroll_last_warp_v4"))
        PrintReduceResult(fastcuda::ReduceAlgorithm::kUnrollLastWarpV4, cfg, input, ref);
    if (ShouldRunKernel(kernel, "completely_unroll_v5"))
        PrintReduceResult(fastcuda::ReduceAlgorithm::kCompletelyUnrollV5, cfg, input, ref);
    if (ShouldRunKernel(kernel, "multi_add_v6"))
        PrintReduceResult(fastcuda::ReduceAlgorithm::kMultiAddV6, cfg, input, ref);
    if (ShouldRunKernel(kernel, "shuffle_v7"))
        PrintReduceResult(fastcuda::ReduceAlgorithm::kShuffleV7, cfg, input, ref);

    std::cout << "status=ok\n";
}

}  // anonymous namespace

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    std::string op    = (argc > 1) ? argv[1] : "gemm";
    std::string kern  = (argc > 2) ? argv[2] : "all";
    std::string shape = (argc > 3) ? argv[3] : "m=1024,n=1024,k=1024";
    std::string dtype = (argc > 4) ? argv[4] : "fp32";

    try {
        std::cout << "operator=" << op << "\n"
                  << "kernel=" << kern << "\n";

        if (op == "gemm") {
            RunGemmBenchmark(kern, shape, dtype);
        } else if (op == "reduce") {
            RunReduceBenchmark(kern, shape);
        } else {
            std::cout << "status=unsupported_operator\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error=" << e.what() << "\n";
        return 1;
    }
}
