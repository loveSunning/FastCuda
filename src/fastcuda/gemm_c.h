#pragma once

#include "fastcuda/export.hpp"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FastCudaStatus {
    FASTCUDA_STATUS_SUCCESS = 0,
    FASTCUDA_STATUS_INVALID_VALUE = 1,
    FASTCUDA_STATUS_CUDA_ERROR = 2,
    FASTCUDA_STATUS_INTERNAL_ERROR = 3
} FastCudaStatus;

typedef enum FastCudaGemmAlgorithm {
    FASTCUDA_GEMM_NAIVE = 0,
    FASTCUDA_GEMM_TILED = 1,
    FASTCUDA_GEMM_REGISTER_BLOCKED = 2
} FastCudaGemmAlgorithm;

typedef struct FastCudaGemmConfig {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    float alpha;
    float beta;
    void* stream;
} FastCudaGemmConfig;

FASTCUDA_API const char* fastcudaGetStatusString(FastCudaStatus status);
FASTCUDA_API const char* fastcudaGetLastErrorMessage(void);
FASTCUDA_API const char* fastcudaGetGemmAlgorithmName(FastCudaGemmAlgorithm algorithm);
FASTCUDA_API FastCudaStatus fastcudaSgemmDevice(
    FastCudaGemmAlgorithm algorithm,
    const FastCudaGemmConfig* config,
    const float* a_device,
    const float* b_device,
    float* c_device,
    int warmup_iterations,
    int timed_iterations,
    float* elapsed_ms);
FASTCUDA_API FastCudaStatus fastcudaSgemmHost(
    FastCudaGemmAlgorithm algorithm,
    const FastCudaGemmConfig* config,
    const float* a_host,
    const float* b_host,
    const float* c_input_host,
    float* c_output_host,
    int warmup_iterations,
    int timed_iterations,
    float* elapsed_ms);

#ifdef __cplusplus
}
#endif