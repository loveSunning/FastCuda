#include "fastcuda/gemm.hpp"
#include "fastcuda/gemm_c.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fastcuda {

namespace detail {

thread_local std::string g_last_error_message;

void SetLastErrorMessage(const std::string& message) {
    g_last_error_message = message;
}

void CheckCuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

cudaStream_t ToCudaStream(void* stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

void ValidateConfig(const GemmConfig& config) {
    if (config.m <= 0 || config.n <= 0 || config.k <= 0) {
        throw std::invalid_argument("GEMM dimensions must be positive");
    }
    if (config.lda < config.k || config.ldb < config.n || config.ldc < config.n) {
        throw std::invalid_argument("Leading dimensions must cover row-major matrix widths");
    }
}

GemmAlgorithm ToCppAlgorithm(FastCudaGemmAlgorithm algorithm) {
    switch (algorithm) {
        case FASTCUDA_GEMM_NAIVE:
            return GemmAlgorithm::kNaive;
        case FASTCUDA_GEMM_TILED:
            return GemmAlgorithm::kTiled;
        case FASTCUDA_GEMM_REGISTER_BLOCKED:
            return GemmAlgorithm::kRegisterBlocked;
        default:
            throw std::invalid_argument("Unknown GEMM algorithm");
    }
}

GemmConfig ToCppConfig(const FastCudaGemmConfig& config) {
    GemmConfig cpp_config;
    cpp_config.m = config.m;
    cpp_config.n = config.n;
    cpp_config.k = config.k;
    cpp_config.lda = config.lda;
    cpp_config.ldb = config.ldb;
    cpp_config.ldc = config.ldc;
    cpp_config.alpha = config.alpha;
    cpp_config.beta = config.beta;
    cpp_config.stream = config.stream;
    return cpp_config;
}

template <typename T>
T CeilDiv(T value, T divisor) {
    return (value + divisor - 1) / divisor;
}

__global__ void NaiveSgemmKernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    float alpha,
    float beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) {
        return;
    }

    float accum = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        accum += a[row * lda + inner] * b[inner * ldb + col];
    }

    c[row * ldc + col] = alpha * accum + beta * c[row * ldc + col];
}

template <int TILE>
__global__ void TiledSgemmKernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    float alpha,
    float beta) {
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float accum = 0.0f;
    for (int tile_k = 0; tile_k < k; tile_k += TILE) {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] = (row < m && a_col < k)
            ? a[row * lda + a_col]
            : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < k && col < n)
            ? b[b_row * ldb + col]
            : 0.0f;

        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < TILE; ++inner) {
            accum += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * ldc + col] = alpha * accum + beta * c[row * ldc + col];
    }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void RegisterBlockedSgemmKernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    float alpha,
    float beta) {
    __shared__ float a_tile[BLOCK_M][BLOCK_K];
    __shared__ float b_tile[BLOCK_K][BLOCK_N];

    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;
    const int linear_tid = thread_y * blockDim.x + thread_x;
    const int threads_per_block = blockDim.x * blockDim.y;

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;
    const int row_base = block_row + thread_y * THREAD_M;
    const int col_base = block_col + thread_x * THREAD_N;

    float accum[THREAD_M][THREAD_N] = {0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += BLOCK_K) {
        for (int index = linear_tid; index < BLOCK_M * BLOCK_K; index += threads_per_block) {
            const int tile_row = index / BLOCK_K;
            const int tile_col = index % BLOCK_K;
            const int global_row = block_row + tile_row;
            const int global_col = tile_k + tile_col;
            a_tile[tile_row][tile_col] = (global_row < m && global_col < k)
                ? a[global_row * lda + global_col]
                : 0.0f;
        }

        for (int index = linear_tid; index < BLOCK_K * BLOCK_N; index += threads_per_block) {
            const int tile_row = index / BLOCK_N;
            const int tile_col = index % BLOCK_N;
            const int global_row = tile_k + tile_row;
            const int global_col = block_col + tile_col;
            b_tile[tile_row][tile_col] = (global_row < k && global_col < n)
                ? b[global_row * ldb + global_col]
                : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < BLOCK_K; ++inner) {
            float a_fragment[THREAD_M];
            float b_fragment[THREAD_N];

#pragma unroll
            for (int row = 0; row < THREAD_M; ++row) {
                a_fragment[row] = a_tile[thread_y * THREAD_M + row][inner];
            }

#pragma unroll
            for (int col = 0; col < THREAD_N; ++col) {
                b_fragment[col] = b_tile[inner][thread_x * THREAD_N + col];
            }

#pragma unroll
            for (int row = 0; row < THREAD_M; ++row) {
#pragma unroll
                for (int col = 0; col < THREAD_N; ++col) {
                    accum[row][col] += a_fragment[row] * b_fragment[col];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int row = 0; row < THREAD_M; ++row) {
        const int global_row = row_base + row;
        if (global_row >= m) {
            continue;
        }

#pragma unroll
        for (int col = 0; col < THREAD_N; ++col) {
            const int global_col = col_base + col;
            if (global_col < n) {
                c[global_row * ldc + global_col] =
                    alpha * accum[row][col] + beta * c[global_row * ldc + global_col];
            }
        }
    }
}

void LaunchKernel(
    GemmAlgorithm algorithm,
    const GemmConfig& config,
    const float* a_device,
    const float* b_device,
    float* c_device) {
    const cudaStream_t stream = ToCudaStream(config.stream);

    switch (algorithm) {
        case GemmAlgorithm::kNaive: {
            const dim3 block(16, 16);
            const dim3 grid(
                CeilDiv(config.n, static_cast<int>(block.x)),
                CeilDiv(config.m, static_cast<int>(block.y)));
            NaiveSgemmKernel<<<grid, block, 0, stream>>>(
                a_device,
                b_device,
                c_device,
                config.m,
                config.n,
                config.k,
                config.lda,
                config.ldb,
                config.ldc,
                config.alpha,
                config.beta);
            break;
        }
        case GemmAlgorithm::kTiled: {
            const dim3 block(16, 16);
            const dim3 grid(
                CeilDiv(config.n, 16),
                CeilDiv(config.m, 16));
            TiledSgemmKernel<16><<<grid, block, 0, stream>>>(
                a_device,
                b_device,
                c_device,
                config.m,
                config.n,
                config.k,
                config.lda,
                config.ldb,
                config.ldc,
                config.alpha,
                config.beta);
            break;
        }
        case GemmAlgorithm::kRegisterBlocked: {
            const dim3 block(16, 16);
            const dim3 grid(
                CeilDiv(config.n, 64),
                CeilDiv(config.m, 64));
            RegisterBlockedSgemmKernel<64, 64, 16, 4, 4><<<grid, block, 0, stream>>>(
                a_device,
                b_device,
                c_device,
                config.m,
                config.n,
                config.k,
                config.lda,
                config.ldb,
                config.ldc,
                config.alpha,
                config.beta);
            break;
        }
        default:
            throw std::invalid_argument("Unknown GEMM algorithm");
    }

    CheckCuda(cudaGetLastError(), "SGEMM kernel launch");
}

FastCudaStatus HandleCError(const std::exception& ex) {
    SetLastErrorMessage(ex.what());
    if (dynamic_cast<const std::invalid_argument*>(&ex) != NULL) {
        return FASTCUDA_STATUS_INVALID_VALUE;
    }
    if (dynamic_cast<const std::runtime_error*>(&ex) != NULL) {
        return FASTCUDA_STATUS_CUDA_ERROR;
    }
    return FASTCUDA_STATUS_INTERNAL_ERROR;
}

}  // namespace detail

const char* GemmAlgorithmName(GemmAlgorithm algorithm) {
    switch (algorithm) {
        case GemmAlgorithm::kNaive:
            return "naive";
        case GemmAlgorithm::kTiled:
            return "tiled_shared";
        case GemmAlgorithm::kRegisterBlocked:
            return "register_blocked";
        default:
            return "unknown";
    }
}

void LaunchSgemmDevice(
    GemmAlgorithm algorithm,
    const GemmConfig& config,
    const float* a_device,
    const float* b_device,
    float* c_device) {
    detail::ValidateConfig(config);
    if (a_device == NULL || b_device == NULL || c_device == NULL) {
        throw std::invalid_argument("Device pointers must be non-null");
    }

    detail::LaunchKernel(algorithm, config, a_device, b_device, c_device);
}

GemmTiming BenchmarkSgemmDevice(
    GemmAlgorithm algorithm,
    const GemmConfig& config,
    const float* a_device,
    const float* b_device,
    float* c_device,
    int warmup_iterations,
    int timed_iterations) {
    detail::ValidateConfig(config);
    if (a_device == NULL || b_device == NULL || c_device == NULL) {
        throw std::invalid_argument("Device pointers must be non-null");
    }
    if (warmup_iterations < 0 || timed_iterations <= 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0 and timed_iterations must be > 0");
    }

    const cudaStream_t stream = detail::ToCudaStream(config.stream);
    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;
    GemmTiming timing;

    detail::CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    detail::CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    try {
        for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
            detail::LaunchKernel(algorithm, config, a_device, b_device, c_device);
        }

        detail::CheckCuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
        for (int iteration = 0; iteration < timed_iterations; ++iteration) {
            detail::LaunchKernel(algorithm, config, a_device, b_device, c_device);
        }
        detail::CheckCuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
        detail::CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        detail::CheckCuda(cudaEventElapsedTime(&timing.elapsed_ms, start, stop), "cudaEventElapsedTime");
        timing.elapsed_ms /= static_cast<float>(timed_iterations);
    } catch (...) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        throw;
    }

    detail::CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    detail::CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return timing;
}

GemmTiming RunSgemmHost(
    GemmAlgorithm algorithm,
    const GemmConfig& config,
    const float* a_host,
    const float* b_host,
    const float* c_input_host,
    float* c_output_host,
    int warmup_iterations,
    int timed_iterations) {
    detail::ValidateConfig(config);
    if (a_host == NULL || b_host == NULL || c_output_host == NULL) {
        throw std::invalid_argument("Host pointers for A, B and output C must be non-null");
    }

    const std::size_t a_bytes = static_cast<std::size_t>(config.m) * config.lda * sizeof(float);
    const std::size_t b_bytes = static_cast<std::size_t>(config.k) * config.ldb * sizeof(float);
    const std::size_t c_bytes = static_cast<std::size_t>(config.m) * config.ldc * sizeof(float);

    float* a_device = NULL;
    float* b_device = NULL;
    float* c_device = NULL;

    detail::CheckCuda(cudaMalloc(&a_device, a_bytes), "cudaMalloc(A)");
    detail::CheckCuda(cudaMalloc(&b_device, b_bytes), "cudaMalloc(B)");
    detail::CheckCuda(cudaMalloc(&c_device, c_bytes), "cudaMalloc(C)");

    try {
        detail::CheckCuda(cudaMemcpy(a_device, a_host, a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(A)");
        detail::CheckCuda(cudaMemcpy(b_device, b_host, b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(B)");

        if (c_input_host != NULL) {
            detail::CheckCuda(cudaMemcpy(c_device, c_input_host, c_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(C input)");
        } else {
            detail::CheckCuda(cudaMemset(c_device, 0, c_bytes), "cudaMemset(C)");
        }

        GemmTiming timing = BenchmarkSgemmDevice(
            algorithm,
            config,
            a_device,
            b_device,
            c_device,
            warmup_iterations,
            timed_iterations);

        detail::CheckCuda(cudaMemcpy(c_output_host, c_device, c_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(C output)");

        cudaFree(a_device);
        cudaFree(b_device);
        cudaFree(c_device);
        return timing;
    } catch (...) {
        cudaFree(a_device);
        cudaFree(b_device);
        cudaFree(c_device);
        throw;
    }
}

void ReferenceSgemm(
    const GemmConfig& config,
    const float* a_host,
    const float* b_host,
    const float* c_input_host,
    float* c_output_host) {
    detail::ValidateConfig(config);
    if (a_host == NULL || b_host == NULL || c_output_host == NULL) {
        throw std::invalid_argument("Host pointers for A, B and output C must be non-null");
    }

    for (int row = 0; row < config.m; ++row) {
        for (int col = 0; col < config.n; ++col) {
            float accum = 0.0f;
            for (int inner = 0; inner < config.k; ++inner) {
                accum += a_host[row * config.lda + inner] * b_host[inner * config.ldb + col];
            }

            const float c_value = (c_input_host != NULL) ? c_input_host[row * config.ldc + col] : 0.0f;
            c_output_host[row * config.ldc + col] = config.alpha * accum + config.beta * c_value;
        }
    }
}

float MaxAbsDiff(const float* lhs, const float* rhs, std::size_t element_count) {
    if (lhs == NULL || rhs == NULL) {
        throw std::invalid_argument("Input arrays for MaxAbsDiff must be non-null");
    }

    float max_abs_diff = 0.0f;
    for (std::size_t index = 0; index < element_count; ++index) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(lhs[index] - rhs[index]));
    }
    return max_abs_diff;
}

}  // namespace fastcuda

const char* fastcudaGetStatusString(FastCudaStatus status) {
    switch (status) {
        case FASTCUDA_STATUS_SUCCESS:
            return "success";
        case FASTCUDA_STATUS_INVALID_VALUE:
            return "invalid_value";
        case FASTCUDA_STATUS_CUDA_ERROR:
            return "cuda_error";
        case FASTCUDA_STATUS_INTERNAL_ERROR:
            return "internal_error";
        default:
            return "unknown_status";
    }
}

const char* fastcudaGetLastErrorMessage(void) {
    return fastcuda::detail::g_last_error_message.c_str();
}

const char* fastcudaGetGemmAlgorithmName(FastCudaGemmAlgorithm algorithm) {
    try {
        return fastcuda::GemmAlgorithmName(fastcuda::detail::ToCppAlgorithm(algorithm));
    } catch (...) {
        return "unknown";
    }
}

FastCudaStatus fastcudaSgemmDevice(
    FastCudaGemmAlgorithm algorithm,
    const FastCudaGemmConfig* config,
    const float* a_device,
    const float* b_device,
    float* c_device,
    int warmup_iterations,
    int timed_iterations,
    float* elapsed_ms) {
    if (config == NULL || elapsed_ms == NULL) {
        fastcuda::detail::SetLastErrorMessage("Config and elapsed_ms must be non-null");
        return FASTCUDA_STATUS_INVALID_VALUE;
    }

    try {
        fastcuda::detail::SetLastErrorMessage("");
        const fastcuda::GemmTiming timing = fastcuda::BenchmarkSgemmDevice(
            fastcuda::detail::ToCppAlgorithm(algorithm),
            fastcuda::detail::ToCppConfig(*config),
            a_device,
            b_device,
            c_device,
            warmup_iterations,
            timed_iterations);
        *elapsed_ms = timing.elapsed_ms;
        return FASTCUDA_STATUS_SUCCESS;
    } catch (const std::exception& ex) {
        return fastcuda::detail::HandleCError(ex);
    }
}

FastCudaStatus fastcudaSgemmHost(
    FastCudaGemmAlgorithm algorithm,
    const FastCudaGemmConfig* config,
    const float* a_host,
    const float* b_host,
    const float* c_input_host,
    float* c_output_host,
    int warmup_iterations,
    int timed_iterations,
    float* elapsed_ms) {
    if (config == NULL || elapsed_ms == NULL) {
        fastcuda::detail::SetLastErrorMessage("Config and elapsed_ms must be non-null");
        return FASTCUDA_STATUS_INVALID_VALUE;
    }

    try {
        fastcuda::detail::SetLastErrorMessage("");
        const fastcuda::GemmTiming timing = fastcuda::RunSgemmHost(
            fastcuda::detail::ToCppAlgorithm(algorithm),
            fastcuda::detail::ToCppConfig(*config),
            a_host,
            b_host,
            c_input_host,
            c_output_host,
            warmup_iterations,
            timed_iterations);
        *elapsed_ms = timing.elapsed_ms;
        return FASTCUDA_STATUS_SUCCESS;
    } catch (const std::exception& ex) {
        return fastcuda::detail::HandleCError(ex);
    }
}