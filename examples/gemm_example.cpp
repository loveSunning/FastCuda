#include "fastcuda/gemm.hpp"
#include "fastcuda/gemm_c.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void FillMatrix(std::vector<float>* values, float scale) {
    for (std::size_t index = 0; index < values->size(); ++index) {
        (*values)[index] = static_cast<float>((index % 19) - 9) * scale;
    }
}

float ComputeMaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    float max_abs_diff = 0.0f;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(lhs[index] - rhs[index]));
    }
    return max_abs_diff;
}

void PrintResult(
    FastCudaGemmAlgorithm algorithm,
    const FastCudaGemmConfig& config,
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c_init,
    const std::vector<float>& c_reference) {
    std::vector<float> c_output(c_init.size(), 0.0f);
    float elapsed_ms = 0.0f;

    const FastCudaStatus status = fastcudaSgemmHost(
        algorithm,
        &config,
        a.data(),
        b.data(),
        c_init.data(),
        c_output.data(),
        5,
        20,
        &elapsed_ms);

    if (status != FASTCUDA_STATUS_SUCCESS) {
        std::cerr << "algorithm=" << fastcudaGetGemmAlgorithmName(algorithm)
                  << " status=" << fastcudaGetStatusString(status)
                  << " error=" << fastcudaGetLastErrorMessage() << "\n";
        std::exit(1);
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "algorithm=" << fastcudaGetGemmAlgorithmName(algorithm)
              << " elapsed_ms=" << elapsed_ms
              << " max_abs_error=" << ComputeMaxAbsDiff(c_output, c_reference)
              << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    FastCudaGemmConfig config;
    config.m = (argc > 1) ? std::atoi(argv[1]) : 512;
    config.n = (argc > 2) ? std::atoi(argv[2]) : 512;
    config.k = (argc > 3) ? std::atoi(argv[3]) : 512;
    config.lda = config.k;
    config.ldb = config.n;
    config.ldc = config.n;
    config.alpha = 1.0f;
    config.beta = 0.0f;
    config.stream = NULL;

    const std::size_t a_size = static_cast<std::size_t>(config.m) * config.lda;
    const std::size_t b_size = static_cast<std::size_t>(config.k) * config.ldb;
    const std::size_t c_size = static_cast<std::size_t>(config.m) * config.ldc;

    std::vector<float> a(a_size, 0.0f);
    std::vector<float> b(b_size, 0.0f);
    std::vector<float> c_init(c_size, 0.0f);
    std::vector<float> c_reference(c_size, 0.0f);

    FillMatrix(&a, 0.125f);
    FillMatrix(&b, 0.0625f);

    fastcuda::GemmConfig cpp_config;
    cpp_config.m = config.m;
    cpp_config.n = config.n;
    cpp_config.k = config.k;
    cpp_config.lda = config.lda;
    cpp_config.ldb = config.ldb;
    cpp_config.ldc = config.ldc;
    cpp_config.alpha = config.alpha;
    cpp_config.beta = config.beta;
    cpp_config.stream = config.stream;
    fastcuda::ReferenceSgemm(cpp_config, a.data(), b.data(), c_init.data(), c_reference.data());

    std::cout << "m=" << config.m << " n=" << config.n << " k=" << config.k << "\n";
    PrintResult(FASTCUDA_GEMM_NAIVE, config, a, b, c_init, c_reference);
    PrintResult(FASTCUDA_GEMM_TILED, config, a, b, c_init, c_reference);
    PrintResult(FASTCUDA_GEMM_REGISTER_BLOCKED, config, a, b, c_init, c_reference);
    return 0;
}