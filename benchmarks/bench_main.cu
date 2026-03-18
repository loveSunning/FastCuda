#/*
 Copyright 2026 victor
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0
*/

#include "fastcuda/gemm.hpp"
#include "fastcuda/runtime.hpp"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

int ParseDimension(const std::string& shape, const std::string& key, int fallback) {
    const std::string token = key + "=";
    const std::size_t start = shape.find(token);
    if (start == std::string::npos) {
        return fallback;
    }

    const std::size_t value_start = start + token.size();
    const std::size_t value_end = shape.find_first_of(", ", value_start);
    return std::atoi(shape.substr(value_start, value_end - value_start).c_str());
}

fastcuda::GemmAlgorithm ParseAlgorithm(const std::string& kernel_name) {
    if (kernel_name == "baseline" || kernel_name == "naive") {
        return fastcuda::GemmAlgorithm::kNaive;
    }
    if (kernel_name == "tiled" || kernel_name == "shared") {
        return fastcuda::GemmAlgorithm::kTiled;
    }
    if (kernel_name == "register" || kernel_name == "register_blocked" || kernel_name == "optimized") {
        return fastcuda::GemmAlgorithm::kRegisterBlocked;
    }
    throw std::invalid_argument("unsupported kernel name");
}

void FillInputs(std::vector<float>* a, std::vector<float>* b, std::vector<float>* c) {
    for (std::size_t index = 0; index < a->size(); ++index) {
        (*a)[index] = static_cast<float>((index % 17) - 8) * 0.125f;
    }
    for (std::size_t index = 0; index < b->size(); ++index) {
        (*b)[index] = static_cast<float>((index % 13) - 6) * 0.25f;
    }
    std::fill(c->begin(), c->end(), 0.0f);
}

void PrintOneResult(
    fastcuda::GemmAlgorithm algorithm,
    const fastcuda::GemmConfig& config,
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c_init,
    int warmup,
    int timed_iterations) {
    std::vector<float> c_output(c_init.size(), 0.0f);
    std::vector<float> c_reference(c_init.size(), 0.0f);

    const fastcuda::GemmTiming timing = fastcuda::RunSgemmHost(
        algorithm,
        config,
        a.data(),
        b.data(),
        c_init.data(),
        c_output.data(),
        warmup,
        timed_iterations);
    fastcuda::ReferenceSgemm(config, a.data(), b.data(), c_init.data(), c_reference.data());

    std::cout << "algorithm=" << fastcuda::GemmAlgorithmName(algorithm) << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "elapsed_ms=" << timing.elapsed_ms << "\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "max_abs_error="
              << fastcuda::MaxAbsDiff(c_output.data(), c_reference.data(), c_output.size())
              << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string operator_name = "gemm";
    std::string kernel_name = "baseline";
    std::string shape = "m=1024,n=1024,k=1024";
    std::string dtype = "fp32";

    if (argc > 1) {
        operator_name = argv[1];
    }
    if (argc > 2) {
        kernel_name = argv[2];
    }
    if (argc > 3) {
        shape = argv[3];
    }
    if (argc > 4) {
        dtype = argv[4];
    }

    try {
        if (operator_name != "gemm") {
            std::cout << "operator=" << operator_name << "\n";
            std::cout << "status=unsupported_operator\n";
            return 1;
        }

        const int m = ParseDimension(shape, "m", 1024);
        const int n = ParseDimension(shape, "n", 1024);
        const int k = ParseDimension(shape, "k", 1024);

        fastcuda::GemmConfig config;
        config.m = m;
        config.n = n;
        config.k = k;
        config.lda = k;
        config.ldb = n;
        config.ldc = n;
        config.alpha = 1.0f;
        config.beta = 0.0f;

        std::vector<float> a(static_cast<std::size_t>(m) * config.lda, 0.0f);
        std::vector<float> b(static_cast<std::size_t>(k) * config.ldb, 0.0f);
        std::vector<float> c(static_cast<std::size_t>(m) * config.ldc, 0.0f);
        FillInputs(&a, &b, &c);

        std::cout << "operator=" << operator_name << "\n";
        std::cout << "kernel=" << kernel_name << "\n";
        std::cout << "shape=" << shape << "\n";
        std::cout << "requested_dtype=" << dtype << "\n";
        std::cout << "effective_dtype=fp32\n";
        std::cout << fastcuda::FormatDeviceSummary(fastcuda::QueryDevices());

        if (kernel_name == "all") {
            PrintOneResult(fastcuda::GemmAlgorithm::kNaive, config, a, b, c, 5, 20);
            PrintOneResult(fastcuda::GemmAlgorithm::kTiled, config, a, b, c, 5, 20);
            PrintOneResult(fastcuda::GemmAlgorithm::kRegisterBlocked, config, a, b, c, 5, 20);
        } else {
            PrintOneResult(ParseAlgorithm(kernel_name), config, a, b, c, 5, 20);
        }

        std::cout << "status=ok\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error=" << ex.what() << "\n";
        return 1;
    }
}
