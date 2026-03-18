/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Internal CUDA helper macros – not part of the public API.
 */

#ifndef FASTCUDA_COMMON_CUDA_CHECK_H
#define FASTCUDA_COMMON_CUDA_CHECK_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace fastcuda {
namespace internal {

inline void CheckCuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(action) + ": " + cudaGetErrorString(status));
    }
}

inline cudaStream_t ToCudaStream(void* s) {
    return reinterpret_cast<cudaStream_t>(s);
}

template <typename T>
inline T CeilDiv(T a, T b) {
    return (a + b - 1) / b;
}

}  // namespace internal
}  // namespace fastcuda

#endif /* FASTCUDA_COMMON_CUDA_CHECK_H */
