/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Runtime device query implementation.
 */

#include "fastcuda/runtime.hpp"
#include "fastcuda/runtime.h"
#include "common/cuda_check.h"

#include <cuda_runtime.h>
#include <cstring>
#include <sstream>

namespace fastcuda {

std::vector<DeviceInfo> QueryDevices() {
    int count = 0;
    internal::CheckCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");

    std::vector<DeviceInfo> devices;
    devices.reserve(static_cast<std::size_t>(count));

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        internal::CheckCuda(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");

        DeviceInfo info;
        info.index = i;
        info.name  = prop.name;
        info.major = prop.major;
        info.minor = prop.minor;
        info.global_memory_bytes  = static_cast<std::size_t>(prop.totalGlobalMem);
        info.multiprocessor_count = prop.multiProcessorCount;
        devices.push_back(info);
    }
    return devices;
}

std::string FormatDeviceSummary(const std::vector<DeviceInfo>& devices) {
    std::ostringstream oss;
    oss << "device_count=" << devices.size() << "\n";
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const DeviceInfo& d = devices[i];
        oss << "device[" << i << "]"
            << " name=" << d.name
            << " cc=" << d.major << "." << d.minor
            << " global_mem_bytes=" << d.global_memory_bytes
            << " sms=" << d.multiprocessor_count
            << "\n";
    }
    return oss.str();
}

}  // namespace fastcuda

/* C API */
extern "C" {

FastCudaStatus fastcuda_get_device_count(int* count) {
    if (!count) return FASTCUDA_STATUS_INVALID_VALUE;
    cudaError_t err = cudaGetDeviceCount(count);
    return (err == cudaSuccess) ? FASTCUDA_STATUS_SUCCESS : FASTCUDA_STATUS_CUDA_ERROR;
}

FastCudaStatus fastcuda_get_device_info(int index, FastCudaDeviceInfo* info) {
    if (!info) return FASTCUDA_STATUS_INVALID_VALUE;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, index);
    if (err != cudaSuccess) return FASTCUDA_STATUS_CUDA_ERROR;

    info->index = index;
    std::strncpy(info->name, prop.name, sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';
    info->major = prop.major;
    info->minor = prop.minor;
    info->global_memory_bytes  = static_cast<size_t>(prop.totalGlobalMem);
    info->multiprocessor_count = prop.multiProcessorCount;
    return FASTCUDA_STATUS_SUCCESS;
}

}  // extern "C"
