#include "fastcuda/runtime.hpp"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace fastcuda {

namespace {

void CheckCuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

std::vector<DeviceInfo> QueryDevices() {
    int count = 0;
    CheckCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");

    std::vector<DeviceInfo> devices;
    devices.reserve(static_cast<std::size_t>(count));

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        CheckCuda(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");

        DeviceInfo info;
        info.index = i;
        info.name = prop.name;
        info.major = prop.major;
        info.minor = prop.minor;
        info.global_memory_bytes = static_cast<std::size_t>(prop.totalGlobalMem);
        info.multiprocessor_count = prop.multiProcessorCount;
        devices.push_back(info);
    }

    return devices;
}

std::string FormatDeviceSummary(const std::vector<DeviceInfo>& devices) {
    std::ostringstream oss;
    oss << "device_count=" << devices.size() << "\n";

    for (std::size_t i = 0; i < devices.size(); ++i) {
        const DeviceInfo& device = devices[i];
        oss << "device[" << i << "]"
            << " name=" << device.name
            << " cc=" << device.major << "." << device.minor
            << " global_mem_bytes=" << device.global_memory_bytes
            << " sms=" << device.multiprocessor_count
            << "\n";
    }

    return oss.str();
}

}  // namespace fastcuda
