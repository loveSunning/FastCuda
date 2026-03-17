#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace fastcuda {

struct DeviceInfo {
    int index;
    std::string name;
    int major;
    int minor;
    std::size_t global_memory_bytes;
    int multiprocessor_count;
};

std::vector<DeviceInfo> QueryDevices();
std::string FormatDeviceSummary(const std::vector<DeviceInfo>& devices);

}  // namespace fastcuda
