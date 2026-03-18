#/*
 Copyright 2026 victor
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0
*/

#pragma once

#include "fastcuda/export.hpp"

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

FASTCUDA_API std::vector<DeviceInfo> QueryDevices();
FASTCUDA_API std::string FormatDeviceSummary(const std::vector<DeviceInfo>& devices);

}  // namespace fastcuda
