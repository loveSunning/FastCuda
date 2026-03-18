/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "fastcuda/export.h"

#include <cstddef>
#include <string>
#include <vector>

namespace fastcuda {

struct DeviceInfo {
    int         index;
    std::string name;
    int         major;
    int         minor;
    std::size_t global_memory_bytes;
    int         multiprocessor_count;
};

FASTCUDA_API std::vector<DeviceInfo> QueryDevices();
FASTCUDA_API std::string FormatDeviceSummary(const std::vector<DeviceInfo>& devices);

}  // namespace fastcuda
