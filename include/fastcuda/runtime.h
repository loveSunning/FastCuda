/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 */

#ifndef FASTCUDA_RUNTIME_H
#define FASTCUDA_RUNTIME_H

#include "fastcuda/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FastCudaDeviceInfo {
    int   index;
    char  name[256];
    int   major;
    int   minor;
    size_t global_memory_bytes;
    int   multiprocessor_count;
} FastCudaDeviceInfo;

FASTCUDA_API FastCudaStatus fastcuda_get_device_count(int* count);
FASTCUDA_API FastCudaStatus fastcuda_get_device_info(int index, FastCudaDeviceInfo* info);

#ifdef __cplusplus
}
#endif

#endif /* FASTCUDA_RUNTIME_H */
