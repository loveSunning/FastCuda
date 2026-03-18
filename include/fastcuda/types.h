/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 */

#ifndef FASTCUDA_TYPES_H
#define FASTCUDA_TYPES_H

#include "fastcuda/export.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FastCudaStatus {
    FASTCUDA_STATUS_SUCCESS        = 0,
    FASTCUDA_STATUS_INVALID_VALUE  = 1,
    FASTCUDA_STATUS_CUDA_ERROR     = 2,
    FASTCUDA_STATUS_INTERNAL_ERROR = 3,
    FASTCUDA_STATUS_NOT_SUPPORTED  = 4
} FastCudaStatus;

FASTCUDA_API const char* fastcuda_get_status_string(FastCudaStatus status);
FASTCUDA_API const char* fastcuda_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTCUDA_TYPES_H */
