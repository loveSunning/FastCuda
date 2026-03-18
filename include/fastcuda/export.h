/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 */

#ifndef FASTCUDA_EXPORT_H
#define FASTCUDA_EXPORT_H

#if defined(_WIN32)
    #if defined(FASTCUDA_BUILD_SHARED)
        #define FASTCUDA_API __declspec(dllexport)
    #elif defined(FASTCUDA_IMPORT_SHARED)
        #define FASTCUDA_API __declspec(dllimport)
    #else
        #define FASTCUDA_API
    #endif
#else
    #define FASTCUDA_API __attribute__((visibility("default")))
#endif

#endif /* FASTCUDA_EXPORT_H */
