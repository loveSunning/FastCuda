#pragma once

#if defined(_WIN32)
#if defined(FASTCUDA_BUILD_SHARED)
#define FASTCUDA_API __declspec(dllexport)
#else
#define FASTCUDA_API
#endif
#else
#define FASTCUDA_API __attribute__((visibility("default")))
#endif