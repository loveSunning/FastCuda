cmake_minimum_required(VERSION 3.24)

project(FastCuda VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows|Linux")
    message(FATAL_ERROR "FastCuda supports only Windows and Linux hosts.")
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_COMPILER_VERSION)
    message(FATAL_ERROR "Unable to determine CUDA compiler version.")
endif()

if(NOT (
    (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8" AND
     CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.9")
    OR
    (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0" AND
     CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.1")
))
    message(FATAL_ERROR
        "FastCuda currently supports CUDA 12.8.x and 13.0.x. "
        "Detected: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(FASTCUDA_CUDA_ARCHITECTURES "89;120" CACHE STRING
    "CUDA architectures to build for. Defaults cover RTX 4090 (89) and RTX 5060 (120).")

set(CMAKE_CUDA_ARCHITECTURES "${FASTCUDA_CUDA_ARCHITECTURES}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fastcuda/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/fastcuda/version.hpp
    @ONLY
)

add_library(fastcuda STATIC
    src/fastcuda/runtime.cu
)

target_include_directories(fastcuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_link_libraries(fastcuda
    PUBLIC
        CUDA::cudart
)

target_compile_features(fastcuda PUBLIC cxx_std_11)
set_target_properties(fastcuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(fastcuda_bench
    benchmarks/bench_main.cu
)

target_link_libraries(fastcuda_bench PRIVATE fastcuda)
target_compile_features(fastcuda_bench PRIVATE cxx_std_11)
set_target_properties(fastcuda_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "FastCuda host platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "FastCuda CUDA compiler: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "FastCuda CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
