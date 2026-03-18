cmake_minimum_required(VERSION 3.24)

project(FastCuda VERSION 0.2.0 LANGUAGES CXX CUDA)

# ── Language standards ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ── Platform guard ──────────────────────────────────────────────────
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows|Linux")
    message(FATAL_ERROR "FastCuda supports only Windows and Linux hosts.")
endif()

# ── CUDA toolkit ────────────────────────────────────────────────────
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
    "CUDA architectures to build for (RTX 4090=89, RTX 5060=120).")
set(CMAKE_CUDA_ARCHITECTURES "${FASTCUDA_CUDA_ARCHITECTURES}")

# ── Options ─────────────────────────────────────────────────────────
option(FASTCUDA_BUILD_PYTHON   "Build pybind11 Python module" OFF)
option(FASTCUDA_BUILD_EXAMPLES "Build example executables"     ON)
option(FASTCUDA_BUILD_BENCH    "Build benchmark executable"    ON)

# ── Version header ──────────────────────────────────────────────────
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fastcuda/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/fastcuda/version.hpp
    @ONLY
)

# ── Source lists ────────────────────────────────────────────────────
set(FASTCUDA_SOURCES
    src/gemm/gemm_v1_naive.cu
    src/gemm/gemm_v2_shared.cu
    src/gemm/gemm_v3_register.cu
    src/gemm/gemm_v4_warp.cu
    src/gemm/gemm_v5_tf32.cu
    src/gemm/gemm_v6_hgemm.cu
    src/gemm/gemm_api.cu
    src/reduce/reduce_v0_baseline.cu
    src/reduce/reduce_v1_no_divergence.cu
    src/reduce/reduce_v2_no_bank_conflict.cu
    src/reduce/reduce_v3_add_during_load.cu
    src/reduce/reduce_v4_unroll_last_warp.cu
    src/reduce/reduce_v5_completely_unroll.cu
    src/reduce/reduce_v6_multi_add.cu
    src/reduce/reduce_v7_shuffle.cu
    src/reduce/reduce_api.cu
    src/runtime/runtime.cu
)

set(FASTCUDA_PUBLIC_HEADERS
    include/fastcuda/export.h
    include/fastcuda/types.h
    include/fastcuda/gemm.h
    include/fastcuda/gemm.hpp
    include/fastcuda/reduce.h
    include/fastcuda/reduce.hpp
    include/fastcuda/runtime.h
    include/fastcuda/runtime.hpp
    include/fastcuda/fastcuda.h
    include/fastcuda/fastcuda.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/generated/fastcuda/version.hpp
)

# ── Shared library ──────────────────────────────────────────────────
add_library(fastcuda SHARED ${FASTCUDA_SOURCES})

target_compile_definitions(fastcuda PRIVATE FASTCUDA_BUILD_SHARED)
target_include_directories(fastcuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(fastcuda PUBLIC CUDA::cudart)
target_compile_features(fastcuda PUBLIC cxx_std_11)
set_target_properties(fastcuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)

# ── Static library ──────────────────────────────────────────────────
add_library(fastcuda_static STATIC ${FASTCUDA_SOURCES})

target_include_directories(fastcuda_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(fastcuda_static PUBLIC CUDA::cudart)
target_compile_features(fastcuda_static PUBLIC cxx_std_11)
set_target_properties(fastcuda_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
    OUTPUT_NAME fastcuda_static
)

# ── Benchmark ───────────────────────────────────────────────────────
if(FASTCUDA_BUILD_BENCH)
    add_executable(fastcuda_bench benchmarks/bench_main.cu)
    target_link_libraries(fastcuda_bench PRIVATE fastcuda CUDA::cublas)
    target_compile_features(fastcuda_bench PRIVATE cxx_std_11)
    set_target_properties(fastcuda_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ── Examples ────────────────────────────────────────────────────────
if(FASTCUDA_BUILD_EXAMPLES)
    add_executable(fastcuda_gemm_example   examples/gemm_example.cpp)
    add_executable(fastcuda_reduce_example examples/reduce_example.cpp)

    target_link_libraries(fastcuda_gemm_example   PRIVATE fastcuda)
    target_link_libraries(fastcuda_reduce_example PRIVATE fastcuda)

    target_compile_features(fastcuda_gemm_example   PRIVATE cxx_std_11)
    target_compile_features(fastcuda_reduce_example PRIVATE cxx_std_11)
endif()

# ── Python bindings (pybind11) ──────────────────────────────────────
if(FASTCUDA_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(fastcuda_python python/fastcuda_python.cpp)
    target_link_libraries(fastcuda_python PRIVATE fastcuda)
    target_compile_features(fastcuda_python PRIVATE cxx_std_11)
endif()

# ── Install rules ───────────────────────────────────────────────────
set(_INSTALL_TARGETS fastcuda fastcuda_static)
if(FASTCUDA_BUILD_BENCH)
    list(APPEND _INSTALL_TARGETS fastcuda_bench)
endif()
if(FASTCUDA_BUILD_EXAMPLES)
    list(APPEND _INSTALL_TARGETS fastcuda_gemm_example fastcuda_reduce_example)
endif()

install(TARGETS ${_INSTALL_TARGETS}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${FASTCUDA_PUBLIC_HEADERS}
    DESTINATION include/fastcuda
)

# ── Summary ─────────────────────────────────────────────────────────
message(STATUS "FastCuda v${PROJECT_VERSION}")
message(STATUS "  Host platform : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  CUDA compiler : ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  Architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build examples: ${FASTCUDA_BUILD_EXAMPLES}")
message(STATUS "  Build bench   : ${FASTCUDA_BUILD_BENCH}")
message(STATUS "  Build python  : ${FASTCUDA_BUILD_PYTHON}")
