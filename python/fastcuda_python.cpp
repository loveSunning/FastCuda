/*
 * Copyright 2026 FastCuda contributors
 * Licensed under the Apache License, Version 2.0
 *
 * Python bindings via pybind11.
 *
 * Build with: cmake -DFASTCUDA_BUILD_PYTHON=ON ..
 * The resulting module exposes the core C++ API to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "fastcuda/gemm.hpp"
#include "fastcuda/reduce.hpp"
#include "fastcuda/runtime.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

/* ---- helpers ---- */

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

/* Run a full SGEMM on host numpy arrays (row-major FP32). */
py::dict run_sgemm(int algo_int, int m, int n, int k,
                   py::array_t<float> a, py::array_t<float> b,
                   float alpha, float beta,
                   int warmup, int timed) {
    auto abuf = a.request();
    auto bbuf = b.request();
    if (abuf.size < (size_t)m * k || bbuf.size < (size_t)k * n)
        throw std::invalid_argument("input array too small");

    fastcuda::GemmAlgorithm algo =
        static_cast<fastcuda::GemmAlgorithm>(algo_int);

    fastcuda::GemmConfig cfg;
    cfg.m = m; cfg.n = n; cfg.k = k;
    cfg.lda = k; cfg.ldb = n; cfg.ldc = n;
    cfg.alpha = alpha; cfg.beta = beta;

    std::vector<float> c_out((size_t)m * n, 0.0f);
    fastcuda::GemmTiming t = fastcuda::RunSgemmHost(
        algo, cfg,
        static_cast<const float*>(abuf.ptr),
        static_cast<const float*>(bbuf.ptr),
        NULL, c_out.data(), warmup, timed);

    py::array_t<float> c_arr({m, n});
    std::memcpy(c_arr.mutable_data(), c_out.data(), c_out.size() * sizeof(float));

    py::dict result;
    result["C"]          = c_arr;
    result["elapsed_ms"] = t.elapsed_ms;
    return result;
}

py::dict run_hgemm(int m, int n, int k,
                   py::array a,
                   py::array b,
                   float alpha, float beta,
                   int warmup, int timed) {
    auto abuf = a.request();
    auto bbuf = b.request();
    if (abuf.itemsize != 2 || bbuf.itemsize != 2)
        throw std::invalid_argument("hgemm expects float16 arrays");
    if (abuf.size < (size_t)m * k || bbuf.size < (size_t)k * n)
        throw std::invalid_argument("input array too small");

    fastcuda::GemmConfig cfg;
    cfg.m = m; cfg.n = n; cfg.k = k;
    cfg.lda = k; cfg.ldb = n; cfg.ldc = n;
    cfg.alpha = alpha; cfg.beta = beta;

    std::vector<float> c_out((size_t)m * n, 0.0f);
    fastcuda::GemmTiming t = fastcuda::RunHgemmHost(
        cfg,
        abuf.ptr,
        bbuf.ptr,
        NULL,
        c_out.data(),
        warmup,
        timed);

    py::array_t<float> c_arr({m, n});
    std::memcpy(c_arr.mutable_data(), c_out.data(), c_out.size() * sizeof(float));

    py::dict result;
    result["C"] = c_arr;
    result["elapsed_ms"] = t.elapsed_ms;
    return result;
}

/* Run a full Reduce on host numpy array (FP32). */
py::dict run_reduce(int algo_int, py::array_t<float> input,
                    int warmup, int timed) {
    auto buf = input.request();
    int n = static_cast<int>(buf.size);

    fastcuda::ReduceAlgorithm algo =
        static_cast<fastcuda::ReduceAlgorithm>(algo_int);

    fastcuda::ReduceConfig cfg;
    cfg.n = n;

    float result_val = 0.0f;
    fastcuda::ReduceTiming t = fastcuda::RunReduceSumHost(
        algo, cfg,
        static_cast<const float*>(buf.ptr),
        &result_val, warmup, timed);

    py::dict result;
    result["sum"]        = result_val;
    result["elapsed_ms"] = t.elapsed_ms;
    return result;
}

}  // anonymous namespace

PYBIND11_MODULE(fastcuda_python, m) {
    m.doc() = "FastCuda Python bindings - CUDA operators for GEMM and Reduce";

    /* ---- Enums ---- */
    py::enum_<fastcuda::GemmAlgorithm>(m, "GemmAlgorithm")
        .value("NaiveV1",     fastcuda::GemmAlgorithm::kNaiveV1)
        .value("SharedMemV2", fastcuda::GemmAlgorithm::kSharedMemV2)
        .value("RegisterV3",  fastcuda::GemmAlgorithm::kRegisterV3)
        .value("WarpV4",      fastcuda::GemmAlgorithm::kWarpV4)
        .value("TF32V5",      fastcuda::GemmAlgorithm::kTF32V5)
        .value("HgemmV6",     fastcuda::GemmAlgorithm::kHgemmV6)
        .export_values();

    py::enum_<fastcuda::ReduceAlgorithm>(m, "ReduceAlgorithm")
        .value("BaselineV0",        fastcuda::ReduceAlgorithm::kBaselineV0)
        .value("NoDivergenceV1",    fastcuda::ReduceAlgorithm::kNoDivergenceV1)
        .value("NoBankConflictV2",  fastcuda::ReduceAlgorithm::kNoBankConflictV2)
        .value("AddDuringLoadV3",   fastcuda::ReduceAlgorithm::kAddDuringLoadV3)
        .value("UnrollLastWarpV4",  fastcuda::ReduceAlgorithm::kUnrollLastWarpV4)
        .value("CompletelyUnrollV5",fastcuda::ReduceAlgorithm::kCompletelyUnrollV5)
        .value("MultiAddV6",        fastcuda::ReduceAlgorithm::kMultiAddV6)
        .value("ShuffleV7",         fastcuda::ReduceAlgorithm::kShuffleV7)
        .export_values();

    /* ---- Device query ---- */
    py::class_<fastcuda::DeviceInfo>(m, "DeviceInfo")
        .def_readonly("index",                &fastcuda::DeviceInfo::index)
        .def_readonly("name",                 &fastcuda::DeviceInfo::name)
        .def_readonly("major",                &fastcuda::DeviceInfo::major)
        .def_readonly("minor",                &fastcuda::DeviceInfo::minor)
        .def_readonly("global_memory_bytes",  &fastcuda::DeviceInfo::global_memory_bytes)
        .def_readonly("multiprocessor_count", &fastcuda::DeviceInfo::multiprocessor_count);

    m.def("query_devices", &fastcuda::QueryDevices,
            "Return a list of CUDA device info structs.");

    /* ---- GEMM ---- */
    m.def("sgemm", &run_sgemm,
          py::arg("algo"), py::arg("m"), py::arg("n"), py::arg("k"),
          py::arg("A"), py::arg("B"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("warmup") = 5, py::arg("timed") = 20,
            "Run SGEMM (FP32) on host arrays. Returns dict with C and elapsed_ms.");

        m.def("hgemm", &run_hgemm,
            py::arg("m"), py::arg("n"), py::arg("k"),
            py::arg("A"), py::arg("B"),
            py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
            py::arg("warmup") = 5, py::arg("timed") = 20,
                "Run HGEMM (FP16 input, FP32 output) on host arrays. Returns dict with C and elapsed_ms.");

    m.def("gemm_algorithm_name",
          [](int a) { return fastcuda::GemmAlgorithmName(
              static_cast<fastcuda::GemmAlgorithm>(a)); },
          "Return human readable name for a GEMM algorithm.");

    /* ---- Reduce ---- */
    m.def("reduce_sum", &run_reduce,
          py::arg("algo"), py::arg("input"),
          py::arg("warmup") = 5, py::arg("timed") = 20,
            "Run reduce sum on a host FP32 array. Returns dict with sum and elapsed_ms.");

    m.def("reduce_algorithm_name",
          [](int a) { return fastcuda::ReduceAlgorithmName(
              static_cast<fastcuda::ReduceAlgorithm>(a)); },
          "Return human readable name for a Reduce algorithm.");
}
