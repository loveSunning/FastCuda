# FastCuda

If you find this project useful, please give it a star.

[中文说明](README.zh-CN.md)

FastCuda is an enterprise-grade CUDA operator library for handwritten kernel development, benchmarking, and profiling on modern NVIDIA GPUs. It ships six progressive GEMM implementations (from naïve to Tensor-Core HGEMM), eight parallel reduction variants, a cuBLAS comparison benchmark, and C / C++ / Python interfaces.

## What You Get

| Category | Content |
|----------|---------|
| **GEMM** | 6 kernels: V1 Naive → V2 Shared Memory → V3 Register Blocking → V4 Double Buffer → V5 TF32 Tensor Core → V6 FP16 HGEMM |
| **Reduce** | 8 kernels: V0 Baseline → V1 No Divergence → V2 No Bank Conflict → V3 Add-During-Load → V4 Unroll Last Warp → V5 Fully Unrolled → V6 Multi-Add → V7 Warp Shuffle |
| **Interfaces** | C header, C++ header, Python (pybind11) bindings |
| **Benchmark** | Built-in cuBLAS comparison, GFLOPS / bandwidth reporting |
| **Build** | CMake, Windows + Linux, shared & static libraries |
| **Targets** | RTX 4090 (`sm_89`), RTX 5060 (`sm_120`) |

## Quick Start

### 1. Configure

Windows:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```

Linux:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

### 2. Build

```bash
cmake --build build --config Release -j
```

### 3. Run

```bash
# GEMM example (runs V1–V5 with CPU reference check)
./build/Release/fastcuda_gemm_example 512 512 512

# Reduce example (runs V0–V7)
./build/Release/fastcuda_reduce_example 1048576

# Benchmark (GEMM vs cuBLAS + all reduce versions)
./build/Release/fastcuda_bench gemm all m=1024,n=1024,k=1024
./build/Release/fastcuda_bench reduce all n=1048576
```

### 4. Python (optional)

```bash
cmake -S . -B build -DFASTCUDA_BUILD_PYTHON=ON
cmake --build build --config Release -j
cd python && pip install -e .
```

```python
import fastcuda
info = fastcuda.query_devices()
C = fastcuda.sgemm(A, B, algorithm=fastcuda.GemmAlgorithm.REGISTER_BLOCKED_V3)
total = fastcuda.reduce_sum(data, algorithm=fastcuda.ReduceAlgorithm.SHUFFLE_V7)
```

## GEMM Implementations

| Version | Strategy | Tile / Block | Precision |
|---------|----------|-------------|-----------|
| V1 | Naïve (1 thread → 1 element) | 16×16 | FP32 |
| V2 | Shared-memory tiling | 32×32, TK=32 | FP32 |
| V3 | Register blocking + float4 | BM=64, BN=64, TM=4, TN=4 | FP32 |
| V4 | Double-buffered shared mem + prefetch | BM=128, BN=128, TM=8, TN=8 | FP32 |
| V5 | Tensor Core (wmma, TF32) | 16×16×8 fragments | TF32→FP32 |
| V6 | Tensor Core HGEMM | 16×16×16 fragments | FP16→FP32 |

See [docs/gemm_optimization.md](docs/gemm_optimization.md) for the full optimization walkthrough.

## Reduce Implementations

| Version | Key Technique |
|---------|--------------|
| V0 | Baseline interleaved (divergent) |
| V1 | Strided index (no warp divergence) |
| V2 | Sequential addressing (no bank conflict) |
| V3 | Add during load (halves blocks) |
| V4 | Unroll last warp (save barriers) |
| V5 | Fully template-unrolled |
| V6 | Multi-element accumulation (8× per thread) |
| V7 | Warp shuffle (`__shfl_down_sync`) |

See [docs/reduce_optimization.md](docs/reduce_optimization.md) for the full optimization walkthrough.

## Public Headers

| Header | Language | Description |
|--------|----------|-------------|
| `fastcuda/gemm.h` | C | SGEMM / TF32 / HGEMM functions |
| `fastcuda/gemm.hpp` | C++ | GEMM namespace API with enum class |
| `fastcuda/reduce.h` | C | Reduce sum functions |
| `fastcuda/reduce.hpp` | C++ | Reduce namespace API with enum class |
| `fastcuda/runtime.h` | C | Device query |
| `fastcuda/runtime.hpp` | C++ | Device query with string formatting |
| `fastcuda/fastcuda.h` | C | Umbrella header |
| `fastcuda/fastcuda.hpp` | C++ | Umbrella header |

## Build Outputs

- `fastcuda.dll` / `libfastcuda.so` — shared library
- `fastcuda_static` — static library
- `fastcuda_bench` — benchmark executable (links cuBLAS)
- `fastcuda_gemm_example` — GEMM example
- `fastcuda_reduce_example` — Reduce example
- `fastcuda_python` — Python module (when `FASTCUDA_BUILD_PYTHON=ON`)

## Project Structure

```text
.
├── include/fastcuda/   # Public C/C++ headers
├── src/
│   ├── gemm/           # 6 GEMM kernel files + API dispatch
│   ├── reduce/         # 8 reduce kernel files + API dispatch
│   ├── runtime/        # Device query
│   └── common/         # Internal helpers
├── python/             # pybind11 bindings
├── examples/           # GEMM and reduce examples
├── benchmarks/         # Benchmark (cuBLAS comparison)
├── docs/               # Optimization guides (EN + CN)
├── scripts/            # Build, benchmark, profile helpers
├── configs/            # Device profiles, benchmark defaults
└── templates/          # Report templates
```

## Planned

The following operators are planned but not yet implemented:

- GEMV (matrix–vector multiply)
- SpMM (sparse × dense matrix multiply)
- SpMV (sparse matrix–vector multiply)
- FlashAttention

## Requirements

- Windows or Linux
- CMake ≥ 3.24
- CUDA Toolkit 12.8.x or 13.0.x
- NVIDIA GPU with `sm_89` or `sm_120` (configurable)
- cuBLAS (for benchmark comparison)
- pybind11 (optional, for Python bindings)

## FAQ

### Does it support FP16 and Tensor Cores?

Yes. V5 uses TF32 Tensor Cores and V6 implements full FP16 HGEMM via `nvcuda::wmma`.

### How do I compare against cuBLAS?

Run `fastcuda_bench gemm all ...` — it automatically includes a cuBLAS SGEMM baseline with GFLOPS comparison.

### Where do profiling and environment scripts live?

Environment checks are under `scripts/env`, benchmarking and profiling wrappers are under `scripts/perf`, and helper hooks are under `scripts/hooks`.

## Contributing

Issues and pull requests are welcome. Small, focused kernel, benchmarking, documentation, and build improvements are preferred.

## Author

Maintained by the FastCuda contributors.

## License

This project is licensed under the Apache License, Version 2.0 — see the [LICENSE](LICENSE) file for details.

## More Documentation

- [GEMM Optimization Guide](docs/gemm_optimization.md) / [中文](docs/gemm_optimization.zh-CN.md)
- [Reduce Optimization Guide](docs/reduce_optimization.md) / [中文](docs/reduce_optimization.zh-CN.md)
- [Architecture](docs/architecture.md)
- [Agent Workflow](docs/agent-workflow.md)
- [AGENTS.md](AGENTS.md)
- [docs/codex-configuration.md](docs/codex-configuration.md)
