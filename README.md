# FastCuda

If you find this project useful, please give it a star.

[中文说明](README.zh-CN.md)

FastCuda is a lightweight CUDA kernel workspace for handwritten operator development, benchmarking, and profiling on modern NVIDIA GPUs.

Today the repository ships a usable SGEMM library with three implementation tiers, a benchmark executable, and a minimal example program.

## What You Get

- Three SGEMM paths in one library: `naive`, `tiled_shared`, and `register_blocked`
- Both C++ API and C ABI for downstream integration
- Benchmark entrypoint for repeatable kernel comparisons
- Device query utilities for quick environment checks
- CMake-based build for Windows and Linux
- Defaults tuned around RTX 4090 (`sm_89`) and RTX 5060 (`sm_120`)

## Demo

### Example Output

```text
m=256 n=256 k=256
algorithm=naive elapsed_ms=0.0340 max_abs_error=0.0000
algorithm=tiled_shared elapsed_ms=0.0359 max_abs_error=0.0000
algorithm=register_blocked elapsed_ms=0.0295 max_abs_error=0.0000
```

### Benchmark Output

```text
operator=gemm
kernel=all
shape=m=256,n=256,k=256
requested_dtype=fp32
effective_dtype=fp32
device_count=1
device[0] name=NVIDIA GeForce RTX 5060 cc=12.0 global_mem_bytes=8546484224 sms=30
algorithm=naive
elapsed_ms=0.0336
max_abs_error=0.000000e+00
algorithm=tiled_shared
elapsed_ms=0.0359
max_abs_error=0.000000e+00
algorithm=register_blocked
elapsed_ms=0.0295
max_abs_error=0.000000e+00
status=ok
```

## Quick Start

### 1. Configure

Windows:

```powershell
cmd /c ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" && cmake -S . -B build -G "Visual Studio 17 2022" -A x64"
```

Linux:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

### 2. Build

Windows:

```powershell
cmake --build build --config Release
```

Linux:

```bash
cmake --build build -j
```

### 3. Run

Run the example:

```powershell
.\build\Release\fastcuda_gemm_example.exe 512 512 512
```

Run the benchmark wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/perf/run-benchmark.ps1 -Operator gemm -Kernel all -Shape m=1024,n=1024,k=1024 -DType fp32
```

## Core Features

- Handwritten CUDA SGEMM kernels with explicit implementation tiers
- Stable public surface through both C++ and C interfaces
- Reproducible benchmark flow through `scripts/perf/run-benchmark.ps1`
- Environment probing through `scripts/env/probe-env.ps1`
- Profiling entrypoints for Nsight Compute and Nsight Systems
- Installation layout with `bin`, `lib`, and `include/fastcuda`

## Implemented Functionality

### SGEMM Algorithms

- `naive`: one thread computes one output element
- `tiled_shared`: `16x16x16` shared-memory tiled kernel
- `register_blocked`: `64x64x16` macro-tiled kernel with `4x4` register blocking per thread

### Public Headers

- `fastcuda/gemm.hpp`: C++ SGEMM API
- `fastcuda/gemm_c.h`: C ABI for external callers
- `fastcuda/runtime.hpp`: device discovery and device summary helpers

### Build Outputs

- Shared library: `fastcuda.dll` on Windows, `libfastcuda.so` on Linux
- Static library: `fastcuda_static`
- Benchmark executable: `fastcuda_bench`
- Example executable: `fastcuda_gemm_example`

## Use Cases

FastCuda is a strong fit if you want to:

- study CUDA GEMM optimization step by step
- build a small CUDA library with a clean ABI boundary
- benchmark kernel changes without wiring a large framework
- validate correctness against a simple reference implementation
- keep benchmarking, profiling, and environment inspection in one repository

## Requirements

- Windows or Linux host
- CMake `>= 3.24`
- CUDA Toolkit `12.8.x` or `13.0.x`
- NVIDIA GPU matching the configured architecture set, defaulting to `89;120`

## FAQ

### Which platforms are supported?

Windows and Linux are supported by the build system.

### Which CUDA versions are supported?

The current CMake configuration accepts CUDA `12.8.x` and `13.0.x`.

### Does it support FP16, BF16, or Tensor Cores?

Not yet. The current library exposes FP32 SGEMM paths only.

### How do I compare all current GEMM implementations?

Use the benchmark wrapper with `-Kernel all` or run `fastcuda_gemm_example`.

### What is the difference between the three GEMM implementations?

`naive` is the correctness-first baseline, `tiled_shared` adds shared-memory blocking, and `register_blocked` adds a larger macro-tile with per-thread register blocking.

### Where do profiling and environment scripts live?

Environment checks are under `scripts/env`, benchmarking and profiling wrappers are under `scripts/perf`, and helper hooks are under `scripts/hooks`.

## Repository Layout

```text
.
|-- AGENTS.md
|-- benchmarks/
|-- configs/
|-- docs/
|-- examples/
|-- scripts/
|-- src/
`-- templates/
```

## Contributing

Issues and pull requests are welcome. Small, focused kernel, benchmarking, documentation, and build improvements are preferred.

## Author

Maintained by the FastCuda contributors.

## License

This project is licensed under the Apache License, Version 2.0 — see the [LICENSE](LICENSE) file for details.

## More Documentation

- [AGENTS.md](AGENTS.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/agent-workflow.md](docs/agent-workflow.md)
- [docs/codex-configuration.md](docs/codex-configuration.md)
