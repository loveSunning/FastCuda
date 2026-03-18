# FastCuda

FastCuda is a Codex-managed workspace for handwritten CUDA operator
development, with a near-term focus on:

- GEMM kernels
- FlashAttention kernels
- benchmarking and regression tracking
- performance analysis and environment inspection

The repository now exports a handwritten SGEMM library with three speed tiers:

- `naive`: one thread computes one output element
- `tiled_shared`: shared-memory tiling over `16x16x16`
- `register_blocked`: `64x64x16` macro tile with `4x4` per-thread register blocking

These kernels are exposed through both a C++ API and a C ABI so other software
can call the shared library directly.

## Build Targets

- build system: CMake
- host language standard: C++11
- CUDA language standard: C++11
- supported host platforms: Windows and Linux
- supported CUDA toolkits: 12.8.x and 13.0.x
- default CUDA architectures: `89;120`
  - `89` for GeForce RTX 4090
  - `120` for GeForce RTX 5060
- default device tiers:
  - RTX 4090: 24 GB
  - RTX 5060: 8 GB

## Codex Layout

This repository now follows Codex's official configuration style:

- root [AGENTS.md](E:/learning/cuda/FastCuda/AGENTS.md)
- [.codex/config.toml](E:/learning/cuda/FastCuda/.codex/config.toml)
- `.codex/rules/`
- `.codex/agents/`
- `.codex/skills/`

Project-specific prompt briefs live under `docs/prompts/`. Benchmark, profile,
and hook wrappers live under `scripts/`. Those are project assets, not Codex
configuration modules.

## Repository Layout

```text
.
|-- AGENTS.md
|-- .codex/
|   |-- config.toml
|   |-- agents/
|   |-- rules/
|   `-- skills/
|-- benchmarks/
|-- configs/
|-- docs/
|   `-- prompts/
|-- scripts/
|   |-- env/
|   |-- hooks/
|   `-- perf/
|-- src/
`-- templates/
```

## Configure And Build

### Windows

```powershell
cmd /c ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" && cmake -S . -B build -G "Visual Studio 17 2022" -A x64 && cmake --build build --config Release"
```

### Linux

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Benchmark Wrapper

```powershell
powershell -ExecutionPolicy Bypass -File scripts/perf/run-benchmark.ps1 -Operator gemm -Kernel baseline -Shape m=1024,n=1024,k=1024 -DType fp32
```

### Example Executable

After building, validate all three GEMM kernels with:

```powershell
.\build\Release\fastcuda_gemm_example.exe 512 512 512
```

On Linux, the shared target is `libfastcuda.so`; on Windows, it is
`fastcuda.dll`. The install step places headers in `include/fastcuda` and the
library in `lib`.

### Exported Headers

- `fastcuda/gemm.hpp`: C++ SGEMM API
- `fastcuda/gemm_c.h`: C ABI for external software
- `fastcuda/runtime.hpp`: device query utilities

See [docs/architecture.md](E:/learning/cuda/FastCuda/docs/architecture.md),
[docs/agent-workflow.md](E:/learning/cuda/FastCuda/docs/agent-workflow.md),
and [docs/codex-configuration.md](E:/learning/cuda/FastCuda/docs/codex-configuration.md).
