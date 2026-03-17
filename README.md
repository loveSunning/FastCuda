# FastCuda

FastCuda is a Codex-managed workspace for handwritten CUDA operator
development, with a near-term focus on:

- GEMM kernels
- FlashAttention kernels
- benchmarking and regression tracking
- performance analysis and environment inspection

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
powershell -ExecutionPolicy Bypass -File scripts/perf/run-benchmark.ps1 -Operator gemm -Kernel baseline
```

See [docs/architecture.md](E:/learning/cuda/FastCuda/docs/architecture.md),
[docs/agent-workflow.md](E:/learning/cuda/FastCuda/docs/agent-workflow.md),
and [docs/codex-configuration.md](E:/learning/cuda/FastCuda/docs/codex-configuration.md).
