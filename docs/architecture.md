# FastCuda Architecture

## Goals

This workspace is designed for direct CUDA C++ operator development. It ships
six progressive GEMM kernels, eight parallel reduction kernels, and a cuBLAS
comparison benchmark. The architecture optimizes for:

- repeatable kernel iteration with progressive optimization
- explicit performance baselines and cuBLAS comparison
- low-friction profiling
- C / C++ / Python interface surface
- stable Codex behavior
- reusable CUDA skills and prompt briefs

## Build And Platform Targets

- build system: CMake
- host language standard: C++11
- CUDA language standard: C++11
- host platforms: Windows and Linux
- CUDA toolkits: 12.8.x and 13.0.x
- default GPU architectures:
  - RTX 4090 -> compute capability 8.9 -> `89`
  - RTX 5060 -> compute capability 12.0 -> `120`

## Codex Control Plane

FastCuda follows Codex's official configuration surfaces:

- `AGENTS.md`
  - project instructions, workflow, and durable constraints
- `.codex/config.toml`
  - project-scoped runtime configuration
- `.codex/rules/*.rules`
  - approval and command rules
- `.codex/agents/*.md`
  - custom subagents
- `.codex/skills/*/SKILL.md`
  - project skills

The following remain ordinary project assets rather than Codex config modules:

- `docs/prompts/`
- `scripts/`
- `configs/`
- `templates/`

## Design Principles

### 1. Codex config stays official

Repository agent behavior should live in `AGENTS.md`, `.codex/config.toml`,
`.codex/rules/`, `.codex/agents/`, and `.codex/skills/`. Do not add shadow
control planes or custom manifests for the same responsibility.

### 2. Execution goes through scripts

Benchmarking, environment inspection, and profiling should route through
`scripts/` instead of ad hoc shell snippets whenever possible.

### 3. Performance work is artifact-first

Performance claims should produce files, not only terminal output:

- environment snapshots
- benchmark JSON
- profiler output locations
- optimization notes

### 4. Device tiers are explicit

RTX 4090 and RTX 5060 are treated as separate benchmark tiers. Kernel behavior,
shape presets, and memory pressure should be interpreted against the active
device profile.

## Execution Plane

### `scripts/env/`

Environment and dependency probes:

- CUDA toolkit version
- NVCC version
- GPU inventory
- driver visibility
- PATH sanity checks

### `scripts/perf/`

Stable wrappers for:

- benchmark runs
- Nsight Compute
- Nsight Systems
- result directory creation

### `scripts/hooks/`

Project hook-like helper scripts for:

- pre-benchmark environment capture
- benchmark artifact verification
- profiler tool checks

### `configs/`

Machine-readable configs, benchmark presets, and device profiles.

## Recommended Workflow

1. Read `AGENTS.md`.
2. Check `.codex/config.toml` and the relevant skills or subagents.
3. Run environment probe if the machine state is uncertain.
4. Establish or refresh the benchmark baseline.
5. Implement a narrow kernel change.
6. Re-run benchmark and profile only when the data calls for it.

## Source Layout

```text
include/fastcuda/      Public C/C++ headers (installed)
  export.h             DLL export/import macros
  types.h              Status enum, error helpers (C)
  gemm.h / gemm.hpp    GEMM API (C / C++)
  reduce.h / reduce.hpp Reduce API (C / C++)
  runtime.h / runtime.hpp Device query (C / C++)
  fastcuda.h / fastcuda.hpp Umbrella headers
  version.hpp.in       Version template (cmake configure_file)

src/common/            Internal helpers
  cuda_check.h         CheckCuda, CeilDiv

src/gemm/              GEMM kernels + dispatch
  gemm_v1_naive.cu     V1 – one thread per element
  gemm_v2_shared.cu    V2 – shared-memory tiling (32×32)
  gemm_v3_register.cu  V3 – register blocking + float4
  gemm_v4_warp.cu      V4 – double-buffered prefetch
  gemm_v5_tf32.cu      V5 – TF32 Tensor Core (wmma)
  gemm_v6_hgemm.cu     V6 – FP16 HGEMM (wmma)
  gemm_api.cu          Public API dispatch, C wrappers
  gemm_internal.h      Internal launch declarations

src/reduce/            Reduce kernels + dispatch
  reduce_v0–v7_*.cu    8 progressive reduction kernels
  reduce_api.cu        Public API dispatch, C wrappers
  reduce_internal.h    Internal launch declarations

src/runtime/           Device query
  runtime.cu           QueryDevices, C wrappers

python/                Python bindings (pybind11)
  fastcuda_python.cpp  Module definition
  setup.py             Package metadata

examples/              Standalone executables
  gemm_example.cpp     V1–V5 SGEMM + CPU reference
  reduce_example.cpp   V0–V7 reduce + CPU reference

benchmarks/            Performance measurement
  bench_main.cu        GEMM (vs cuBLAS) + Reduce bandwidth
```

## Operator Inventory

| Operator | Versions | Status |
|----------|----------|--------|
| GEMM | V1–V6 (6 kernels) | Implemented |
| Reduce | V0–V7 (8 kernels) | Implemented |
| GEMV | — | Planned |
| SpMM | — | Planned |
| SpMV | — | Planned |
| FlashAttention | — | Planned |
