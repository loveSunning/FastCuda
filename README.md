# FastCuda

FastCuda is a Codex-agent-managed workspace for handwritten CUDA operator
development, with a near-term focus on:

- GEMM kernels
- FlashAttention kernels
- Benchmarking and regression tracking
- Performance analysis and environment inspection

## Build Targets

- build system: CMake
- host language standard: C++11
- CUDA language standard: C++11
- supported host platforms: Windows and Linux
- supported CUDA toolkits: 12.8.x and 13.0.x
- default CUDA architectures: `89;120`
  - `89` for GeForce RTX 4090
  - `120` for GeForce RTX 5060

The repository is organized as an "engineering operating system" for CUDA
kernel work. It separates:

- project rules and agent behavior
- reusable prompts and skills
- tool entrypoints for benchmarking and profiling
- hooks for environment snapshots and guardrails

## Repository Layout

```text
.
|-- .codex/
|   |-- project.toml
|   |-- instructions/
|   |-- prompts/
|   |-- agents/
|   |-- skills/
|   `-- hooks/
|-- docs/
|-- configs/
|-- scripts/
|   |-- env/
|   |-- perf/
|   `-- hooks/
`-- templates/
```

## Usage Model

1. Read `.codex/instructions/` before making code or benchmark changes.
2. Select a prompt in `.codex/prompts/` that matches the task.
3. Load one or more skills in `.codex/skills/` for focused execution.
4. Use scripts in `scripts/` as the stable execution surface.
5. Let hooks collect environment and benchmark metadata.

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

### Run Benchmark Stub

```powershell
powershell -ExecutionPolicy Bypass -File scripts/perf/run-benchmark.ps1 -Operator gemm -Kernel baseline
```

## Initial Priorities

- add GEMM kernel baselines
- add FlashAttention kernel baselines
- standardize benchmark output
- standardize profiling workflow
- capture GPU/compiler environment on each serious run

See [docs/architecture.md](E:/learning/cuda/FastCuda/docs/architecture.md) for
the full design.
