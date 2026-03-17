# Tools

FastCuda treats scripts under `scripts/` as stable tool entrypoints.

## Registered Tools

- `scripts/env/probe-env.ps1`
  - capture CUDA, NVCC, and profiler visibility
- `scripts/perf/run-benchmark.ps1`
  - standard benchmark artifact wrapper
- `scripts/perf/profile-ncu.ps1`
  - Nsight Compute command builder
- `scripts/perf/profile-nsys.ps1`
  - Nsight Systems command builder

## Tooling Rules

- prefer scripts over one-off shell snippets
- keep outputs in `artifacts/`
- if a script is used repeatedly, promote it to a tool entrypoint
