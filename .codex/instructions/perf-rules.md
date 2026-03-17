# Performance Rules

## Benchmarking

- Use a stable runner script instead of one-off shell commands.
- Save benchmark outputs under `artifacts/benchmarks/`.
- Include shape, dtype, GPU, and timestamp in result filenames.
- Report median and p95 latency when possible.

## Profiling

- Run Nsight Compute for kernel-level bottlenecks.
- Run Nsight Systems for timeline and overlap questions.
- Do not profile before a reproducible benchmark exists.

## Analysis

- Distinguish measured facts from hypotheses.
- Tie each optimization idea to a limiting resource:
  - memory bandwidth
  - latency hiding
  - occupancy
  - instruction throughput
  - synchronization overhead

## Regression Handling

- Preserve last-known-good benchmark artifacts.
- Record both absolute numbers and relative delta.
- If environment changed, treat the comparison as provisional.
