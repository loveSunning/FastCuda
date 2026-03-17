# Skill: benchmark-harness

Use this skill when creating or updating benchmark runners.

## Workflow

1. Define benchmark shapes and dtypes.
2. Add warmup and timed iterations.
3. Ensure outputs go to `artifacts/benchmarks/`.
4. Include environment snapshot reference.
5. Preserve machine-readable output.

## Minimum Metrics

- median latency
- p95 latency
- throughput if meaningful
