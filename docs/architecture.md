# FastCuda Architecture

## Goals

This workspace is designed for direct CUDA C++ operator development. The first
target kernels are GEMM and FlashAttention. The architecture optimizes for:

- repeatable kernel iteration
- explicit performance baselines
- low-friction profiling
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
