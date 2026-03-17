# FastCuda Architecture

## Goals

This workspace is designed for direct CUDA C++ operator development. The first
target kernels are GEMM and FlashAttention. The architecture optimizes for:

- repeatable kernel iteration
- explicit performance baselines
- low-friction profiling
- stable agent behavior
- reusable prompts and skills

## Build And Platform Targets

- build system: CMake
- host language standard: C++11
- CUDA language standard: C++11
- host platforms: Windows and Linux
- CUDA toolkits: 12.8.x and 13.0.x
- default GPU architectures:
  - RTX 4090 -> compute capability 8.9 -> `89`
  - RTX 5060 -> compute capability 12.0 -> `120`

## Design Principles

### 1. Stable execution surfaces

Agents should not invent ad hoc shell commands when a reusable script can be
used instead. Benchmarking, environment inspection, and profiling should route
through `scripts/`.

### 2. Rules before prompts

Instructions define non-negotiable behavior. Prompts are task entrypoints. A
prompt may be swapped; the rules should remain stable.

### 3. Skills are workflow modules

Each skill owns a repeatable sub-problem, such as:

- bringing up a benchmark harness
- profiling with Nsight tools
- checking occupancy or memory bandwidth assumptions

### 4. Hooks enforce hygiene

Hooks are used for lightweight guardrails:

- snapshot environment before a benchmark
- verify benchmark output path conventions
- capture metadata after profiling

### 5. Artifact-first performance work

Performance claims should produce files, not only terminal output:

- environment snapshots
- benchmark JSON/CSV
- profiler output locations
- optimization notes

## Control Plane

### `.codex/project.toml`

Central manifest that documents:

- default instructions
- available prompts
- custom agents
- registered skills
- hook entrypoints
- tool scripts

This is the top-level index for both humans and agents.

### `.codex/instructions/`

Contains durable operating rules:

- project scope
- CUDA coding rules
- benchmarking and profiling rules
- artifact conventions

### `.codex/prompts/`

Task-focused prompt files that agents can use as starting points:

- operator implementation
- performance optimization
- benchmark analysis
- environment investigation

### `.codex/agents/`

Custom agent role definitions. These are not model-specific internals; they are
project-facing role contracts. Example roles:

- kernel-architect
- kernel-optimizer
- perf-analyst
- env-investigator

### `.codex/skills/`

Reusable workflow packs with local instructions:

- benchmark design
- profiler runs
- roofline-style analysis
- CUDA environment checks

### `.codex/hooks/`

Hook configuration and mapping to executable scripts in `scripts/hooks/`.

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

### `configs/`

Machine-readable configs, benchmark presets, and output conventions.

### `templates/`

Reusable templates for notes, benchmark specs, and optimization reports.

## Recommended Workflow

1. Run environment probe.
2. Establish or refresh the baseline benchmark.
3. Implement a focused kernel change.
4. Re-run benchmark with metadata capture.
5. Profile only after a measurable change or a hypothesis mismatch.
6. Write optimization notes against data, not intuition alone.

## Future Expansion

- `src/` for kernels and harness code
- `benchmarks/` for operator-specific runners
- `results/` for benchmark and profiling artifacts
- `experiments/` for temporary tuning branches
