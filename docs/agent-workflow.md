# Agent Workflow

## Control Flow

```text
AGENTS.md -> .codex/config.toml -> subagent or skill -> script wrapper -> artifact
```

## Recommended Task Recipes

### 1. New GEMM Kernel

- read `AGENTS.md`
- use subagent `kernel-architect` if design decomposition is non-trivial
- use skills `gemm-kernel-design` and `benchmark-harness`
- use the brief in `docs/prompts/implement-gemm-kernel.md` if a reusable prompt starter helps

### 2. New FlashAttention Kernel

- read `AGENTS.md`
- use subagent `kernel-architect`
- use skills `flashattention-kernel-design` and `benchmark-harness`
- use the brief in `docs/prompts/implement-flashattention-kernel.md`

### 3. Perf Regression

- use subagent `perf-analyst` first
- add skills `ncu-profiling` and `roofline-analysis`
- use `nsys-timeline` only for overlap, stream, or launch questions
- use the brief in `docs/prompts/optimize-kernel.md` if needed

### 4. Machine Setup Or Failure Diagnosis

- use subagent `env-investigator`
- add skill `cuda-env-audit`
- use the brief in `docs/prompts/investigate-environment.md`

## Rules Of Engagement

- do not skip baseline benchmarking
- do not mix environment debugging with kernel optimization unless required
- do not make performance claims without artifacts
