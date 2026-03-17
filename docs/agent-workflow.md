# Agent Workflow

## Control Flow

```text
instructions -> prompt -> agent role -> skill bundle -> tool script -> hook -> artifact
```

## Recommended Task Recipes

### 1. New GEMM Kernel

- load `project-scope.md`, `cuda-rules.md`, `perf-rules.md`
- use prompt `implement-gemm-kernel.md`
- assign role `kernel-architect`
- add skills `gemm-kernel-design` and `benchmark-harness`
- run hook `pre-benchmark` before serious measurement

### 2. New FlashAttention Kernel

- load the instruction set
- use prompt `implement-flashattention-kernel.md`
- assign role `kernel-architect`
- add skills `flashattention-kernel-design` and `benchmark-harness`

### 3. Perf Regression

- use prompt `optimize-kernel.md`
- assign role `perf-analyst` first
- add skills `ncu-profiling` and `roofline-analysis`
- use `nsys-timeline` only if overlap or launch behavior is suspicious

### 4. Machine Setup or Failure Diagnosis

- use prompt `investigate-environment.md`
- assign role `env-investigator`
- add skill `cuda-env-audit`

## Rules of Engagement

- do not skip baseline benchmarking
- do not mix environment debugging with kernel optimization unless required
- do not make performance claims without artifacts
