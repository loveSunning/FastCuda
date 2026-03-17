# CUDA Rules

## Implementation Rules

- Prefer direct CUDA C++ and NVCC-compatible code.
- State target GPU architecture assumptions explicitly.
- Separate correctness work from optimization work when possible.
- Keep kernel launch configuration visible and easy to inspect.
- Document tile sizes, memory movement strategy, and synchronization model.

## Review Checklist

- global memory coalescing
- shared memory bank conflict risk
- register pressure risk
- occupancy tradeoff
- synchronization correctness
- edge-tile behavior
- numeric stability and accumulation type

## GEMM-Specific Guidance

- define data layout and transpose assumptions up front
- state accumulation precision
- separate naive, tiled, and advanced variants

## FlashAttention-Specific Guidance

- state masking, causal mode, and supported sequence bounds
- state on-chip memory strategy for Q/K/V tiles
- make online softmax logic auditable

## Benchmark Discipline

- do not compare kernels across different shapes without naming shapes
- always record warmup, iteration count, and dtype
- always record GPU model and clock mode if known
