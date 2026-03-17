# Project Scope

## Mission

Develop handwritten CUDA operators with priority on correctness, measurable
performance, and reproducible analysis.

## Near-Term Kernel Targets

- GEMM
- FlashAttention

## Standard Task Order

1. clarify target operator and constraints
2. inspect environment and architecture assumptions
3. implement or modify the kernel
4. benchmark against a baseline
5. profile if the result is slower than expected or unexplained
6. record findings in artifacts

## Non-Goals

- hidden magic wrappers around CUDA code
- vague optimization claims without benchmark data
- mixing unrelated experiments into one change set
