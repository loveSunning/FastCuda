# Prompt: Implement GEMM Kernel

Implement or revise a handwritten CUDA GEMM kernel.

Required output:

- kernel design summary
- launch configuration assumptions
- memory hierarchy strategy
- correctness risks
- benchmark plan

Constraints:

- use direct CUDA C++
- keep baseline and optimized variants separable
- make tile sizes and accumulation type explicit
- produce or update benchmark hooks if needed
