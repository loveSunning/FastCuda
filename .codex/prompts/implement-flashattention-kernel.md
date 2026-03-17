# Prompt: Implement FlashAttention Kernel

Implement or revise a handwritten CUDA FlashAttention kernel.

Required output:

- algorithm variant summary
- tiling and memory residency plan
- online softmax handling notes
- masking and boundary assumptions
- benchmark and profiling plan

Constraints:

- use direct CUDA C++
- prioritize auditable correctness first
- isolate numerical stability decisions
- document sequence length and head dimension assumptions
