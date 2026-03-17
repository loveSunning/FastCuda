# Skill: flashattention-kernel-design

Use this skill when building or reviewing FlashAttention kernels.

## Checklist

- sequence length assumptions
- head dimension assumptions
- masking mode
- online softmax recurrence
- memory residency strategy for Q, K, V, and partial outputs

## Focus

- correctness visibility first
- stable benchmark shapes second
- aggressive optimization only after baseline validation
