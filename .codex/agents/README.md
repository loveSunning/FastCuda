# Custom Agents

Custom agents are role contracts for the project.

Recommended routing:

- `kernel-architect`: decomposition, tiling, memory strategy
- `kernel-optimizer`: narrow performance improvements
- `perf-analyst`: interpret benchmark and profiler outputs
- `env-investigator`: inspect CUDA and profiling environment issues

Keep role boundaries narrow. If a task changes from design to measurement, hand
off from architect or optimizer to perf analyst.
