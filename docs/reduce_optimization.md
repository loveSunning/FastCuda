# Reduce Optimization Guide

This document describes the eight parallel reduction kernel versions implemented
in FastCuda, grouped into four optimisation categories.

The operation computed is:

$$\text{output} = \sum_{i=0}^{N-1} \text{input}[i]$$

All kernels reduce an FP32 array to a single scalar sum.

---

## Category 1 – Basic Shared-Memory Reduction (V0 – V2)

These three versions share a common structure: standard block-level shared-
memory reduction.  The main differences lie in addressing patterns and conflict
behaviour.

### V0 – Baseline

**Source:** `src/reduce/reduce_v0_baseline.cu`

```cuda
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

* Interleaved addressing causes **warp divergence** – threads within the same
  warp take different branches.
* Simple and correct; serves as the performance baseline.

### V1 – No Divergent Branching

**Source:** `src/reduce/reduce_v1_no_divergence.cu`

```cuda
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    unsigned int index = 2 * s * tid;
    if (index < blockDim.x)
        sdata[index] += sdata[index + s];
    __syncthreads();
}
```

* Replaces modulo with a strided index: contiguous threads now take the same
  branch → **no warp divergence**.
* However the stride pattern causes **shared-memory bank conflicts** (threads
  in the same warp access banks with a power-of-2 stride).

### V2 – No Bank Conflict (Sequential Addressing)

**Source:** `src/reduce/reduce_v2_no_bank_conflict.cu`

```cuda
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

* Reverses the loop direction: stride starts large and halves each step.
* Adjacent threads access adjacent shared-memory locations → **no bank
  conflicts**.
* Half the threads become idle after the first step (natural for reductions).

---

## Category 2 – Computation Fusion and Loop-Unrolling (V3 – V5)

These versions move beyond shared-memory access patterns and start reducing
instruction and synchronisation overhead.

### V3 – Add During Load

**Source:** `src/reduce/reduce_v3_add_during_load.cu`

```cuda
sdata[tid] = input[i] + input[i + blockDim.x];
```

* Each thread loads **two** elements from global memory and adds them during
  the store.
* Halves the number of blocks required.
* Reduces the idle fraction in the first reduction step.

### V4 – Unroll Last Warp

**Source:** `src/reduce/reduce_v4_unroll_last_warp.cu`

```cuda
for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) { ... }
if (tid < 32) warp_reduce(sdata, tid);
```

* When the stride drops to 32 or below, all active threads are within a
  single warp.  Intra-warp execution is inherently synchronous (SIMT).
* The last 6 reduction steps are unrolled without `__syncthreads()`.
* Saves 5 barrier instructions per block.

### V5 – Completely Unrolled

**Source:** `src/reduce/reduce_v5_completely_unroll.cu`

```cuda
template <unsigned int blockSize>
__global__ void reduce_v5_kernel(...) { ... }
```

* Templates the kernel on block size → the compiler fully unrolls every
  reduction step at compile time.
* Combines add-during-load (V3) and warp unroll (V4).
* Removes all loop control overhead.

---

## Category 3 – Multi-Element Accumulation (V6)

### V6 – Multi Add

**Source:** `src/reduce/reduce_v6_multi_add.cu`

```cuda
float val = 0;
for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
    unsigned int idx = base + e * blockDim.x;
    if (idx < n) val += input[idx];
}
sdata[tid] = val;
```

* Each thread first accumulates **ELEMS_PER_THREAD = 8** values from global
  memory before entering the block-level reduction.
* Increases per-thread work, amortising launch and memory latency.
* Reduces the number of blocks by a factor of ELEMS_PER_THREAD.
* Transitions from "optimise how we reduce" to "optimise how much work each
  thread does."

---

## Category 4 – Warp Shuffle Reduction (V7)

### V7 – Shuffle

**Source:** `src/reduce/reduce_v7_shuffle.cu`

```cuda
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
```

* Uses `__shfl_down_sync` to exchange data between threads within a warp
  directly through registers – **no shared-memory read/write**.
* Cross-warp partial sums are collected in a small shared-memory array.
* Combined with multi-element load (8 elements per thread) for peak
  throughput.

### Why this is different

| Aspect | V0–V6 (shared-memory path) | V7 (shuffle path) |
|--------|----------------------------|--------------------|
| Intra-warp data movement | shared memory → register | register → register (shuffle) |
| Shared-memory traffic | high (every step) | minimal (cross-warp only) |
| Synchronisation points | every step | cross-warp only |
| Register pressure | low | slightly higher |

---

## Multi-block handling

All kernel versions are **block-level** reductions producing one partial sum
per block.  For inputs larger than one block's coverage, the dispatch layer
(**`src/reduce/reduce_api.cu`**) recursively invokes the same kernel on the
partial-sums array until a single result remains.

---

## Comparison matrix

| Version | Warp divergence | Bank conflict | Add during load | Warp unroll | Template unroll | Multi-add | Shuffle |
|---------|-----------------|---------------|-----------------|-------------|-----------------|-----------|---------|
| V0 | ✗ | ✗ | — | — | — | — | — |
| V1 | ✓ | ✗ | — | — | — | — | — |
| V2 | ✓ | ✓ | — | — | — | — | — |
| V3 | ✓ | ✓ | ✓ | — | — | — | — |
| V4 | ✓ | ✓ | ✓ | ✓ | — | — | — |
| V5 | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| V6 | ✓ | ✓ | — | ✓ | ✓ | ✓ (8×) | — |
| V7 | ✓ | ✓ | — | — | — | ✓ (8×) | ✓ |

(✓ = addressed / used, ✗ = problem present, — = not applicable)

## How to run

```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Example
./build/fastcuda_reduce_example 1048576

# Benchmark (all versions)
./build/fastcuda_bench reduce all n=1048576
```

## References

* Mark Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA, 2007)
* NVIDIA CUDA C++ Programming Guide – Warp Shuffle Functions
* [How to optimize in GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU)
