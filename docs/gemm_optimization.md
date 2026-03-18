# GEMM Optimization Guide

This document describes the six GEMM (General Matrix Multiply) kernel versions
implemented in FastCuda.  Each version builds on the previous one and
targets a specific level of the GPU memory / compute hierarchy.

The operation computed is:

$$C = \alpha \cdot A \times B + \beta \cdot C$$

where $A$ is $M \times K$, $B$ is $K \times N$, and $C$ is $M \times N$, all
stored in **row-major** order.

---

## Version 1 – Naive SGEMM

**Source:** `src/gemm/gemm_v1_naive.cu`

### Concept

Each CUDA thread computes exactly one element of $C$.  Thread $(row, col)$
iterates over the entire $K$ dimension, reading one element from $A$ and one
from $B$ per step.

```
for i in 0..K:
    accum += A[row][i] * B[i][col]
C[row][col] = alpha * accum + beta * C[row][col]
```

### Launch geometry

| Parameter | Value |
|-----------|-------|
| Block     | 16 × 16 |
| Grid      | ⌈N/16⌉ × ⌈M/16⌉ |

### Performance characteristics

* **Arithmetic intensity** is $O(1)$ – every FMA reads two floats from global
  memory.
* Many redundant global loads: each row of $A$ is loaded $N$ times; each column
  of $B$ is loaded $M$ times.
* The only advantage is simplicity and correctness – used as the baseline.

### When to use

Correctness reference and first debugging target.

---

## Version 2 – Shared Memory Block-Tiled SGEMM

**Source:** `src/gemm/gemm_v2_shared.cu`

### Concept

The output matrix is partitioned into **TILE × TILE** blocks.  For each
block, the $K$ dimension is walked in tiles of size TILE.  Before computing,
each thread cooperatively loads one element of the A-tile and one of the B-tile
into shared memory, then the entire block multiplies from shared memory.

```
for each K-tile:
    cooperatively load A[block_row..][tile_k..] → As
    cooperatively load B[tile_k..][block_col..] → Bs
    __syncthreads()
    for i in 0..TILE:
        accum += As[ty][i] * Bs[i][tx]
    __syncthreads()
```

### Key details

* **TILE = 32** with `+1` padding on shared arrays to eliminate bank conflicts.
* Each global-memory element is loaded once per tile step, then reused TILE
  times from shared memory.
* Arithmetic intensity rises to $O(\text{TILE})$.

### Launch geometry

| Parameter | Value |
|-----------|-------|
| Block     | 32 × 32 (1024 threads) |
| Grid      | ⌈N/32⌉ × ⌈M/32⌉ |

### Bank conflict mitigation

Shared-memory arrays are declared as `float As[TILE][TILE+1]`.  The extra
column shifts each row by one bank, preventing stride-32 accesses from hitting
the same bank.

---

## Version 3 – Register Blocking + Vectorised SGEMM

**Source:** `src/gemm/gemm_v3_register.cu`

### Concept

Each thread now computes a **TM × TN** (4 × 4) sub-tile of the output, using a
set of register-resident accumulators.  The block tile grows to
**BM × BN × BK = 64 × 64 × 16**, and global loads use `float4` for 128-bit
transactions.

```
float accum[TM][TN] = {0};

for each K-tile:
    cooperative float4 load → As, Bs
    __syncthreads()
    for bk in 0..BK:
        load a_frag[TM], b_frag[TN] from shared
        outer product: accum += a_frag * b_frag^T
    __syncthreads()
```

### Key details

* **Block size** = 16 × 16 = 256 threads.
* Each thread owns 16 FMAs per inner-loop step (4 × 4), reaching higher
  instruction-level parallelism.
* `float4` vectorized loads issue 128-bit transactions, saturating memory bus
  width.
* Register blocking increases the compute-to-memory ratio to
  $O(\text{TM} \times \text{TN})$ per shared-memory read.

### Tile configuration

| Param | Value | Meaning |
|-------|-------|---------|
| BM    | 64    | Block rows |
| BN    | 64    | Block cols |
| BK    | 16    | K-tile width |
| TM    | 4     | Thread rows |
| TN    | 4     | Thread cols |

---

## Version 4 – Warp Cooperative + Prefetch + Double-Buffer SGEMM

**Source:** `src/gemm/gemm_v4_warp.cu`

### Concept

This version is the high-performance completion of CUDA-core SGEMM.  Three
techniques are combined:

1. **Warp-level tiling** — each warp in the block owns a clearly defined
   sub-region of the output tile.
2. **Double-buffered shared memory** — two sets of shared arrays (`As[2]`,
   `Bs[2]`) alternate between loading and computing roles.
3. **Software pipelining / prefetch** — while the current tile is being
   computed from one buffer, the next tile is being loaded into the other
   buffer.

```
load tile 0 → As[0], Bs[0]
__syncthreads()

for tk = BK .. K+BK step BK:
    load tile tk → As[1-buf], Bs[1-buf]    -- prefetch
    compute from As[buf], Bs[buf]            -- current
    swap buf
    __syncthreads()
```

### Tile configuration

| Param | Value | Meaning |
|-------|-------|---------|
| BM    | 128   | Block rows |
| BN    | 128   | Block cols |
| BK    | 8     | K-tile width |
| TM    | 8     | Thread rows |
| TN    | 8     | Thread cols |

### Why this helps

* The overlap between memory loading and computation hides most of the global
  memory latency.
* Larger tile (128 × 128) amortises the block scheduling overhead and
  maximises data reuse.
* Each thread computes 8 × 8 = 64 output elements, giving very high compute
  density.

---

## Version 5 – TF32 Tensor Core GEMM

**Source:** `src/gemm/gemm_v5_tf32.cu`

### Concept

Input and output remain FP32, but the inner matrix multiply-accumulate is
performed using the **TF32** data path on Tensor Cores.  TF32 is a 19-bit
format (1 sign + 8 exponent + 10 mantissa) that retains the FP32 exponent
range while reducing mantissa precision.

The kernel uses CUDA's `nvcuda::wmma` API with fragment type
`wmma::precision::tf32` and shape **M=16, N=16, K=8**.

### Architecture requirements

* Requires **SM 80+** (Ampere and later).
* Supported on both RTX 4090 (SM 89) and RTX 5060 (SM 120).

### Block layout

| Param | Value |
|-------|-------|
| BM    | 64    |
| BN    | 64    |
| BK    | 8     |
| Threads | 128 (4 warps) |

Each warp owns one 16-row slice of the output and iterates over 4 column tiles
(4 × 16 = 64 columns).

### Performance notes

* Tensor Core operations deliver significantly higher throughput than CUDA cores
  for the same matrix size.
* The numerical result will differ slightly from IEEE FP32 due to the shorter
  mantissa.  For deep-learning workloads this is generally acceptable.
* NVIDIA's cuBLAS exposes an equivalent mode via
  `CUBLAS_COMPUTE_32F_FAST_TF32`.

---

## Version 6 – FP16 Tensor Core HGEMM

**Source:** `src/gemm/gemm_v6_hgemm.cu`

### Concept

The most natural Tensor Core GEMM path.  Input matrices are FP16 (`half`),
accumulation and output use FP32.  This follows the canonical mixed-precision
pattern described in NVIDIA's Tensor Core programming guides since Volta.

Uses `nvcuda::wmma` with FP16 fragments and shape **M=16, N=16, K=16**.

### Architecture requirements

* Requires **SM 70+** (Volta and later).
* Covers RTX 4090 (SM 89) and RTX 5060 (SM 120).

### Block layout

| Param | Value |
|-------|-------|
| BM    | 64    |
| BN    | 64    |
| BK    | 16    |
| Threads | 128 (4 warps) |

Shared-memory arrays padded by 8 to avoid bank conflicts on half-precision
loads.

### Performance notes

* FP16 Tensor Core GEMM typically achieves the highest throughput on modern
  NVIDIA GPUs (2× to 4× over FP32 Tensor Core paths, architecture-dependent).
* Output is FP32 to avoid accumulation error.  Downstream code can convert to
  FP16 if needed.

---

## Comparison matrix

| Version | Precision | Core type | Shared mem | Register blocking | Double buffer | WMMA |
|---------|-----------|-----------|------------|-------------------|---------------|------|
| V1      | FP32      | CUDA      | —          | —                 | —             | —    |
| V2      | FP32      | CUDA      | ✓          | —                 | —             | —    |
| V3      | FP32      | CUDA      | ✓          | ✓ (4×4)           | —             | —    |
| V4      | FP32      | CUDA      | ✓ (2×)     | ✓ (8×8)           | ✓             | —    |
| V5      | TF32      | Tensor    | ✓          | via WMMA          | —             | ✓    |
| V6      | FP16→FP32 | Tensor    | ✓          | via WMMA          | —             | ✓    |

## How to run

```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Example (runs v1-v5)
./build/fastcuda_gemm_example 1024 1024 1024

# Benchmark (all versions + cuBLAS)
./build/fastcuda_bench gemm all m=1024,n=1024,k=1024
```

## References

* Mark Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA, 2007)
* NVIDIA CUDA C++ Programming Guide – Warp Matrix Functions (WMMA)
* NVIDIA cuBLAS documentation – `cublasGemmEx`, `CUBLAS_COMPUTE_32F_FAST_TF32`
* [How to optimize in GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU)
* [Optimizing SGEMM on NVIDIA Turing GPUs](https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs)
