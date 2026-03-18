# GEMM 优化指南

本文档介绍 FastCuda 中实现的六个 GEMM（通用矩阵乘法）内核版本。
每个版本都在前一版本基础上演进，针对 GPU 存储/计算层级的不同层面进行优化。

计算操作为：

$$C = \alpha \cdot A \times B + \beta \cdot C$$

其中 $A$ 为 $M \times K$，$B$ 为 $K \times N$，$C$ 为 $M \times N$，
全部采用**行主序**存储。

---

## Version 1 – 朴素 SGEMM

**源码：** `src/gemm/gemm_v1_naive.cu`

### 思路

每个 CUDA 线程计算输出矩阵 $C$ 的一个元素。线程 $(row, col)$ 在整个 $K$ 维度上
迭代，每步从 $A$ 和 $B$ 各读取一个元素。

```
for i in 0..K:
    accum += A[row][i] * B[i][col]
C[row][col] = alpha * accum + beta * C[row][col]
```

### 启动配置

| 参数 | 值 |
|------|------|
| Block | 16 × 16 |
| Grid  | ⌈N/16⌉ × ⌈M/16⌉ |

### 性能特征

* **算术强度**为 $O(1)$——每次 FMA 都需要从全局内存读取两个浮点数。
* 大量冗余全局访存：$A$ 的每一行被加载 $N$ 次，$B$ 的每一列被加载 $M$ 次。
* 唯一优势是简洁、正确——用作 baseline。

---

## Version 2 – Shared Memory 分块 SGEMM

**源码：** `src/gemm/gemm_v2_shared.cu`

### 思路

将输出矩阵划分为 **TILE × TILE** 的块。对于每个块，沿 $K$ 维度按 TILE 大小步进。
每步中，线程协作将 A 和 B 的分块加载到 shared memory，再从 shared memory 计算。

### 关键细节

* **TILE = 32**，shared 数组声明为 `float As[TILE][TILE+1]`，+1 的 padding 消除
  bank conflict。
* 全局内存中的每个元素每个 tile 步仅加载一次，随后从 shared memory 中被复用 TILE 次。
* 算术强度提升至 $O(\text{TILE})$。

### 启动配置

| 参数 | 值 |
|------|------|
| Block | 32 × 32 (1024 线程) |
| Grid  | ⌈N/32⌉ × ⌈M/32⌉ |

---

## Version 3 – 寄存器分块 + 向量化 SGEMM

**源码：** `src/gemm/gemm_v3_register.cu`

### 思路

每个线程计算 **TM × TN**（4 × 4）大小的输出子块，使用寄存器驻留的累加器。
块大小提升至 **BM × BN × BK = 64 × 64 × 16**，全局加载使用 `float4` 实现
128 位事务。

### 关键细节

* **Block** = 16 × 16 = 256 个线程。
* 每个线程在内循环每步执行 16 次 FMA（4 × 4），实现更高的指令级并行。
* `float4` 向量化加载发起 128 位事务，充分利用内存带宽。
* 寄存器分块将每次 shared memory 读取的计算访存比提升至
  $O(\text{TM} \times \text{TN})$。

### Tile 配置

| 参数 | 值 | 含义 |
|------|------|------|
| BM | 64 | 块行数 |
| BN | 64 | 块列数 |
| BK | 16 | K-tile 宽度 |
| TM | 4  | 线程行数 |
| TN | 4  | 线程列数 |

---

## Version 4 – Warp 协同 + 预取 + 双缓冲 SGEMM

**源码：** `src/gemm/gemm_v4_warp.cu`

### 思路

CUDA Core SGEMM 的高性能完成态，组合三种技术：

1. **Warp 级 tiling**——块中的每个 warp 拥有输出 tile 的明确子区域。
2. **双缓冲 shared memory**——两组 shared 数组（`As[2]`、`Bs[2]`）在加载和计算之间
   交替。
3. **软件流水 / 预取**——当前 tile 正在从一个缓冲区计算时，下一个 tile 正在被加载到
   另一个缓冲区。

### Tile 配置

| 参数 | 值 | 含义 |
|------|------|------|
| BM | 128 | 块行数 |
| BN | 128 | 块列数 |
| BK | 8   | K-tile 宽度 |
| TM | 8   | 线程行数 |
| TN | 8   | 线程列数 |

### 优势

* 内存加载与计算重叠，隐藏大部分全局内存延迟。
* 更大 tile（128 × 128）摊销块调度开销，最大化数据复用。
* 每线程计算 8 × 8 = 64 个输出元素，计算密度极高。

---

## Version 5 – TF32 Tensor Core GEMM

**源码：** `src/gemm/gemm_v5_tf32.cu`

### 思路

输入和输出仍为 FP32，但内部矩阵乘累加在 Tensor Core 上使用 **TF32** 数据路径。
TF32 是一种 19 位格式（1 符号 + 8 指数 + 10 尾数），保留 FP32 指数范围，但减少了
尾数精度。

使用 CUDA `nvcuda::wmma` API，fragment 类型为 `wmma::precision::tf32`，
shape 为 **M=16, N=16, K=8**。

### 架构要求

* 需要 **SM 80+**（Ampere 及更新架构）。
* RTX 4090（SM 89）和 RTX 5060（SM 120）均支持。

### 性能说明

* Tensor Core 操作在相同矩阵规模下，吞吐量显著高于 CUDA Core。
* 由于尾数截断，数值结果与 IEEE FP32 存在微小差异。
  对于深度学习工作负载，这通常是可接受的。
* NVIDIA cuBLAS 通过 `CUBLAS_COMPUTE_32F_FAST_TF32` 暴露了等效模式。

---

## Version 6 – FP16 Tensor Core HGEMM

**源码：** `src/gemm/gemm_v6_hgemm.cu`

### 思路

最经典的 Tensor Core GEMM 路径。输入矩阵为 FP16（`half`），
累加和输出使用 FP32。这是 NVIDIA 自 Volta 以来在 Tensor Core 编程指南中
描述的标准混合精度模式。

使用 `nvcuda::wmma`，FP16 fragment，shape **M=16, N=16, K=16**。

### 架构要求

* 需要 **SM 70+**（Volta 及更新架构）。
* RTX 4090（SM 89）和 RTX 5060（SM 120）均支持。

### 性能说明

* FP16 Tensor Core GEMM 通常在现代 NVIDIA GPU 上实现最高吞吐量
  （比 FP32 Tensor Core 路径快 2×–4×，视架构而定）。
* 输出为 FP32 以避免累加误差。如有需要，下游代码可自行转换为 FP16。

---

## 版本对比矩阵

| 版本 | 精度 | 核心类型 | Shared Mem | 寄存器分块 | 双缓冲 | WMMA |
|------|------|----------|------------|------------|--------|------|
| V1 | FP32 | CUDA | — | — | — | — |
| V2 | FP32 | CUDA | ✓ | — | — | — |
| V3 | FP32 | CUDA | ✓ | ✓ (4×4) | — | — |
| V4 | FP32 | CUDA | ✓ (2×) | ✓ (8×8) | ✓ | — |
| V5 | TF32 | Tensor | ✓ | via WMMA | — | ✓ |
| V6 | FP16→FP32 | Tensor | ✓ | via WMMA | — | ✓ |

## 运行方法

```bash
# 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 示例（运行 v1-v5）
./build/fastcuda_gemm_example 1024 1024 1024

# Benchmark（全部版本 + cuBLAS 对比）
./build/fastcuda_bench gemm all m=1024,n=1024,k=1024
```

## 参考资料

* Mark Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA, 2007)
* NVIDIA CUDA C++ Programming Guide – Warp Matrix Functions (WMMA)
* NVIDIA cuBLAS 文档 – `cublasGemmEx`, `CUBLAS_COMPUTE_32F_FAST_TF32`
* [How to optimize in GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU)
* [Optimizing SGEMM on NVIDIA Turing GPUs](https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs)
