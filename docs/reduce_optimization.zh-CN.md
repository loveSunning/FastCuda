# Reduce 优化指南

本文档介绍 FastCuda 中实现的八个并行归约核函数版本，按四个优化类别组织。

计算的操作为：

$$\text{output} = \sum_{i=0}^{N-1} \text{input}[i]$$

所有核函数对 FP32 数组进行归约，输出单个标量和。

---

## 类别一 — 基础共享内存归约（V0 – V2）

这三个版本共享相同的基本结构：标准的块级共享内存归约。主要差异在于寻址模式
和冲突行为。

### V0 – 基线版本

**源文件：** `src/reduce/reduce_v0_baseline.cu`

```cuda
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

* 交错寻址导致 **warp 分歧** — 同一 warp 内的线程走不同的分支。
* 简单且正确；作为性能基线使用。

### V1 – 无分歧分支

**源文件：** `src/reduce/reduce_v1_no_divergence.cu`

```cuda
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    unsigned int index = 2 * s * tid;
    if (index < blockDim.x)
        sdata[index] += sdata[index + s];
    __syncthreads();
}
```

* 用步长索引替代取模：连续线程现在走相同分支 → **消除 warp 分歧**。
* 但步长模式会导致 **共享内存 bank 冲突**（同一 warp 内的线程以2的幂步长
  访问 bank）。

### V2 – 无 Bank 冲突（顺序寻址）

**源文件：** `src/reduce/reduce_v2_no_bank_conflict.cu`

```cuda
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

* 反转循环方向：步长从大开始，每步减半。
* 相邻线程访问相邻的共享内存位置 → **无 bank 冲突**。
* 第一步后一半线程变为空闲（对归约而言是自然的）。

---

## 类别二 — 计算融合与循环展开（V3 – V5）

这些版本超越了共享内存访问模式，开始减少指令和同步开销。

### V3 – 加载时累加

**源文件：** `src/reduce/reduce_v3_add_during_load.cu`

```cuda
sdata[tid] = input[i] + input[i + blockDim.x];
```

* 每个线程从全局内存加载 **两个** 元素，并在写入时进行累加。
* 所需的块数减半。
* 减少第一个归约步骤中的空闲比例。

### V4 – 展开最后一个 Warp

**源文件：** `src/reduce/reduce_v4_unroll_last_warp.cu`

```cuda
for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) { ... }
if (tid < 32) warp_reduce(sdata, tid);
```

* 当步长降至 32 或以下时，所有活跃线程在同一个 warp 内。Warp 内执行天然
  同步（SIMT）。
* 最后 6 个归约步骤展开，无需 `__syncthreads()`。
* 每个块节省 5 个屏障指令。

### V5 – 完全展开

**源文件：** `src/reduce/reduce_v5_completely_unroll.cu`

```cuda
template <unsigned int blockSize>
__global__ void reduce_v5_kernel(...) { ... }
```

* 将块大小作为模板参数 → 编译器在编译时完全展开每个归约步骤。
* 结合加载时累加（V3）和 warp 展开（V4）。
* 消除所有循环控制开销。

---

## 类别三 — 多元素累加（V6）

### V6 – 多元素累加

**源文件：** `src/reduce/reduce_v6_multi_add.cu`

```cuda
float val = 0;
for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
    unsigned int idx = base + e * blockDim.x;
    if (idx < n) val += input[idx];
}
sdata[tid] = val;
```

* 每个线程在进入块级归约前，先从全局内存累加 **ELEMS_PER_THREAD = 8** 个值。
* 增加每线程工作量，摊销启动和内存延迟。
* 块数减少 ELEMS_PER_THREAD 倍。
* 从"优化归约方式"转向"优化每个线程的工作量"。

---

## 类别四 — Warp Shuffle 归约（V7）

### V7 – Shuffle

**源文件：** `src/reduce/reduce_v7_shuffle.cu`

```cuda
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
```

* 使用 `__shfl_down_sync` 在 warp 内线程之间直接通过寄存器交换数据 —
  **无共享内存读写**。
* 跨 warp 的部分和通过小型共享内存数组收集。
* 结合多元素加载（每线程 8 个元素）以获得峰值吞吐量。

### 为什么这个版本不同

| 方面 | V0–V6（共享内存路径） | V7（shuffle 路径） |
|------|----------------------|-------------------|
| Warp 内数据移动 | 共享内存 → 寄存器 | 寄存器 → 寄存器（shuffle） |
| 共享内存流量 | 高（每个步骤） | 最小（仅跨 warp） |
| 同步点 | 每个步骤 | 仅跨 warp |
| 寄存器压力 | 低 | 略高 |

---

## 多块处理

所有核函数版本都是 **块级** 归约，每个块产生一个部分和。对于超过单个块覆盖
范围的输入，分发层（**`src/reduce/reduce_api.cu`**）递归调用同一核函数处理
部分和数组，直到得到单个结果。

---

## 对比矩阵

| 版本 | Warp 分歧 | Bank 冲突 | 加载时累加 | Warp 展开 | 模板展开 | 多元素累加 | Shuffle |
|------|-----------|-----------|-----------|-----------|----------|-----------|---------|
| V0 | ✗ | ✗ | — | — | — | — | — |
| V1 | ✓ | ✗ | — | — | — | — | — |
| V2 | ✓ | ✓ | — | — | — | — | — |
| V3 | ✓ | ✓ | ✓ | — | — | — | — |
| V4 | ✓ | ✓ | ✓ | ✓ | — | — | — |
| V5 | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| V6 | ✓ | ✓ | — | ✓ | ✓ | ✓ (8×) | — |
| V7 | ✓ | ✓ | — | — | — | ✓ (8×) | ✓ |

（✓ = 已解决/使用，✗ = 存在问题，— = 不适用）

## 运行方式

```bash
# 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 示例
./build/fastcuda_reduce_example 1048576

# 基准测试（所有版本）
./build/fastcuda_bench reduce all n=1048576
```

## 参考文献

* Mark Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA, 2007)
* NVIDIA CUDA C++ 编程指南 — Warp Shuffle 函数
* [How to optimize in GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU)
