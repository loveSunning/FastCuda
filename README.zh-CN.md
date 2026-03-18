# FastCuda

如果这个项目对你有帮助，欢迎给一个 Star。

[English README](README.md)

FastCuda 是一个面向现代 NVIDIA GPU 的企业级 CUDA 算子库，用于手写内核开发、基准测试和性能分析。包含六种渐进式 GEMM 实现（从朴素到 Tensor Core HGEMM）、八种并行归约变体、cuBLAS 对比基准测试，以及 C / C++ / Python 接口。

## 你能得到什么

| 类别 | 内容 |
|------|------|
| **GEMM** | 6 个内核：V1 朴素 → V2 共享内存 → V3 寄存器分块 → V4 双缓冲 → V5 TF32 Tensor Core → V6 FP16 HGEMM |
| **Reduce** | 8 个内核：V0 基线 → V1 无分歧 → V2 无bank冲突 → V3 加载时累加 → V4 展开末warp → V5 完全展开 → V6 多元素累加 → V7 Warp Shuffle |
| **接口** | C 头文件、C++ 头文件、Python (pybind11) 绑定 |
| **基准测试** | 内置 cuBLAS 对比，输出 GFLOPS / 带宽 |
| **构建** | CMake，Windows + Linux，动态库 & 静态库 |
| **目标设备** | RTX 4090 (`sm_89`)、RTX 5060 (`sm_120`) |

## 快速开始

### 1. 配置

Windows：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```

Linux：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

### 2. 构建

```bash
cmake --build build --config Release -j
```

### 3. 运行

```bash
# GEMM 示例（运行 V1–V5 并与 CPU 参考实现对比）
./build/Release/fastcuda_gemm_example 512 512 512

# HGEMM 示例（FP16 输入，FP32 输出）
./build/Release/fastcuda_hgemm_example 512 512 512

# Reduce 示例（运行 V0–V7）
./build/Release/fastcuda_reduce_example 1048576

# 基准测试（GEMM vs cuBLAS + 全部 reduce 版本）
./build/Release/fastcuda_bench gemm all m=1024,n=1024,k=1024 fp32
./build/Release/fastcuda_bench gemm all m=1024,n=1024,k=1024 fp16
./build/Release/fastcuda_bench reduce all n=1048576
```

### 4. Python（可选）

```bash
cmake -S . -B build-py -DFASTCUDA_BUILD_PYTHON=ON \
	-Dpybind11_DIR="<你的-python-site-packages>/pybind11/share/cmake/pybind11"
cmake --build build-py --config Release -j
```

```python
import numpy as np
import os
import sys

sys.path.append("build-py/Release")
os.add_dll_directory(r"build-py/Release")

import fastcuda_python as fc

devices = fc.query_devices()
a = np.arange(64, dtype=np.float32).reshape(8, 8) / 32
b = np.arange(64, dtype=np.float32).reshape(8, 8) / 64

sgemm = fc.sgemm(int(fc.GemmAlgorithm.NaiveV1), 8, 8, 8, a, b)
hgemm = fc.hgemm(8, 8, 8, a.astype(np.float16), b.astype(np.float16))
reduced = fc.reduce_sum(int(fc.ReduceAlgorithm.BaselineV0), a.reshape(-1))
```

## GEMM 实现

| 版本 | 策略 | 分块/线程块 | 精度 |
|------|------|------------|------|
| V1 | 朴素（1线程→1元素） | 16×16 | FP32 |
| V2 | 共享内存分块 | 32×32, TK=32 | FP32 |
| V3 | 寄存器分块 + float4 | BM=64, BN=64, TM=4, TN=4 | FP32 |
| V4 | 双缓冲共享内存 + 预取 | BM=128, BN=128, TM=8, TN=8 | FP32 |
| V5 | Tensor Core (wmma, TF32) | 16×16×8 fragments | TF32→FP32 |
| V6 | Tensor Core HGEMM | 16×16×16 fragments | FP16→FP32 |

详见 [docs/gemm_optimization.zh-CN.md](docs/gemm_optimization.zh-CN.md)。

## Reduce 实现

| 版本 | 关键技术 |
|------|---------|
| V0 | 基线交错寻址（有 warp 分歧） |
| V1 | 步长索引（消除 warp 分歧） |
| V2 | 顺序寻址（消除 bank 冲突） |
| V3 | 加载时累加（块数减半） |
| V4 | 展开最后一个 warp（节省屏障指令） |
| V5 | 完全模板展开 |
| V6 | 多元素累加（每线程 8×） |
| V7 | Warp shuffle (`__shfl_down_sync`) |

详见 [docs/reduce_optimization.zh-CN.md](docs/reduce_optimization.zh-CN.md)。

## 对外头文件

| 头文件 | 语言 | 说明 |
|--------|------|------|
| `fastcuda/gemm.h` | C | SGEMM / TF32 / HGEMM 函数 |
| `fastcuda/gemm.hpp` | C++ | GEMM 命名空间 API |
| `fastcuda/reduce.h` | C | Reduce sum 函数 |
| `fastcuda/reduce.hpp` | C++ | Reduce 命名空间 API |
| `fastcuda/runtime.h` | C | 设备查询 |
| `fastcuda/runtime.hpp` | C++ | 设备查询与格式化 |
| `fastcuda/fastcuda.h` | C | 总头文件 |
| `fastcuda/fastcuda.hpp` | C++ | 总头文件 |

## 构建产物

- `fastcuda.dll` / `libfastcuda.so` — 动态库
- `fastcuda_static` — 静态库
- `fastcuda_bench` — 基准测试（链接 cuBLAS）
- `fastcuda_gemm_example` — GEMM 示例
- `fastcuda_hgemm_example` — HGEMM 示例
- `fastcuda_reduce_example` — Reduce 示例
- `fastcuda_python` — Python 模块（需 `FASTCUDA_BUILD_PYTHON=ON`）

## 项目结构

```text
.
├── include/fastcuda/   # 公开 C/C++ 头文件
├── src/
│   ├── gemm/           # 6 个 GEMM 内核 + API 分发
│   ├── reduce/         # 8 个 reduce 内核 + API 分发
│   ├── runtime/        # 设备查询
│   └── common/         # 内部辅助工具
├── python/             # pybind11 绑定
├── examples/           # GEMM 和 reduce 示例
├── benchmarks/         # 基准测试（cuBLAS 对比）
├── docs/               # 优化指南（中英文）
├── scripts/            # 构建、测试、分析辅助脚本
├── configs/            # 设备配置、benchmark 默认值
└── templates/          # 报告模板
```

## 计划中

以下算子已规划但暂未实现：

- GEMV（矩阵-向量乘法）
- SpMM（稀疏×稠密矩阵乘法）
- SpMV（稀疏矩阵-向量乘法）
- FlashAttention

## 环境要求

- Windows 或 Linux
- CMake ≥ 3.24
- CUDA Toolkit 12.8.x 或 13.0.x
- 支持 `sm_89` 或 `sm_120` 的 NVIDIA GPU（可配置）
- cuBLAS（用于基准测试对比）
- pybind11（可选，用于 Python 绑定）

## 常见问题

### 支持 FP16 和 Tensor Core 吗？

支持。V5 使用 TF32 Tensor Core，V6 实现完整的 FP16 HGEMM（通过 `nvcuda::wmma`）。

### 如何与 cuBLAS 对比？

运行 `fastcuda_bench gemm all ... fp32` 可比较 SGEMM/TF32 与 cuBLAS SGEMM，运行 `fastcuda_bench gemm all ... fp16` 可比较 HGEMM 与 cuBLAS HGEMM。

### 环境检查和性能分析脚本在哪里？

环境检查脚本在 `scripts/env`，benchmark 和 profiling 包装脚本在 `scripts/perf`，辅助 hook 在 `scripts/hooks`。

## 贡献

欢迎提交 Issue 和 PR。建议优先提交小而聚焦的 kernel、benchmark、文档或构建改进。

## 作者

由 FastCuda contributors 维护。

## 许可证

本项目采用 Apache License 2.0 许可，详见 [LICENSE](LICENSE)。

## 更多文档

- [GEMM 优化指南](docs/gemm_optimization.zh-CN.md) / [English](docs/gemm_optimization.md)
- [Reduce 优化指南](docs/reduce_optimization.zh-CN.md) / [English](docs/reduce_optimization.md)
- [Architecture](docs/architecture.md)
- [Agent Workflow](docs/agent-workflow.md)
- [AGENTS.md](AGENTS.md)
- [docs/codex-configuration.md](docs/codex-configuration.md)