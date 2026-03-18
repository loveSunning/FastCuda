# FastCuda

如果这个项目对你有帮助，欢迎给一个 Star。

[English README](README.md)

FastCuda 是一个面向现代 NVIDIA GPU 的轻量 CUDA 内核工作区，用于手写算子开发、基准测试和性能分析。

当前仓库已经提供可直接使用的 SGEMM 库，包含三种实现层级、一个 benchmark 可执行程序，以及一个最小示例程序。

## 你能得到什么

- 一个库里集成三种 SGEMM 路径：`naive`、`tiled_shared`、`register_blocked`
- 同时提供 C++ API 和 C ABI，便于下游项目集成
- 用于重复对比内核性能的 benchmark 入口
- 用于快速检查环境的设备查询工具
- 基于 CMake 的 Windows / Linux 构建方式
- 默认面向 RTX 4090（`sm_89`）和 RTX 5060（`sm_120`）

## 效果预览

### 示例程序输出

```text
m=256 n=256 k=256
algorithm=naive elapsed_ms=0.0340 max_abs_error=0.0000
algorithm=tiled_shared elapsed_ms=0.0359 max_abs_error=0.0000
algorithm=register_blocked elapsed_ms=0.0295 max_abs_error=0.0000
```

### Benchmark 输出

```text
operator=gemm
kernel=all
shape=m=256,n=256,k=256
requested_dtype=fp32
effective_dtype=fp32
device_count=1
device[0] name=NVIDIA GeForce RTX 5060 cc=12.0 global_mem_bytes=8546484224 sms=30
algorithm=naive
elapsed_ms=0.0336
max_abs_error=0.000000e+00
algorithm=tiled_shared
elapsed_ms=0.0359
max_abs_error=0.000000e+00
algorithm=register_blocked
elapsed_ms=0.0295
max_abs_error=0.000000e+00
status=ok
```

## 快速开始

### 1. 配置

Windows：

```powershell
cmd /c ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" && cmake -S . -B build -G "Visual Studio 17 2022" -A x64"
```

Linux：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

### 2. 构建

Windows：

```powershell
cmake --build build --config Release
```

Linux：

```bash
cmake --build build -j
```

### 3. 运行

运行示例程序：

```powershell
.\build\Release\fastcuda_gemm_example.exe 512 512 512
```

运行 benchmark 包装脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/perf/run-benchmark.ps1 -Operator gemm -Kernel all -Shape m=1024,n=1024,k=1024 -DType fp32
```

## 核心特点

- 手写 CUDA SGEMM 内核，分层清晰，便于学习和演进
- 同时暴露 C++ 和 C 接口，适合集成到其他程序中
- 通过 `scripts/perf/run-benchmark.ps1` 提供可重复 benchmark 流程
- 通过 `scripts/env/probe-env.ps1` 提供环境探测能力
- 提供 Nsight Compute / Nsight Systems 的分析入口
- 安装结果统一输出到 `bin`、`lib` 和 `include/fastcuda`

## 已实现功能

### SGEMM 算法层级

- `naive`：每个线程负责一个输出元素
- `tiled_shared`：基于 `16x16x16` 的共享内存分块实现
- `register_blocked`：基于 `64x64x16` 宏块和每线程 `4x4` 寄存器块的实现

### 对外头文件

- `fastcuda/gemm.hpp`：C++ SGEMM API
- `fastcuda/gemm_c.h`：供外部调用的 C ABI
- `fastcuda/runtime.hpp`：设备发现和设备摘要工具

### 构建产物

- 动态库：Windows 下为 `fastcuda.dll`，Linux 下为 `libfastcuda.so`
- 静态库：`fastcuda_static`
- Benchmark 可执行程序：`fastcuda_bench`
- 示例程序：`fastcuda_gemm_example`

## 适用场景

FastCuda 适合以下需求：

- 想分阶段学习 CUDA GEMM 优化
- 想构建一个 ABI 边界清晰的小型 CUDA 库
- 想在不引入大型框架的前提下做 kernel benchmark
- 想用简单 reference 实现校验结果正确性
- 想把 benchmark、profiling、环境检查集中在一个仓库里维护

## 环境要求

- Windows 或 Linux 主机
- CMake `>= 3.24`
- CUDA Toolkit `12.8.x` 或 `13.0.x`
- 与配置架构匹配的 NVIDIA GPU，默认值为 `89;120`

## 常见问题

### 支持哪些平台？

当前构建系统支持 Windows 和 Linux。

### 支持哪些 CUDA 版本？

当前 CMake 配置接受 CUDA `12.8.x` 和 `13.0.x`。

### 现在支持 FP16、BF16 或 Tensor Core 吗？

还不支持。目前对外提供的是 FP32 SGEMM 路径。

### 如何一次比较全部 GEMM 实现？

使用 benchmark 包装脚本并传入 `-Kernel all`，或者直接运行 `fastcuda_gemm_example`。

### 三种 GEMM 实现有什么区别？

`naive` 是以正确性为主的基线版本，`tiled_shared` 增加了共享内存分块，`register_blocked` 进一步增加了更大的宏块和每线程寄存器分块。

### 环境检查和性能分析脚本在哪里？

环境检查脚本在 `scripts/env`，benchmark 和 profiling 包装脚本在 `scripts/perf`，辅助 hook 在 `scripts/hooks`。

## 仓库结构

```text
.
|-- AGENTS.md
|-- benchmarks/
|-- configs/
|-- docs/
|-- examples/
|-- scripts/
|-- src/
`-- templates/
```

## 贡献

欢迎提交 Issue 和 PR。建议优先提交小而聚焦的 kernel、benchmark、文档或构建改进。

## 作者

由 FastCuda contributors 维护。

## 许可证

本项目采用 Apache License 2.0 许可，详见 [LICENSE](LICENSE)。

## 更多文档

- [AGENTS.md](AGENTS.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/agent-workflow.md](docs/agent-workflow.md)
- [docs/codex-configuration.md](docs/codex-configuration.md)