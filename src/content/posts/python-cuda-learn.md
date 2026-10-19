---
title: python调用cuda算子
published: 2026-08-01
description: ''
image: ''
tags: ["python", "算子开发", "AI Infra"]
category: '算子开发'
draft: false
lang: ''
slug: 算子
---

# Python 调用 CUDA 算子：PyTorch C++ Extension

## 1. 调用链路

Python 不能直接导入 `.cu` 源文件并调用其中的 `__global__` 函数。使用 PyTorch C++ Extension 时，调用过程为：

```text
Python (test.py)
  └─ extension.forward(x, residual, weight, eps)
       └─ pybind11：将 Python 函数名绑定到 C++ 函数
            └─ C++ Wrapper (fused_rmsnorm.cpp)：检查参数、分配输出
                 └─ CUDA Launcher (fused_rmsnorm_kernel.cu)：配置 Grid/Block/Stream
                      └─ __global__ Kernel：在 GPU 上执行计算
```

涉及三个源文件：

```text
project/
├── test.py                  # Python：编译、加载、调用、验证
├── fused_rmsnorm.cpp        # C++：函数接口与 pybind11 绑定
└── fused_rmsnorm_kernel.cu  # CUDA：Launcher 与 GPU Kernel
```

`.cpp` 和 `.cu` 编译成 Python 扩展动态库（Linux 下一般为 `.so`），随后由 Python 加载。

## 2. Python：编译并加载扩展

```python
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent

extension = load(
    name="fused_rmsnorm_ext",
    sources=[
        str(root / "fused_rmsnorm.cpp"),
        str(root / "fused_rmsnorm_kernel.cu"),
    ],
    verbose=True,
)

x = torch.randn(8, 4096, device="cuda", dtype=torch.float32)
residual = torch.randn_like(x)
weight = torch.ones(4096, device="cuda", dtype=torch.float32)

y, updated_residual = extension.forward(x, residual, weight, 1e-6)
```

`load()` 会调用本地编译器与 `nvcc`，生成并加载动态库。通常第一次运行需要编译；源码和编译参数不变时，可以复用缓存。这里的 `forward` 不是 PyTorch 自动生成的，而是下一步在 C++ 中显式注册的接口。

## 3. C++：Wrapper 与 pybind11 绑定

文件：`fused_rmsnorm.cpp`

```cpp
#include <torch/extension.h>
#include <vector>

// CUDA Launcher 的声明；定义放在 .cu 文件中
void fused_rmsnorm_cuda(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    at::Tensor& y,
    at::Tensor& updated_residual,
    float eps
);

std::vector<at::Tensor> forward(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double eps
) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat &&
                residual.scalar_type() == at::kFloat &&
                weight.scalar_type() == at::kFloat,
                "FP32 only");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() &&
                weight.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(x.dim() == 2 && residual.sizes() == x.sizes(),
                "x/residual must have shape [B, H]");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1),
                "weight must have shape [H]");
    TORCH_CHECK(x.get_device() == residual.get_device() &&
                x.get_device() == weight.get_device(),
                "all tensors must be on the same GPU");

    auto y = torch::empty_like(x);
    auto updated_residual = torch::empty_like(x);

    fused_rmsnorm_cuda(x, residual, weight, y, updated_residual,
                       static_cast<float>(eps));

    return {y, updated_residual};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Add + RMSNorm (FP32)");
}
```

关键点：

- `at::Tensor` / `torch::Tensor`：PyTorch 在 C++ 侧使用的 Tensor 类型。
- `torch::empty_like(x)`：按 `x` 的形状、dtype 和设备分配输出；若 `x` 在 GPU，输出也在 GPU。
- `m.def("forward", &forward)`：把 C++ 的 `forward` 暴露为 Python 的 `extension.forward`。
- Wrapper 负责接口检查和显存分配，本身不执行逐元素 CUDA 计算。

## 4. CUDA：Launcher 和 Kernel

文件：`fused_rmsnorm_kernel.cu`

```cpp
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void fused_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* y,
    float* updated_residual,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * hidden_size;
    float local_sum = 0.0f;

    // 1. 每个线程处理同一行中的若干元素
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        int idx = base + col;
        float r = x[idx] + residual[idx];
        updated_residual[idx] = r;
        local_sum += r * r;
    }

    // 2. Block 内规约：计算整行平方和
    __shared__ float shared[256];
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // 3. RMSNorm：整行共用相同的归一化系数
    float inv_rms = rsqrtf(shared[0] / hidden_size + eps);
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        int idx = base + col;
        y[idx] = updated_residual[idx] * inv_rms * weight[col];
    }
}

void fused_rmsnorm_cuda(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    at::Tensor& y,
    at::Tensor& updated_residual,
    float eps
) {
    int rows = static_cast<int>(x.size(0));
    int hidden_size = static_cast<int>(x.size(1));

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    fused_rmsnorm_kernel<<<rows, 256, 0, stream.stream()>>>(
        x.data_ptr<float>(),
        residual.data_ptr<float>(),
        weight.data_ptr<float>(),
        y.data_ptr<float>(),
        updated_residual.data_ptr<float>(),
        hidden_size,
        eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

两个需要区分的函数：

| 函数 | 执行位置 | 作用 |
|---|---|---|
| `fused_rmsnorm_cuda()` | CPU 端 | 取得 Tensor 地址、配置参数、发起 Kernel Launch |
| `__global__ fused_rmsnorm_kernel()` | GPU 端 | 通过线程并行执行算子 |

`<<<rows, 256, 0, stream.stream()>>>` 的四个参数分别是 Grid 中 Block 数、每个 Block 的线程数、动态 Shared Memory 字节数、CUDA Stream。上述实现每个 Block 处理一行。静态声明的 `__shared__ float shared[256]` 不需要通过第三个参数分配。

`data_ptr<float>()` 返回指向 GPU Tensor 底层存储的指针；**这里只传递地址，不把数据复制回 CPU**。Kernel Launch 通常是异步的，`C10_CUDA_KERNEL_LAUNCH_CHECK()` 检查启动错误，不代表 GPU 计算已经同步完成。

## 5. 正确性验证

在 `test.py` 的调用代码之后添加：

```python
# PyTorch Reference
r_ref = x + residual
y_ref = r_ref * torch.rsqrt(r_ref.square().mean(dim=-1, keepdim=True) + 1e-6)
y_ref = y_ref * weight

# GPU 操作默认异步；这里显式同步，便于定位错误
try:
    torch.cuda.synchronize()
    torch.testing.assert_close(updated_residual, r_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)
except Exception:
    raise

print("PASS")
```

先验证 `B=8, H=4096`，再增加 `H=4097` 这样的非对齐维度。当前实现只支持 **CUDA + FP32 + 连续内存布局 + 二维输入**，不能直接传 FP16 Tensor。不同规约顺序会引入浮点舍入差异，因此用容差比较，而非逐位相等。

运行前检查环境：

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python test.py
```

## 6. 常见问题

**为什么不能 `import fused_rmsnorm_kernel.cu`？** `.cu` 是 CUDA C++ 源文件，需要编译成可加载的动态库。PyTorch C++ Extension + pybind11 是一种调用方案；此外还有 ctypes、CUDA Driver API、CuPy RawKernel 等方式。

**Python 调用算子会导致 GPU 逐元素运行 Python 吗？** 不会。Python 只负责发起调用，实际计算由编译好的 GPU Kernel 执行。但对很小的算子，Python/C++ 调度和 Kernel Launch 开销可能占比较高。

**`getCurrentCUDAStream()` 为什么重要？** 使用 PyTorch 当前 Stream，保证扩展 Kernel 能与前后 PyTorch 操作按预期顺序执行；不要随意固定到默认 Stream。

**什么时候需要 `torch.cuda.synchronize()`？** 正确性排查和计时中可能需要；普通前向调用一般不应每次同步，否则会阻塞异步执行。做性能测试时优先使用 CUDA Events。

**后续优化是否要修改 Python 调用方式？** 通常不需要。只要对外函数签名保持不变，可以在 `.cu` 中替换 Reduction、Warp Shuffle、向量化等实现。

---

**要记住的接口关系：** `load()` 编译并加载模块 → `PYBIND11_MODULE` 注册 Python 接口 → C++ Wrapper 检查参数和分配 Tensor → `.cu` Launcher 获取 `data_ptr` 并配置执行参数 → `__global__` Kernel 在 GPU 上计算。
